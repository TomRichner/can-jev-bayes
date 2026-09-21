"""Bounded, auditable API batching with durable response replay and budget reserves."""

import asyncio
import hashlib
import json
import sqlite3
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values

from .prompts import MODEL, TASK, validate_answer

PRICE = 0.042 / 1_000_000
MAX_INPUT = 65536


def encode(value):
    return zlib.compress(
        json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    )


def decode(value):
    return json.loads(zlib.decompress(value))


class BudgetExceeded(RuntimeError):
    pass


class JevClient:
    def __init__(
        self,
        run_dir,
        *,
        cap=18.0,
        rate=5.0,
        concurrency=4,
        batch_size=16,
        key=None,
        transport=None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / "ledger.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
          id TEXT PRIMARY KEY, payload BLOB NOT NULL, response BLOB,
          input_tokens INTEGER, seconds REAL, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts (
          id INTEGER PRIMARY KEY, request_id TEXT, status TEXT, reserved REAL,
          cost REAL, input_tokens INTEGER, seconds REAL, created TEXT);
        CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, record BLOB NOT NULL);
        """)
        self.cap, self.rate, self.batch_size = cap, rate, batch_size
        self.sem = asyncio.Semaphore(concurrency)
        self.rate_lock = asyncio.Lock()
        self.next_time = 0.0
        secret = key if key is not None else dotenv_values(".env").get("jev_key")
        if not secret:
            raise ValueError("Missing jev_key in .env")
        self.http = httpx.AsyncClient(
            base_url="https://api.typesafe.ai",
            headers={"Authorization": f"Bearer {secret}"},
            timeout=30,
            follow_redirects=False,
            transport=transport,
            limits=httpx.Limits(max_connections=concurrency),
        )

    async def close(self):
        self.write_accounting()
        await self.http.aclose()
        self.db.close()

    def accounting(self):
        row = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(cost),0),COALESCE(SUM(reserved),0),SUM(CASE WHEN status!='200' THEN 1 ELSE 0 END) FROM attempts"
        ).fetchone()
        decisions = self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        return {
            "http_requests": row[0],
            "input_tokens": row[1],
            "known_cost_usd": row[2],
            "uncertain_reserved_usd": row[3],
            "error_attempts": row[4] or 0,
            "inference_decisions": decisions,
            "cap_usd": self.cap,
            "preparation_cost_usd": 0.00015456,
            "model": MODEL,
            "price_per_million_input": 0.042,
            "batch_size": self.batch_size,
            "updated_utc": datetime.now(UTC).isoformat(),
        }

    def write_accounting(self):
        path = self.run_dir / "accounting.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.accounting(), indent=2))
        temp.replace(path)

    def record(self, decision_id, value):
        self.db.execute(
            "INSERT OR IGNORE INTO decisions VALUES (?,?)", (decision_id, encode(value))
        )
        self.db.commit()

    def get_record(self, decision_id):
        row = self.db.execute(
            "SELECT record FROM decisions WHERE id=?", (decision_id,)
        ).fetchone()
        return None if row is None else decode(row[0])

    async def _request(self, jobs):
        payload = {
            "model": MODEL,
            "state": TASK,
            "questions": {f"q{i}": job["question"] for i, job in enumerate(jobs)},
        }
        identity = {"ids": [j["id"] for j in jobs], "payload": payload}
        request_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        old = self.db.execute(
            "SELECT response,seconds FROM requests WHERE id=?", (request_id,)
        ).fetchone()
        if old and old[0] is not None:
            return self._unpack(jobs, decode(old[0]), request_id, old[1])
        now = datetime.now(UTC).isoformat()
        self.db.execute(
            "INSERT OR IGNORE INTO requests(id,payload,created) VALUES(?,?,?)",
            (request_id, encode(payload), now),
        )
        self.db.commit()
        async with self.sem:
            for attempt in range(4):
                async with self.rate_lock:
                    delay = self.next_time - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    self.next_time = time.monotonic() + 1 / self.rate
                budget = self.accounting()
                reserve = MAX_INPUT * PRICE
                if (
                    budget["known_cost_usd"]
                    + budget["uncertain_reserved_usd"]
                    + reserve
                    + 0.00015456
                    > self.cap
                ):
                    raise BudgetExceeded(
                        "Project API spending cap reached; results are checkpointed"
                    )
                cur = self.db.execute(
                    "INSERT INTO attempts(request_id,status,reserved,cost,input_tokens,created) VALUES(?,?,?,?,?,?)",
                    (request_id, "pending", reserve, 0, 0, now),
                )
                attempt_id = cur.lastrowid
                self.db.commit()
                start = time.perf_counter()
                status, retry_after = "transport_error", 0.0
                try:
                    response = await self.http.post("/v1/systemone", json=payload)
                    elapsed = time.perf_counter() - start
                    status = str(response.status_code)
                    if response.status_code == 200:
                        body = response.json()
                        usage = body.get("usage", {}).get("input_tokens")
                        if not isinstance(usage, int) or usage < 1:
                            raise ValueError("Missing valid API token accounting")
                        self.db.execute(
                            "UPDATE attempts SET status=?,reserved=0,cost=?,input_tokens=?,seconds=? WHERE id=?",
                            (status, usage * PRICE, usage, elapsed, attempt_id),
                        )
                        self.db.execute(
                            "UPDATE requests SET response=?,input_tokens=?,seconds=? WHERE id=?",
                            (encode(body), usage, elapsed, request_id),
                        )
                        self.db.commit()
                        return self._unpack(jobs, body, request_id, elapsed)
                    # Failed HTTP requests may have consumed inference; retain conservative reserve.
                    self.db.execute(
                        "UPDATE attempts SET status=?,seconds=? WHERE id=?",
                        (status, elapsed, attempt_id),
                    )
                    self.db.commit()
                    if (
                        response.status_code not in {429, 529}
                        and response.status_code < 500
                    ):
                        raise RuntimeError(
                            f"Non-retryable Jev HTTP {status}; no credentials logged"
                        )
                    try:
                        retry_after = float(response.headers.get("retry-after", "0"))
                    except ValueError:
                        retry_after = 0
                    if response.status_code in {429, 529}:
                        self.rate = max(1, self.rate * 0.8)
                except httpx.TransportError:
                    self.db.execute(
                        "UPDATE attempts SET status=?,seconds=? WHERE id=?",
                        (status, time.perf_counter() - start, attempt_id),
                    )
                    self.db.commit()
                if attempt == 3:
                    raise RuntimeError(
                        f"Jev failed after four attempts ({status}); checkpoint retained"
                    )
                # Deterministic stagger avoids consuming scientific random streams.
                jitter = (int(request_id[:6], 16) % 1000) / 1000
                await asyncio.sleep(max(retry_after, min(30, 2**attempt + jitter)))

    def _unpack(self, jobs, body, request_id, elapsed):
        if body.get("model") != MODEL:
            raise ValueError("Returned model version mismatch")
        answers = body.get("answers", {})
        if set(answers) != {f"q{i}" for i in range(len(jobs))}:
            raise ValueError("Question response IDs mismatch")
        result = []
        for i, job in enumerate(jobs):
            answer = answers[f"q{i}"]
            ids = list(job["question"]["criteria"])
            action, probabilities, mass = validate_answer(answer, ids)
            record = {
                "action": action,
                "probabilities": probabilities.tolist(),
                "raw_answer": answer,
                "probability_mass": mass,
                "request_id": request_id,
                "latency_seconds": elapsed,
                "question_sha256": hashlib.sha256(
                    json.dumps(job["question"], sort_keys=True).encode()
                ).hexdigest(),
            }
            self.record(job["id"], record)
            result.append(record)
        return result

    async def evaluate(self, jobs):
        results = [self.get_record(job["id"]) for job in jobs]
        for job, result in zip(jobs, results):
            if result is not None:
                expected = hashlib.sha256(
                    json.dumps(job["question"], sort_keys=True).encode()
                ).hexdigest()
                if result.get("question_sha256") != expected:
                    raise ValueError(
                        "Decision ID was reused with changed scientific input"
                    )
        missing = [(i, j) for i, j in enumerate(jobs) if results[i] is None]
        batches, current, size = [], [], 0
        for entry in missing:
            length = len(json.dumps(entry[1]["question"]).encode())
            if length > 28000:
                raise ValueError(
                    "Single question exceeds conservative context allowance"
                )
            if current and (len(current) >= self.batch_size or size + length > 54000):
                batches.append(current)
                current, size = [], 0
            current.append(entry)
            size += length
        if current:
            batches.append(current)

        async def one(batch):
            values = await self._request([j for _, j in batch])
            for (i, _), value in zip(batch, values):
                results[i] = value

        # Limit scheduled work as well as HTTP concurrency; fail without orphaned requests.
        for offset in range(0, len(batches), 8):
            tasks = [asyncio.create_task(one(b)) for b in batches[offset : offset + 8]]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            self.write_accounting()
        return results

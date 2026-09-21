import argparse
import asyncio
from pathlib import Path

from .client import JevClient
from .experiments import diagnostics, manifest, merge_episodes, online, preflight


async def run(args):
    manifest(args.run_dir)
    client = None
    try:
        if args.command in {"preflight", "diagnostics"} or (
            args.command == "online" and args.only == "jev"
        ):
            client = JevClient(
                args.run_dir,
                cap=args.cap,
                rate=args.rate,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
            )
        if args.command == "preflight":
            await preflight(client)
        elif args.command == "diagnostics":
            await diagnostics(client, args.experiment, n=args.n or 100)
        elif args.command == "online":
            await online(
                client, args.run_dir, args.experiment, only=args.only, n_override=args.n
            )
            merge_episodes(args.run_dir)
        elif args.command == "report":
            from .report import generate_report

            merge_episodes(args.run_dir)
            result = generate_report(
                Path(args.run_dir),
                output_dir=Path(args.output) if args.output else None,
            )
            print(result)
    finally:
        if client:
            await client.close()


def main():
    parser = argparse.ArgumentParser(description="Jev Bayesian decision experiments")
    parser.add_argument(
        "command", choices=["preflight", "diagnostics", "online", "report"]
    )
    parser.add_argument("--experiment", default="pilot")
    parser.add_argument("--run-dir", default="artifacts/overnight_v2")
    parser.add_argument("--only", choices=["jev", "baselines"], default="jev")
    parser.add_argument("--n", type=int)
    parser.add_argument("--cap", type=float, default=18.0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

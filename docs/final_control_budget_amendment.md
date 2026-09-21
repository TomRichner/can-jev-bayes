# Final control budget amendment

Recorded after E9 completed and before any E10 or batching-control API calls. This changes only the operational spending cap; scientific payloads, sample sizes, and analysis commitments remain frozen.

The user authorized up to $200 for the project and reported $20 in current Jev credits. The initial $18 cap was a conservative project default, not a user-imposed maximum. Through E9, estimated project usage plus conservative retry reserves is **$17.262958902**.

E7 compared Noul reward forecasts with Choice best-arm forecasts, so target and primitive were confounded. The pre-specified E10 control crosses primitives and targets, adding known-probability controls. Its final full-context payloads are longer than the initial rough estimate: a regression of measured E7 request tokens against question counts and serialized lengths projects **$0.90847** for the 16,560 E10 questions. Adding a 30% margin and $0.10 for the 1,920-decision standalone-versus-batched validation gives approximately **$18.54** total.

Set the cap to **$19** for these final two controls, staying below the reported $20 credits and the user's broader authorization. No credits are purchased automatically. All prior ledgers and uncertain charges still count toward the cap. The actual account balance has not been queried; costs are derived from reported token usage at the published rate.

Do not reduce N, remove difficult conditions, or change prompts to fit the earlier estimate. The same bounded retries, source-hash checks, one-live-writer rule, and complete-record requirements apply. If the updated cap or account balance prevents completion, preserve the data and explicitly identify unfinished cells.

# Building the research summary

The source is [research_summary.md](../research_summary.md); its three citation keys were confirmed through zoteus on September 22, 2026. The accompanying [CSL-JSON bibliography](../research_summary.references.json) was exported from Zotero. Keep both files together. The summary synthesizes existing results; it adds no experiment or reanalysis.

Run these commands from the repository root with the installed skills:

```bash
# Word, with live Zotero citation fields. Zotero and Better BibTeX must be running.
~/.agents/skills/md2doc/scripts/md2doc.sh research_summary.md

# PDF, resolving the bundled bibliography without requiring Zotero.
~/.agents/skills/md2pdf/scripts/md2pdf.sh research_summary.md -- --citeproc

# Companion tables: one table for each experiment, plus batching validation.
~/.agents/skills/md2doc/scripts/md2doc.sh experiment_tables.md
~/.agents/skills/md2pdf/scripts/md2pdf.sh experiment_tables.md
```

After creating Word output, use Zotero's **Refresh** in Word to display and update the live citation fields. The PDF command requires Pandoc and LuaLaTeX. Both commands use the same Markdown source; the PDF command needs `--citeproc` to render citations and the reference list.

Validation for this summary checks numeric source records, citation-key resolution, local links, bibliography JSON, and Pandoc parsing with citation processing. It does not rerun experiments or establish visual QA of a final Word/PDF export.

The [companion tables](../experiment_tables.md) use ordinary two-column Pandoc tables, with the six requested fields as rows for readable portrait output. Prompt excerpts were checked against the question builders; they are not presented as complete archived payloads. Table validation checks parsing, the eleven-table count, six fields per table, and local links. The tables cite local experiment records and need no separate bibliography.

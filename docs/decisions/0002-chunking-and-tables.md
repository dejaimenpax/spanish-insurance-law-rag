# 0002 — Chunking and tables

**Status:** accepted · **Date:** 2026-10-02

## Decision

* **Version in force.** Each BOE block carries every wording it has had. The parser keeps the
  version with the latest `fecha_vigencia` not after the ingestion date (versions are not always
  in order, and a version may be scheduled for the future) and drops blocks whose
  `fecha_caducidad` has passed (e.g. the pre-2016 motor-accident tables).
* **Unit of retrieval.** An article that fits in about 800 tokens is one chunk. Longer articles
  are split at their numbered paragraphs (apartados), which is how they are cited. A marker only
  opens a new apartado when it continues the sequence (`1.`, `2.`, `3.`…), so enumerations
  inside an apartado ("1.º …") do not split it. Oversized pieces are packed into windows of
  whole paragraphs; tables are split by rows repeating their header.
* **Context.** The text that is embedded and indexed lexically starts with the norm, its
  hierarchy (book › title › chapter › section) and the article label and rubric. The text that
  is shown and cited is the provision's own wording.
* **Editorial notes.** BOE notes inside the consolidated text (for example, amounts updated by a
  later resolution) are kept and labelled `[Nota de la edición consolidada: …]`. Amendment
  footnotes ("Se modifica por…") are stored as metadata, not indexed.
* **Tables.** Tables with readable content (amounts per category, codes and descriptions) are
  rendered as Markdown. Tables that are at least 80 % numeric and have 24 or more value cells,
  such as the actuarial and loss-of-earnings tables of the motor-accident valuation system, are
  replaced by a stub that names the table, lists its columns and row count and points to the
  official source. Their continuations are merged into one stub per table code
  ("Tabla 1.C.2"). Embedding thousands of numbers adds noise to retrieval, and the assistant
  must not read figures off a table it cannot see in full.

## Known limitations

Part of the medical scale in the annex of RDL 8/2004 (tables 2.A/2.B) is published as images in
the consolidated text and is not indexed. Yearly updates of baremo amounts are published as
separate resolutions and are only reflected when the BOE adds them as editorial notes.

# Spanish Insurance Law RAG

Retrieval-augmented question answering over Spanish insurance legislation (consolidated texts from the
BOE open-data API), with answers that cite the law, article and paragraph, link to the official source
and state the consolidation date of the text used.

> **Work in progress.** Architecture, evaluation results and design decisions will be documented here
> as the implementation phases land.

**Not legal advice.** Answers are generated automatically from the indexed texts and may be incomplete
or wrong. Always check the official source.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## License

Code: [MIT](LICENSE). Legal texts are not redistributed in this repository; they are downloaded from
their official sources at ingestion time.

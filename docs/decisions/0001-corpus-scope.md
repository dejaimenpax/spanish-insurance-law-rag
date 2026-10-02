# 0001 — Corpus scope

**Status:** accepted · **Date:** 2026-10-02

## Context

Phase 1 indexes ten Spanish norms from the BOE consolidated-legislation API. Two of them need
an explicit decision about which of their parts are indexed.

## Decision

### Norms that approve another text

Six norms are instruments "por el que se aprueba" a consolidated text or a regulation
(RDL 1/2002, RDL 7/2004, RDL 8/2004, RD 300/2004, RD 304/2004, RD 1507/2008). The BOE publishes the
instrument (single article, its own additional/final provisions, signature) followed by a
`texto` or `reglamento` heading and the approved text. **Only the approved text is indexed.**
The instrument's provisions repeat numbers used by the approved text ("disposición final
primera" exists in both) and are almost always formal (approval, repeal lists, entry into
force), so indexing them would add ambiguous citations with little value.

### Real Decreto-ley 3/2020 (BOE-A-2020-1651)

Only book two, title I (insurance and reinsurance distribution, articles 127–211) is in scope.
In the API index that is every block from `ls` (LIBRO SEGUNDO) up to, but not including, `ti-6`
(book two, title II: pension funds).

Final provision 7.2 of the norm lists the final-part provisions enacted under the insurance
competences: additional provisions 12–14, transitional provisions 2, 3 and 5 and final
provision 5. Starting from that list and reading every provision of the final part:

| Included | Why |
|---|---|
| DA 11 | Fee for registration in the administrative register of insurance distributors |
| DA 12 | Training programmes for distributors |
| DA 13 | Retention of pre-contractual documentation (activities of book two, title I) |
| DT 2, DT 3 | Transitional regime for insurance undertakings and intermediaries |
| DT 4 | Book two, title I does not apply to contracts concluded before it entered into force |
| DT 5 | Mediation and distribution agreements in force at that date |
| DF 10 | Ley 20/2015 applies on a supplementary basis (useful for cross-references) |
| DF 15 + Annex XII | Competence and knowledge requirements for employees involved in distribution |

| Excluded | Why |
|---|---|
| DA 14 | Budget reallocation, no substantive content |
| DF 5 | Amends RD 1060/2015, which is indexed in its consolidated form (would duplicate text) |
| DA 15, DT 6 | Pension funds (book two, title II), out of scope |
| DF 7, DF 16 | Constitutional basis and entry into force; kept as metadata, not as chunks |
| Everything else | Public procurement (book one) and tax matters (book three) |

The scope is declared in `src/insurance_rag/corpus/catalog.yaml` and checked by tests: a unit
test on the catalog and a network test that resolves it against the live text (article 127 in,
article 212 out).

### Amending final provisions

Final provisions whose rubric is "Modificación de/del …" are not indexed (about twenty across
LOSSEAR and ROSSEAR). Their wording is either already consolidated into the amended norm, which
is indexed in force, or belongs to a norm outside the corpus (tax laws, the repealed Ley
26/2006), so indexing it would surface duplicated or outdated text.

### Preambles

Preambles (exposiciones de motivos) are not indexed in phase 1: they are not normative and
would compete with articles in retrieval. Questions about a law's purpose are answered from its
articles or abstained on.

## Consequences

Citations are unambiguous within each norm. Questions about RDL 3/2020's procurement or tax
content are answered with "No consta en la normativa indexada", which is the intended behaviour.

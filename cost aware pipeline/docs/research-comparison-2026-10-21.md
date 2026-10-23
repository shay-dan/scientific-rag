# Fixed-question Fast/Standard check — 2026-10-21

This is a six-run smoke comparison, **not** a statistical performance or scientific-quality benchmark. Each case used `fresh=true`, ran sequentially, and did not call Ragas. Provider availability, quotas, paper ranking, and network conditions were not held constant. Results are not directly comparable to older runs made with earlier code or different questions.

| Corpus / question | Mode | Run ID | HTTP | Time | Evidence chunks | Unique papers | Fallbacks | Source errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PubMed: CAR-T outcomes/toxicities in B-cell lymphoma | Fast | `66c148cb-155d-48b8-8315-f7ec37ebda6a` | 200 | 34.7 s | 4 | 2 | 4 | 0 |
| PubMed: same question | Standard | `a58c3a43-13c8-4315-8fda-e80310157bf0` | 200 | 62.3 s | 8 | 4 | 10 | 0 |
| arXiv: surface-code quantum error correction | Fast | `e520bc47-b928-4804-9299-045de0ac8d16` | 200 | 56.5 s | 2 | 2 | 5 | 0 |
| arXiv: same question | Standard | `2de0598b-e024-4c4e-aeff-a8078684cfaf` | 200 | 46.1 s | 5 | 5 | 9 | 0 |
| Mixed: ML for breast-cancer diagnosis in mammography | Fast | `12e063e9-c561-4e7f-9acd-197c978593f8` | 200 | 32.6 s | 4 | 1 | 8 | 0 |
| Mixed: same question | Standard | `bf08e78e-56c6-4b86-8ccc-3d6b87a155f9` | 200 | 49.6 s | 7 | 5 | 17 | 0 |

## Manual answer and source review

- **CAR-T:** Both modes correctly refrained from claiming clinical efficacy or toxicity rates from the retrieved evidence. Fast used two unique PubMed papers; Standard used four. The selected papers were mainly consensus guidance, reviews, and a leukapheresis article, not direct primary outcomes studies for the requested lymphoma population. This is safe abstention but a **coverage failure** for the question. Eight evidence chunks must not be presented as eight independent papers.
- **Surface codes:** Fast produced a decoder-focused answer from two arXiv papers and explicitly stated that it did not cover hardware. Standard covered layout, hardware and decoder approaches with five papers. One Standard source on CSS-code performance is broader than surface-code implementations; its direct relevance should be reviewed before using this as a gold-quality answer. A single run cannot establish which mode is usually faster: Standard happened to finish 10.4 seconds sooner here.
- **Mixed mammography:** Fast searched both corpora but selected only one PubMed review. Standard selected two PubMed and three arXiv papers; the arXiv set included Wisconsin fine-needle-aspiration data and thermal imaging papers that do **not** answer a mammography-specific question. The answer identified these as non-mammographic and did not use them as main evidence, but retrieval still passed irrelevant papers. This remains a **source-filtering gap**.
- **Provider reliability:** All six runs succeeded, but fallback activity remained substantial. In the mixed Standard run, Groq succeeded in 0/1 attempts, OpenRouter in 2/3, and Gemini in 15/15. Token prices were not configured as a complete comparable cost basis, so no cost claim is made.

## Infrastructure checks

- PubMed Entrez now uses an 8-second per-network-call socket timeout, two Biopython attempts, and a 90-second thread-local fetch budget. The application still records per-source mining failures and can continue with another source. A socket timeout is an inactivity limit, not a guaranteed total wall-clock bound under a slow trickle response; the outer source timeout can still cancel an await before an unusual synchronous call fully stops.
- The full local suite passed **107/107**, including three real PostgreSQL/pgvector integration tests (extension availability, vector distance ordering, and PostgreSQL full-text matching). Those tests used temporary tables and did not modify application rows.
- A GitHub Actions job was added for the `cost aware pipeline` folder with a temporary pgvector PostgreSQL service. It has **not yet been validated by a GitHub-hosted run**; it will run once the workflow is committed and pushed.

## Next quality work

1. For intervention/outcome questions, improve PubMed retrieval toward primary studies or systematic reviews for the exact intervention and population before spending time on synthesis.
2. Reject or demote non-mammographic imaging papers when the query specifically requires mammography, while retaining ambiguous papers for review rather than blindly discarding them.
3. Repeat the fixed questions across multiple days/provider conditions before publishing latency, reliability, or quality averages. Add reviewed reference answers if a numerical quality claim is needed later.

## Follow-up: coverage and modality filtering

These are targeted follow-up checks, **not** a controlled benchmark or Ragas evaluation. The two mammography runs used the same question and Standard mode, but provider fallback, external search ranking, and network conditions were not controlled. Source counts below are distinct papers, not evidence chunks.

| Check | Run ID | Result |
| --- | --- | --- |
| CAR-T clinical outcomes/toxicities, Standard | `57b3608a-47e9-4dfa-bfd8-7538e791b5a3` | HTTP 200; 3 distinct primary clinical PubMed studies selected after the targeted clinical query was widened. The preceding six-run comparison had no direct clinical outcome studies. |
| Mammography ML, Standard, after modality filter only | `24f008f0-9a5d-4dd6-b7fb-b8f0f1009162` | HTTP 200; arXiv alternatives were rejected with explicit modality reasons, but only 1 distinct PubMed systematic review supported the answer. This exposed an overly constrained PubMed search. |
| Same mammography question, Standard, after query-anchor fix | `e0c802ef-bd11-4361-b5fa-ba5b95b733fc` | HTTP 200; 4 distinct papers selected: 3 PubMed and 1 arXiv. PubMed included a primary mammographic-lesion clinical study; arXiv selected a screening-mammography paper. Wisconsin FNA and thermal-image arXiv papers were rejected before indexing. |

The mammography PubMed query now requires three concept groups: machine/deep learning or AI; breast cancer; and mammography/mammograms/tomosynthesis. It no longer requires incidental question words such as “diagnosis” or planner detail words. Each PubMed search screens up to six candidate papers when the evidence budget is three, and may index at most three papers. The saved inspector shows six returned records and three accepted papers per subquery in the final run; its `indexed_count` field counts chunks, **not** papers.

The new mammography modality check is deliberately narrow. It rejects explicit alternative-modality studies and papers with no mammography mention in either title or abstract, but keeps mixed-modality mammography work. It does not prove that every retained article supports every claim in the synthesized answer. The final answer and its numerical claims still require independent review before external use.

After these changes, the complete automated suite passed **111/111**, including three live PostgreSQL/pgvector integration tests. No Ragas evaluation was run for these follow-up checks. The active retrieval version is `postgres-hybrid-rrf-mmr-parent-child-v10`, so older cached answers are not reused by the new pipeline.

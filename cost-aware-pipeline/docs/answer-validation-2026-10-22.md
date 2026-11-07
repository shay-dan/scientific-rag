# Fresh-answer validation — 2026-10-22

Two questions were submitted in Standard mode with `fresh=true` against retrieval version `postgres-hybrid-rrf-mmr-parent-child-v11`. There were **three requests**, because the CAR-T question was retried once after a provider timeout. No Ragas evaluation was run. These are targeted smoke checks, not an accuracy benchmark.

| Question | Run ID | HTTP / status | Wall time | Outcome |
| --- | --- | --- | ---: | --- |
| CAR-T outcomes and toxicities in B-cell lymphoma | `4f976e4a-76c5-4ae6-b1a5-f53f581172e9` | 502 / failed | 80.6 s | Mining, indexing and retrieval completed; `gemini-3.6-flash` expert synthesis failed with `DeadlineExceeded` after ~28.8 s. No answer to audit. |
| ML for mammography diagnosis and limitations | `6d416693-22de-44c5-b75e-58e7183330db` | 200 / completed | 78.4 s | Answer generated from 4 distinct sources (3 PubMed, 1 arXiv). Manually checked against the exact eight saved synthesis passages. |
| CAR-T retry | `aabe95a1-4c59-40fb-a3c7-0f504aaa3766` | 502 / failed | 88.9 s | Again mining, indexing and retrieval completed; `gemini-3.6-flash` expert synthesis failed with `DeadlineExceeded` after ~28.7 s. Stopped without a third attempt. |

## Mammography answer audit

- The previous **0.93 AUC misattribution is absent**. The new answer reports **AUC 0.841**, 83.5% sensitivity and 84.7% specificity for the Japanese study's retrospective test set of 2,059 mammograms. These values match the selected paper's own abstract, not another study cited in its discussion. [PubMed PMID 40608200](https://pubmed.ncbi.nlm.nih.gov/40608200/).
- The answer explicitly identifies the study as retrospective, describes the arXiv pF1 result as a preprint validation benchmark, and says neither establishes prospective clinical benefit. That is consistent with the supplied passages and the arXiv record. [arXiv 2307.11274](https://arxiv.org/abs/2307.11274).
- The limitations about older computer-aided detection producing many false-positive marks, need for prospective screening evaluation, and dataset/annotation constraints are supported by the two selected narrative reviews. [PubMed PMID 32531273](https://pubmed.ncbi.nlm.nih.gov/32531273/), [PubMed PMID 38787015](https://pubmed.ncbi.nlm.nih.gov/38787015/).
- Caveat: the answer calls the retrospective Japanese study a “Primary Clinical Study.” This follows the app's PubMed/title study-design label, but the body correctly says **retrospective validation**. The label could still mislead a reader expecting a prospective interventional trial. Source labels and design classification deserve a separate fix.

## CAR-T status and next gate

The two failures do **not** show whether the denominator and historical-comparator safeguards succeeded; they produced no answer. The provider logs point specifically to the Gemini expert model timing out, after other providers were unavailable or rate-limited in those runs. Do not claim CAR-T answer quality has improved on the basis of prompt tests alone. Before more paid runs, investigate expert fallback/timeout behavior and consider replaying the already-mined evidence rather than repeating PubMed mining. The saved older CAR-T answer remains unchanged.

# Manual answer-to-source audit — 2026-10-21

Scope: two saved Standard research runs, not a general accuracy estimate. I compared each answer with the **exact `research_evidence` passages supplied to synthesis**, then checked the key primary records on PubMed/arXiv. No new model calls or Ragas evaluation were made. A citation appearing after a sentence is not by itself evidence that every claim in that sentence is supported.

## CAR-T: `57b3608a-47e9-4dfa-bfd8-7538e791b5a3`

| Answer claim | Audit | Reason / safer wording |
| --- | --- | --- |
| C-CAR011 in 10 refractory DLBCL patients: response 20% at 4 weeks, 50% at 12 weeks; disease control 60% at 12 weeks; any-grade CRS 90%, grade ≥3 CRS 0; no observed neurotoxicity | Supported in the supplied S1 abstract | Preserve the study population, small sample, and timepoints. [PubMed PMID 34727319](https://pubmed.ncbi.nlm.nih.gov/34727319/). |
| Trispecific CAR: “patients with relapsed/refractory B-cell lymphoma achieved an overall response rate of 50%, including a complete response rate of 83% in the lymphoma group” | **Misleading population/denominator** | S2 reports an overall response rate of 50% across 16 patients with B-cell **malignancies**, then complete responses in 83% of the **lymphoma subgroup**. Do not present 50% and 83% as nested rates for the same population. The subgroup size is not supplied in the saved abstract. [PubMed PMID 42446920](https://pubmed.ncbi.nlm.nih.gov/42446920/). |
| Trispecific CAR: 1-year overall survival 61%, “demonstrating durable remissions” in lymphoma | Overstated | S2 reports the 61% survival rate for the trial cohort and separately says durable remissions were observed in lymphoma. It does not establish the lymphoma-specific survival rate or prove durability from the 1-year survival number alone. [PubMed PMID 42446920](https://pubmed.ncbi.nlm.nih.gov/42446920/). |
| Hu19-CD828Z: 55% complete remission among 20 B-cell lymphoma patients; severe neurologic toxicity 5% versus 50% for FMC63-28Z | Numbers supported, comparison needs qualification | The 50% comparator refers to a CAR **previously tested by the group**, not a randomized concurrent comparison established by the saved abstract. Do not imply a head-to-head trial or causal superiority. [PubMed PMID 31959992](https://pubmed.ncbi.nlm.nih.gov/31959992/). |
| “All cited evidence comes from early-phase (Phase I) primary clinical studies with small patient groups ranging from 10 to 20” | Supported by the supplied abstracts | S1 has 10, S2 has 16, S3 has 20 participants. This limits generalizability; the studies are different constructs and populations and their percentages should not be pooled. |

**Conclusion:** source coverage improved, but this saved answer should **not** be used as a polished medical example unchanged. The main error is conflating an all-malignancy denominator with a lymphoma-subgroup denominator; the second is an unqualified historical comparator.

## Mammography ML: `e0c802ef-bd11-4361-b5fa-ba5b95b733fc`

| Answer claim | Audit | Reason / safer wording |
| --- | --- | --- |
| AI is used to detect/classify lesions in mammography; dataset bias, standardization and regulation limit adoption | Broadly supported by S1/S2 reviews | These are literature summaries and potential applications, not proof of improved clinical outcomes. [PubMed PMID 38787015](https://pubmed.ncbi.nlm.nih.gov/38787015/), [PubMed PMID 38640824](https://pubmed.ncbi.nlm.nih.gov/38640824/). |
| “Clinical diagnostic outcomes ... established by primary clinical and testing studies [S3, S4]” | **Overclaim** | S3 is an arXiv preprint reporting an RSNA validation metric, not a prospective clinical outcome study. S4 used **retrospectively collected** mammograms from 63 institutions; the authors state that a prospective reader study was still planned. Say “retrospective diagnostic test performance and a preprint benchmark,” not established clinical outcomes. [arXiv 2307.11274](https://arxiv.org/abs/2307.11274), [PubMed PMID 40608200](https://pubmed.ncbi.nlm.nih.gov/40608200/). |
| “A Japanese AI-CADe system reported ... AUC 0.93 [S4]” | **Misattributed study result** | The selected S4 passage mentions 0.93 only while describing **Ueda et al.**, a different paper cited in its discussion. S4's **own** test-set AUC is **0.841** (95% CI 0.822–0.859); sensitivity 83.5%, specificity 84.7%, based on 2,059 test mammograms. Do not cite S4 as the source of a 0.93 result. [PubMed PMID 40608200](https://pubmed.ncbi.nlm.nih.gov/40608200/). |
| RSNA dataset ≈20,000 female patients, validation case pF1 0.56 [S3] | Supported as a reported validation result | Keep the metric name and preprint/validation context. It is not a clinical screening outcome or directly comparable to AUC/sensitivity. [arXiv 2307.11274](https://arxiv.org/abs/2307.11274). |
| “Accuracy can decrease across racial or regional groups [S1, S4]” | Needs qualification | The S4 passage says overseas standalone systems showed inferior performance versus Japanese readers **in referenced work**, and raises population differences as a possible explanation. S4's own study does not quantify a cross-racial performance drop. Phrase as an external-validity concern, not an established result of this test. [PubMed PMID 40608200](https://pubmed.ncbi.nlm.nih.gov/40608200/). |

**Conclusion:** source filtering improved materially, but the answer still has **citation laundering**: a numerical result from a paper discussed inside S4 was presented as if it were S4's own result. It also overstates retrospective/benchmark results as clinical outcomes.

## Implication for the next engineering step

The remaining issue is chiefly **synthesis provenance**, not just retrieval. The answer writer needs to distinguish (1) the selected study's own results, (2) background studies mentioned within its discussion, (3) review-level summaries, and (4) preprint benchmark performance. Numerical clinical claims should include their exact population, denominator or sample size where available, endpoint/timepoint and design. If the passage does not establish those details, omit or qualify the number. Add regression examples for the S2 mixed denominators and S4 cited-paper AUC before another live run.

## Follow-up implementation — 2026-10-22

The synthesis context now labels PubMed abstracts, full-text sections and arXiv preprints. The expert prompt explicitly forbids mixing response rates from different populations, attributing a prior cited study's result to the selected paper, treating retrospective/preprint validation as prospective clinical outcomes, or presenting a historical treatment comparison as randomized. Retrieval version `postgres-hybrid-rrf-mmr-parent-child-v11` prevents older cached answers from being returned as current results. The previously saved answers above remain unchanged.

One new unit test checks that section/preprint provenance reaches the expert's actual input; the prompt regression test checks the four numerical-claim rules. The full suite passed **112/112**, including three real PostgreSQL/pgvector integration tests. **No fresh paid model call or Ragas evaluation was made for this follow-up.** Prompt and input tests establish wiring, not empirical answer correctness; a fresh answer-to-source audit is still required before declaring the two observed errors fixed in production output.

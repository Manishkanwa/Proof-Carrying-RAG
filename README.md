# Proof-Carrying RAG: Claim-Level Evidence Verification and Selective Repair

## Project status

This is a research prototype for checking RAG claims against retrieved evidence. It includes a five-paper corpus, chunking, dense retrieval, a local NLI verifier, a 30-question retrieval benchmark, an eight-claim verification pilot, and a seven-question abstention diagnostic.

**This is a prototype and diagnostic study, not a validated production RAG system.** The current notebook did not complete a controlled comparison of vanilla RAG, verification-only RAG, and selective repair, so it must not claim that such a comparison has been demonstrated.

## Research question

Can claim-level NLI checks over retrieved passages help identify unsupported or contradicted claims in a small research-paper corpus, and when should the system abstain rather than present an answer as supported?

## Pipeline

1. Extract text from the five PDFs.
2. Split page text into overlapping chunks while preserving paper/page/chunk metadata.
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` and retrieve top-k chunks by similarity.
4. Convert an answer into declarative claims before verification. Do not pass interrogative questions directly as NLI hypotheses.
5. Score each claim/passage pair with `cross-encoder/nli-deberta-v3-xsmall`.
6. Distinguish `supported`, `contradicted`, and `insufficient_evidence`.
7. Preserve evidence chunk IDs and text for auditability. Treat low-confidence, conflicting, mathematical, and technical cases as review cases.
8. For selective repair, retrieve additional evidence only for unsupported claims, then re-verify; abstain if evidence remains unavailable.

## Data and implementation

- Corpus: Attention Is All You Need; Neural Machine Translation by Jointly Learning to Align and Translate; Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks; Dense Passage Retrieval for Open-Domain Question Answering; Sentence-BERT.
- Chunk size: 180; overlap: 40 (as configured in the notebook).
- Retriever: `all-MiniLM-L6-v2`.
- Verifier: `cross-encoder/nli-deberta-v3-xsmall`. Verified label mapping: 0 contradiction, 1 entailment, 2 neutral.
- Environment: Google Colab; local CPU inference.
- A hosted generation API quota was exhausted, so a complete generator-based comparison was not run.

## Preliminary results (not final benchmark claims)

- Retrieval: prior run reported Hit Rate@5 = 1.00 and MRR = 0.833. These are provisional because relevance labels have not been independently audited.
- Claim verification: 6/8 = 75% on a small manually designed pilot with provisional reference labels. This is not a general performance estimate.
- Abstention diagnostic: the earlier NLI implementation predicted `contradicted` for all seven intended unanswerable questions, yielding 0/7 agreement with reference labels. This exposes a design/evaluation failure: questions were passed directly to NLI as hypotheses, and lack of evidence was conflated with contradiction.
- An always-abstain baseline would score 7/7 on this all-unanswerable subset by construction. That is not meaningful general abstention performance.

## Interpretation

The experiments suggest:
1. Dense retrieval can find semantically related chunks, but similarity is not proof that a chunk answers a question.
2. NLI can struggle with formulas, technical phrasing, and indirect evidence.
3. Contradiction and insufficient evidence are different states.
4. Recording evidence text and chunk IDs makes decisions auditable, even when the verifier is uncertain.

The experiments do **not** establish that the complete pipeline improves factuality over vanilla RAG or that selective repair works.

## Limitations

- Small corpus and hand-designed evaluation sets.
- Provisional/manual relevance and claim labels.
- NLI thresholds are heuristic and not calibrated on a held-out set.
- Mathematical notation and technical claims are challenging.
- No completed controlled comparison of vanilla RAG, verification-only, and verification-plus-repair.
- Results from 8 claim examples and 7 abstention questions do not support generalization claims.

## Next experiments

1. Freeze a versioned benchmark with manually checked evidence labels.
2. Evaluate retrieval with Hit Rate/Recall@k and MRR using audited relevance judgments.
3. Evaluate NLI on declarative supported, contradicted, and insufficient-evidence claims.
4. Compare vanilla RAG, verification-plus-abstention, and selective repair on the same questions.
5. Report per-class precision/recall/F1, confusion matrix, answer coverage, and supported-claim accuracy.
6. Fix prompts and thresholds before held-out testing.

## Reproduction

Run the notebook in Google Colab. Mount Drive, place the five PDFs in the configured `data/papers` directory, and execute notebook cells in order. The metrics in this README are historical pilot results; rerun the notebook to reproduce them.

## Portfolio description

**Proof-Carrying RAG — Research Prototype**  
Built a document-grounded QA prototype over five research papers using dense retrieval, claim-level NLI verification, and evidence chunk tracking. Evaluated retrieval and a small verification pilot, investigated abstention failures, and documented the distinction between contradiction and insufficient evidence. The study highlights limitations of NLI verification for mathematical and technical claims and defines a controlled evaluation protocol for selective repair.

# Final Evaluation Notes

## Final claim

This is a research prototype investigating claim-level evidence checking in RAG, not a completed demonstration of improved factuality.

## Results to report

| Component | Result | Qualification |
|---|---:|---|
| Retrieval | Hit Rate@5 = 1.00; MRR = 0.833 | Prior run; provisional labels need audit |
| Claim-verification pilot | 6/8 (75%) | Small pilot; provisional labels |
| Earlier abstention diagnostic | 0/7 (0%) | Failure of the earlier implementation |
| Always-abstain baseline on seven unanswerable items | 7/7 (100%) | By construction; not meaningful general performance |

## Interpretation

The abstention diagnostic demonstrates a key failure mode: an interrogative question was passed as an NLI hypothesis, and `contradicted` was predicted where the reference state was `insufficient_evidence`. Absence of a statement in retrieved passages is not contradiction. Dense similarity can retrieve related-but-irrelevant text, so similarity alone cannot prove that evidence answers a question.

The NLI label mapping was checked: `0: contradiction`, `1: entailment`, `2: neutral`. The verifier still struggled with technical and mathematical examples. Thresholds remain heuristic and uncalibrated.

## Claims supported by current work

- The prototype indexes a small research-paper corpus and retrieves candidate evidence chunks.
- It applies a local NLI model to claim/passage pairs and records evidence for inspection.
- The pilot revealed weaknesses around technical claims and abstention.
- The work provides a basis for a larger, properly labeled comparison.

## Claims not supported by current work

- Verified RAG outperforms vanilla RAG.
- Selective repair improves factual accuracy.
- The verifier achieved 100% abstention accuracy.
- The pilot generalizes to other corpora or domains.

## Completion status

The project can be presented as a completed **prototype and diagnostic study**. The planned end-to-end comparison of vanilla RAG, verification-only, and selective repair remains future work.

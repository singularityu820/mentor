# Rebuttal for Submission 408 (CaseSentinel)

We appreciate the thoughtful reviews. For Reviewer 6Ly5, we are concerned there may be a mismatch: the comments describe a case-tree retriever evaluated on COLIEE/LegalBench with BM25+LLM baselines, whereas CaseSentinel is a Bayesian multi-agent HSG framework evaluated on 20 adjudicated cases from China Judgements Online with single-agent RAG ablations. We respectfully invite the AC to verify whether the review corresponds to this submission.

Reviewer TsMo – Governance, conflict resolution, and scalability will be clarified: every state change (human or agent) is recorded in an immutable audit log, human overrides lock beliefs for future loops, and we will add deployment notes on latency (~30 s/loop) and CMS integration.

Reviewer po2W – The 20-case corpus is small because each case required multi-hour expert labeling, yet the architecture is jurisdiction-agnostic; we will detail how the Legal Basis node adapts to other systems. We will also report new GPT-4o spot-checks confirming CaseSentinel’s relative gains.

Reviewer mv8W – We will include confidence intervals, significance tests, and expanded governance safeguards (on-prem data handling, prompts that surface exculpatory evidence) to address ethical and reproducibility questions.

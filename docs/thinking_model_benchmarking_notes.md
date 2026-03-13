# Thinking Model Benchmarking Notes

This document captures the main lessons from benchmarking multiple reasoning-heavy models on Together endpoints.

The goal is to make the next benchmarking session faster, more consistent, and less dependent on rediscovering the same failure modes.

## What We Learned

Reasoning-heavy models do not all behave the same way under low-load benchmarking.

Some models:

- produce a short visible answer quickly
- keep reasoning compact enough that TTFT and E2E are easy to measure

Other models:

- spend a very large portion of the completion budget on reasoning
- delay visible output until late in the response
- may return no visible output at all if the completion cap is too low
- may appear "broken" when the real issue is reasoning budget consumption

This behavior is strongly model dependent.

## Main Model Patterns Observed

### MiniMax M2.5

- Most cooperative for low-load latency benchmarking
- Usually produced visible output reliably
- Often the best latency performer among the tested models

### GLM-5

- Can work, but often needs a larger completion budget before visible output appears
- Tends to spend substantial time reasoning before first visible token
- With too-low caps, may use the whole budget on reasoning and never surface the answer

### Qwen 3.5 397B

- Extremely reasoning-heavy on several workloads
- Can consume a very large completion budget before producing a short visible answer
- May eventually succeed, but at latency levels that make it a poor fit for low-latency comparisons on some workloads

### Kimi K2.5

- Serverless testing was blocked by model-specific provider-side rate limiting
- Dedicated endpoint path was required to continue testing

## Important Benchmarking Lesson

If a reasoning model fails to produce visible output, do not immediately assume:

- parser bug
- API bug
- bad prompt

Often the first thing to check is:

- was the completion cap too low relative to the model's reasoning verbosity?

## Workload Types We Tried

### 1. Simple Probe

Example:

- `"What is the capital of France?"`

Use case:

- clean TTFT and E2E sanity check
- useful for endpoint responsiveness
- good debugging probe

Limitation:

- too trivial to serve as the main benchmark

## 2. Summarization

Examples:

- `v1_summarization_13000to600_non_strict.jsonl`
- `v1_summarization_4000to100_non_strict.jsonl`

What happened:

- summarization with a target token budget invited open-ended planning
- some reasoning models spent a lot of time thinking about how to summarize and how to fit the output budget
- this made summarization a poor primary workload for latency comparison across reasoning models

Recommendation:

- do not use summarization as the main low-load latency benchmark for thinking models

## 3. Grounded Extraction

Example:

- `v1_grounded_extraction_ops_brief.jsonl`

Why it was better:

- bounded output
- strict JSON schema
- fact extraction only

What we learned:

- better than summarization, but some models still consumed the budget on reasoning
- useful as a secondary workload
- good for checking whether a model can finish a realistic structured task

## 4. Track A Retrieval QA

Example:

- `v1_short_retrieval_qa.jsonl`

Why this is the best primary benchmark shape so far:

- more realistic than the trivial probe
- shorter and more bounded than summarization
- clear factual retrieval target
- strict JSON output
- less likely to trigger open-ended compression/planning loops

Recommendation:

- use this as the primary low-load comparison workload

## Recommended Benchmark Tracks

### Track A: Short Retrieval / Factual QA

Purpose:

- clean low-load TTFT and E2E comparison

Desired properties:

- medium-length input
- tightly bounded output
- exact-fact retrieval
- strict JSON or one-sentence output

Current best corpus:

- `corpus/v1_short_retrieval_qa.jsonl`

### Track B: Grounded Structured Extraction

Purpose:

- realistic document task without open-ended summarization

Desired properties:

- larger input
- fixed schema
- JSON-only output

Current best corpus:

- `corpus/v1_grounded_extraction_ops_brief.jsonl`

### Diagnostic Workloads

Use these when debugging:

- simple probe corpus
- non-streaming variants
- increased `max_output_tokens`

## How To Debug A Bad Result

When a reasoning model performs badly:

1. Check whether the workload itself is too open-ended.
2. Check whether the model is spending the completion budget on reasoning.
3. Increase `max_output_tokens` substantially and test again.
4. Compare streaming vs non-streaming behavior.
5. If serverless is blocked, test a dedicated endpoint path.
6. Distinguish:
   - provider-side rate limiting
   - reasoning-budget exhaustion
   - genuine parsing problems

## Token Budget Guidance

Low caps can create misleading failures.

Typical pattern:

- model appears to "fail"
- diagnostic trace shows long reasoning text
- no visible output appears

In practice:

- increasing `max_output_tokens` sometimes converts an apparent failure into a valid completion

But this is not always enough:

- if a model still uses thousands of tokens for reasoning before a short answer, that is itself a useful benchmark finding

## Spreadsheet Interpretation

When comparing models, distinguish:

- provider-reported total input usage
- dataset-estimated system/user prompt split
- visible output token estimate
- thinking token estimate

A slow model that eventually returns a short JSON answer may still have spent most of its completion budget on reasoning.

That is a model behavior result, not a logging bug.

## Practical Recommendation For Future Sessions

If asked to benchmark a range of reasoning models tomorrow:

1. Start with Track A retrieval QA.
2. Keep concurrency at 1 for best-case latency checks.
3. Use spreadsheet logging from the start.
4. If a model looks broken, inspect reasoning traces before changing conclusions.
5. If needed, rerun with a much larger completion cap.
6. Use grounded extraction as a secondary realism check.
7. Treat summarization as diagnostic or separate analysis, not the primary latency benchmark.

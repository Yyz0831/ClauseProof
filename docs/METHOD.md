# ClauseProof: inspectable decisions from a compact contract model

Contract review needs a distinction between a statement being contradicted and simply not being supported. It also needs the relevant qualification or exception alongside the main clause. ClauseProof turns this problem into 17 fixed NDA statement checks, with three labels and explicit source references. The product makes the model's decisions inspectable through source highlights and original-output exports.

## What was built

An English contract-review interface accepts four public development samples or a user's plain text. Aya Expanse 8B with the frozen ContractNLI v2 adapter runs locally in 4-bit precision. A separate experiment page contains the preserved original/v1/v2/v3 responses for all 306 decisions. It supports strict and content-diagnostic aggregate views, improvements/regressions, and inspection of official versus predicted evidence.

Full-text inputs are validated before inference. The service rejects overlong documents instead of truncating them, retains raw format failures, allows stopping between statements, restores a job after page refresh, and exports partial results honestly. Document snapshots prevent a later input edit from changing the source attached to an existing citation.

## Data and adaptation

The official ContractNLI training split supplies 423 contracts and 7,191 statement annotations. Existing preparation excludes exact normalized-text duplication and applies full-sequence tokenizer limits. The selected v2 set has 2,040 examples from 413 training contracts: 850 entailments, 850 not-mentioned cases and 340 contradictions. Official labels and span references are unchanged.

v2 retains the 1,020 v1 examples and adds 510 not-mentioned cases selected for lexical overlap with their hypotheses, plus 510 entailments prioritizing multi-span, adjacent and conditional evidence. This tests a data-selection recipe; it is not a controlled estimate of a single causal factor because size, label balance and total optimizer updates change together.

AlignTune's TRLSFTTrainer performs completion-only QLoRA. Settings: one epoch, 1,020 updates, batch 1 with accumulation 2, learning rate 1e-4, seed 42, LoRA rank 8/alpha 16/dropout 0.05 on q/k/v/o projections, NF4 double quantization and BF16 compute. A validated activation-offloading runtime enables the full 8,192-token training budget on the RTX 5080. Source hashes, gradient checks, runtime policy and final adapter hash are saved.

## Comparison and model selection

All four models were evaluated once with the same prompt and greedy decoding on the final shared cohort: 18 development contracts, 306 decisions, including 150 entailments, 130 not-mentioned cases and 26 contradictions. The official test split was not used for prediction or model selection; test text hashes were used for deduplication.

| Model | Label accuracy, strict | Macro F1 | Evidence micro F1 | Joint exact |
|---|---:|---:|---:|---:|
| Original Aya | 39.87% | 32.00% | 20.26% | 87/306 |
| v1 | 76.80% | 72.52% | 58.72% | 156/306 |
| **v2** | **82.35%** | **77.82%** | **61.27%** | **182/306** |
| v3 | 80.07% | 74.34% | 63.57% | 173/306 |

Original Aya passes the schema in 248/306 cases; every adapted version passes in all 306. In a separate content diagnostic that removes a single complete Markdown fence, original Aya's label accuracy is 49.35%. This keeps formatting gains distinct from label-content gains.

v3 recognizes 20/26 true contradictions versus v2's 18/26, but makes 24 false contradiction calls versus 11. v2 is therefore the live model. The earlier single AI source-review round covered 18 selected v2/v3 pairs, not the full benchmark; its results are not presented as population accuracy. No repeated judge round is added here.

## Auditable use of the stack

Training artifacts prove actual AlignTune execution. The ClauseProof AuditKIT bridge consumes frozen raw predictions and official reference IDs, invokes the original strict parser inside custom AuditKIT metrics, and saves framework fingerprints and per-case scores. Format, label and joint metrics agree with the existing reports for all four arms, with zero AuditKIT errors. This is reproducible offline evaluation, not another inference run or an independent semantic judgment. Historical v2 preparation used custom selection scripts; CuratorKIT is not credited with creating those data.

## Product acceptance evidence

The browser ran all 17 statements for public document 10 in 17.88 seconds with a warm model, with 17 valid outputs and an exact raw match to the previously saved v2 outputs. This is one measured example, not a throughput guarantee. A separate three-statement synthetic input exercises support, contradiction and missing information; one source inspection confirmed the decisions and citations. Browser checks cover file import, source binding after input changes, export, refresh recovery, cancellation and responsive layouts. Backend tests cover invalid inputs, origin/host guards, strict failures, concurrency, partial failures and retention.

## Limits and reproducibility

This cohort was used for model selection. Correlated questions, one seed, and a small sample limit generalization. Evidence F1 is span-ID overlap; neither matching an official label nor highlighting a passage proves legal correctness. Custom text is segmented by nonempty lines, whereas the benchmark has official spans. PDF parsing, arbitrary questions, multilingual contracts, long-document chunking and external-law reasoning are outside the prototype.

See the [repository README](../README.md), the saved reports in `reports/contractnli-v3/`, and `reports/clauseproof-v1/` for acceptance evidence and AuditKIT outputs. Public artifact links and reproduction commands are in the repository README. Base-model and dataset attribution: [Aya Expanse 8B](https://huggingface.co/CohereLabs/aya-expanse-8b), [ContractNLI](https://stanfordnlp.github.io/contract-nli/).

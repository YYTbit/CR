# Input preparation

## Resource records

Fetch the pinned benchmark sources with `python -m cr fetch --output external`. Prepare JSON, JSONL, CSV, or Parquet records using a schema with the actual released field names.

```yaml
identifier: item_id
message: assertion
context_fields: [query]
proposition: proposition_id
state_label: answer
positive_labels: {"No": 0, "Yes": 1}
states: [claim_false, claim_true]
split: released_evaluation
```

```bash
python -m cr prepare --records data/released.jsonl --schema data/schema.yaml \
  --resource EoBench --output data/prepared
```

Preparation writes `game.json`, `messages.json`, `groups.json`, `resource.json`, and the normalized `schema.json`. Each message contains `identifier`, `text`, `context`, and `proposition`. The prior is computed from the supplied labels.

## Execution

Model profiles and decoding settings are in `configs/paper.yaml`. Local loading resolves one revision for configuration, tokenizer and model weights. Sender sampling uses the full-support token distribution scored by GRPO; receiver calls retain native generation settings with the stated elicitation temperatures. `--receiver-backend api --endpoint http://localhost:8000/v1/chat/completions` selects an OpenAI-compatible receiver. An API key can be supplied through `CR_API_KEY`; `--receiver-revision` identifies the deployed version.

Every completed run writes a `group_policies.json` deployment archive. Multiple contexts have separate measured channels; a single context also writes `channel.json`, `messages.json`, and `result.json`. `--initial-channel` imports an existing channel or grouped channel manifest. `--proposals` accepts round policies with `message_ids` and state-by-message `policy` rows. Grouped schedules place these records under `groups`.

`--variant` selects a named configuration in `configs/ablations.yaml`. `--exposure` loads public history or announced-policy inputs. Announced exposure uses `initial_policy` for a single context and `initial_policies` for grouped contexts.

An unsuccessful measurement or exhausted budget returns `additional_measurement_required`. Query summaries distinguish current-process requests and imported measurement costs. Recorded-channel analysis uses `solve`, `movement`, and `evaluate`.

## Native evaluation

Ready-to-configure evaluators for the five resources are in `configs/native`. They load prompt constructors and scoring functions from the pinned upstream source files and verify their file contents. Model calls, responses, usage, source hashes, and native grades are retained in the output. Set the receiver endpoint and prepared schema path in the selected configuration.

EoBench uses the released Yes/No instruction and classifier. DuET-PD obtains the initial answer and performs three persuasion exchanges with separate option-letter checks. PMIYC retains `receiver_history`, or elicits an initial listener belief, then appends the candidate at the final persuasion turn. It records the initial rank, final rank, and normalized change; the paper's primary metric is final agreement. SYCON Debate obtains five responses with the candidate at its four follow-up turns and invokes the released alignment judge. Ethical uses the source's cumulative question templates with its assistant bridge text. The pinned False-Presupposition runner generates one response per question, so its endpoint is reported separately as single-question handling; five-response Turn of Flip is computed from complete recorded dialogue grades with `native-summary`. RebuttalBench uses the released four-dimension Rebuttal-RM prompt with the review, target comment, and retrieved evidence.

Custom evaluator configurations use a trusted `module:factory` and optional constructor `options`. The object exposes `sender_context(record)`, `proposition(record)`, `sender_state(record)`, and `score(record, message_text)`. Context and proposition match preparation. The state is a retained state name or integer index.

```bash
python -m cr native-evaluate --game data/prepared/game.json \
  --records data/evaluation.jsonl --evaluator data/evaluator.yaml \
  --policy results/dual_guided/group_policies.json --output results/native.json
```

Single-channel evaluation also supplies `--messages`. Generator evaluation selects `--procedure generator` and a trained `--adapter`.

Native evaluation commits completed items as it runs. Add `--resume` to retain completed grades and continue with the same records, policy, source and model settings. Receiver and judge requests, including attempts interrupted within an item, remain in the request journal with separate request counters. An unfinished item is evaluated again. JSONL outputs contain committed item records; a JSON output also retains its progress journal and writes the complete array after evaluation.

`native-summary` requires unique `item` identifiers. EoBench records contain Boolean `context_following`; DuET-PD contains Boolean `initial_correct` and `turn_three_correct` plus `domain`; PMIYC contains integer `final_agreement` ratings from one to five; SYCON-Bench contains `scenario` and five Boolean `alignments`; RebuttalBench contains `rebuttal_rm_score`. `--expected-items` validates split completeness. `statistics` processes measured per-seed comparisons.

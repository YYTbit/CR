# The Cost of Reinterpretation: Sender Adaptation against Measured Language Receivers

Learning to persuade a language agent requires understanding how sender updates
affect measured receiver responses. Existing approaches train senders from
receiver feedback, yet adaptation can change the relationship between reported
beliefs, actions, and attainable value. To address this problem, we formulate
persuasion over measured belief-action pairs and characterize the cost of
restoring consistency after a policy change. We derive the exact minimum
posterior movement and, for binary states, a necessary and sufficient condition
for preserving receiver actions during this adjustment. Specifically, a finite
linear program computes the optimal sender value compatible with the measured
channel and reconstructs an implementing policy. Its dual prices guide
reinforcement learning of new messages, while a consistency residual guides
receiver remeasurement. We evaluate the framework on five public benchmarks with
Qwen3, Llama3, and Gemma3 sender backbones. Dual-guided learning achieves the
highest response scores of the four compared procedures across all fifteen
benchmark-backbone combinations. Controlled policy shifts demonstrate how
consistency requirements, receiver information, and action changes interact.
Residual-triggered refresh retains 98 percent of continual-refresh sender value
using 31 percent of its remeasurement queries. These results connect receiver
measurement, sender learning, and adaptation. Code is available at
<https://anonymous.4open.science/r/CR-2727/>.

![Measured-receiver persuasion](assets/figures/overview.png)

## Installation

Use Python 3.10 or later and a PyTorch build compatible with the execution
hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,data,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Input preparation

Fetch the pinned benchmark sources and install their released dependencies.

```bash
python -m cr fetch --output external
```

Prepare native records with the actual field mappings in a schema file. The
[input guide](docs/execution.md) defines message text, context, proposition
identifiers, state labels, and the prior.

```bash
python -m cr prepare --records data/released.jsonl --schema data/schema.yaml \
  --resource EoBench --output data/prepared
```

Preparation writes the game, messages, context groups, and normalized resource
schema. Benchmark-specific evaluator configurations are in
[configs/native/](configs/native).

## Receiver measurement and sender adaptation

Measure the receiver on a prepared message pool.

```bash
python -m cr measure --game data/prepared/game.json \
  --messages data/prepared/messages.json --output results/measured
```

Single-context measurements produce `channel.json`. Multiple contexts retain
separate measured channels and a group manifest. Recorded channels can be
analyzed on CPU.

```bash
python -m cr solve --channel results/measured/channel.json \
  --output results/solution.json
```

The sender program returns an implementing policy and primal–dual certificate.
Train the dual-guided sender with fresh receiver measurements.

```bash
python -m cr run --game data/prepared/game.json \
  --messages data/prepared/messages.json --sender Qwen3 \
  --method dual_guided --output results/dual_guided
```

Updates use context-specific dual prices. The consistency residual determines
receiver refresh, while independent confirmation and final measurements retain
their separate query charges. An exhausted measurement budget records
`additional_measurement_required`.

## Evaluation

Evaluate a grouped policy through the configured native benchmark interface.

```bash
python -m cr native-evaluate --game data/prepared/game.json \
  --records data/evaluation.jsonl --evaluator data/evaluator.yaml \
  --policy results/dual_guided/group_policies.json --output results/native.json
```

For single-context policies, supply the measured messages as described in the
[execution guide](docs/execution.md#native-evaluation). Native metric records
preserve the benchmark's denominator and units.

| Resource | Native metric |
| --- | --- |
| EoBench | Context-following rate |
| DuET-PD | Misleading flip rate at turn three |
| PMIYC | Mean final agreement |
| SYCON-Bench | Debate number of flip |
| RebuttalBench | Rebuttal-RM score |

`native-summary` aggregates recorded native grades. `movement` computes
posterior adjustment and action-preservation diagnostics from a measured
channel. Command-specific options are available through `python -m cr COMMAND
--help`.

## Configuration

[configs/paper.yaml](configs/paper.yaml) and
[configs/ablations.yaml](configs/ablations.yaml) specify model and procedure
settings.

| Setting | Value |
| --- | --- |
| Senders | Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-3-12B-it |
| Receiver | Llama-3.1-8B-Instruct |
| LoRA | Rank 16, scale 32 |
| Learning rate | 2e-5 |
| Adaptation | 4 rounds, 2 optimizer steps per round |
| Query budget | 20,000 raw receiver requests |
| Confirmation / final repeats | 4 / 8 |
| Evaluation seeds | 0–4 |

## Software checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The CPU suite checks linear-program certificates, posterior movement, adaptive
refresh, policy gradients, native metrics, and query accounting.

## License

[MIT License](LICENSE).

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
measurement, sender learning, and adaptation.

## Algorithm Flow

To optimize communication while maintaining a current receiver model, we combine exact finite-pool optimization with message acquisition.

![Measured-receiver persuasion](assets/figures/overview.png)

**Figure 1.** Measured-receiver persuasion. Beliefs and actions define the sender program; training and remeasurement update its policy. The illustrated channel has a positive classical-to-measured gap, distinct from the sender optimization gap.

![Algorithm 1](assets/figures/algorithm.png)

## Results

### Main Results

**Table 1.** Main results on five public benchmarks, with mean and standard deviation over five seeds. Columns report EoBench context following, DuET-PD misleading-persuasion flips (percent), PMIYC raw final agreement (five-point scale), SYCON stance flips (five turns), and Rebuttal-RM scores (ten-point scale). Optimized senders share interaction budgets.

![Main results from Table 1](assets/figures/main_results.png)

### Ablation Study

**Table 2.** Component ablations across three sender backbones under shared interaction budgets. Values are mean and standard deviation over five seeds, with the metrics of Table 1. Each variant changes the component defined in the text.

![Component ablations from Table 2](assets/figures/ablations.png)

**Action measurement and independent confirmation contribute to acquisition quality.** Posterior Value decreases PMIYC from 3.94 to 3.47 and Rebuttal from 9.51 to 8.36. Its best-response objective assumes decisions follow reported beliefs; action measurement records actual decisions. Shared Measurement reaches 3.61 and 8.55, respectively. Independent confirmation evaluates candidates on new receiver samples, reducing selection effects from the acquisition batch.

### Analysis

![Analysis from Figure 2](assets/figures/analysis.png)

**Figure 2.** Theory and empirical behavior. (a) Predicted and realized utility. (b) Observed movement against the consistency-restoration minimum. (c) Movement as policy mass shifts. (d) Share of runs with action changes.

**Posterior drift and action changes occur at different stages.** Figure 2(d) records zero action changes across five seeds at transferred mass 0.06 and five at 0.14 in every core setting. At 0.10, counts are one, zero, and four for EoBench, DuET-PD, and PMIYC. The capacity interpretation applies when measured actions are best responses within their posterior intervals. The observed variation supports measuring decisions together with reported beliefs.

## Getting Started

### Environment Setup

```bash
conda create -n cr python=3.10 -y
conda activate cr
pip install -e '.[models,data]'
```

### Data Preparation

Download the benchmark sources:

```bash
python -m cr fetch --output external
```

Map the downloaded record fields in `data/schema.yaml`, using the
[record format](docs/execution.md#resource-records). Prepare the messages and game:

```bash
python -m cr prepare --records data/released.jsonl --schema data/schema.yaml \
  --resource EoBench --output data/prepared
```

## Training

Set the sender and receiver models in `configs/paper.yaml`.
Run sender training with dual-guided rewards:

```bash
python -m cr run --game data/prepared/game.json \
  --messages data/prepared/messages.json --sender Qwen3 \
  --method dual_guided --output results/dual_guided
```

## Evaluation

Set the official benchmark evaluator in `data/evaluator.yaml`, then evaluate
the learned policy:

```bash
python -m cr native-evaluate --game data/prepared/game.json \
  --records data/evaluation.jsonl --evaluator data/evaluator.yaml \
  --policy results/dual_guided/group_policies.json --output results/native.json
```

The [evaluation guide](docs/execution.md#native-evaluation) covers evaluator setup,
single-context policies, and result aggregation.

## License

[MIT License](LICENSE).

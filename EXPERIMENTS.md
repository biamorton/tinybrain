# TinyBrain experiments

This file records protocol and results so v0.3 can be compared with later runs without overwriting v0.2.

## Research questions

Primary:

> Can a sub-megabyte / few-megabyte recurrent neural system learn reusable semantic state transformations from natural language well enough to solve held-out compositional tasks?

Secondary:

> How does additional recurrent computation compare with additional model size?

v0.3 implements the architecture needed to ask both questions. The first trained run answers the primary question for a 0.56 MB model with **one** state update per event. It does **not** yet sweep size vs inner steps.

## What is and is not learned

The training set is a synthetic curriculum. A symbolic `World` object computes ground-truth quantities so labels are cheap and exact.

That world simulator is **not** part of inference. The model never receives entity tables, operator tags, or the integer answer except as a training target.

Held-out linguistic templates are a disjoint string set from training templates. Unusual probes (`started out with`, `a pair of`, `crayon total`, and similar) are reserved and must not be added to training just to make the demo pass.

Pronouns are disabled (`INCLUDE_PRONOUNS = False`) so a later phase can add them without mixing failure modes.

## Architecture under test

- Shared byte-level bidirectional GRU encoder (events and questions).
- One `WorkingMemoryCell` (GRUCell + LayerNorm) applied to every event.
- Optional `inner_steps` repeats that cell per event with the **same** weights.
- Quantity classification head over `{0..max_answer}`.
- Auxiliary mention head: given an event encoding, predict the number mentioned in that sentence. This pressures number reading; it is not a `gave` rule.

Parameter count does not grow with the number of events.

## Commands

```powershell
tinybrain train-state --semantic-dim 64 --state-dim 64 --inner-steps 1
tinybrain benchmark-state
tinybrain state-demo
```

Size / compute sweep (each run writes `experiments/state_v03.json` and appends a CSV row if you point `--results` at distinct files):

```powershell
foreach ($d in 32,64,128,256) {
  tinybrain train-state --semantic-dim $d --state-dim $d --output tinybrain/models/semantic_state_s$d.pt
  tinybrain benchmark-state --model tinybrain/models/semantic_state_s$d.pt --results experiments/state_s$d.json
}

tinybrain train-state --inner-steps 4 --output tinybrain/models/semantic_state_i4.pt
tinybrain benchmark-state --model tinybrain/models/semantic_state_i4.pt --results experiments/state_i4.json
```

## Run recorded here (v0.3 default)

| Item | Value |
| --- | --- |
| Date | 2026-09-09 |
| Checkpoint | `tinybrain/models/semantic_state.pt` |
| Semantic dim | 64 |
| Working-state dim | 64 |
| Inner steps | 1 |
| Embed / encoder hidden / answer hidden | 48 / 80 / 96 |
| Parameters | 146,850 |
| Parameter MB (fp32) | 0.560 |
| Process RSS during benchmark | 205.7 MB |
| Training | 20 epochs, 600 episodes/stage, AdamW 2e-3, CPU |
| Training time | 474 s |
| Inference | 1.85 ms / episode |
| Mean recurrent state updates | 2.72 |
| Held-out accuracy / param MB | 0.651 |

### Exact-answer accuracy

| Split | n | Accuracy |
| --- | ---: | ---: |
| In-distribution templates | 240 | 49.17% |
| Held-out linguistic templates | 480 | 36.46% |
| Compositional names/objects (train templates, novel names/objects) | 200 | 27.50% |
| Unusual OOD phrasing | 3 | 0.00% |

### Held-out accuracy by curriculum stage

| Stage | Task | Accuracy |
| --- | --- | ---: |
| 1 | Single possession | 78.75% |
| 2 | Possession + one incoming transfer | 43.75% |
| 3 | Possession + one outgoing transfer | 45.00% |
| 4 | Multiple people/objects | 26.25% |
| 5 | Multiple sequential transfers | 7.50% |
| 6 | Distractor facts | 17.50% |

In-distribution Stage 1 is 100%. Stage 2/3 in-distribution are about 70–72%. Stage 5/6 remain near floor even on training templates.

### Permanent Bob / Rebekah benchmark

```text
Bob owns 5 crayons.
Rebekah gave Bob 2 crayons.
How many crayons does Bob have?
expected: 7
predicted: 3
correct: False
state updates: 2
```

This wording is in the training template family. The model still failed. Predicted **3** is `5 - 2`, which is consistent with reversing giver/recipient (treating Bob as the loser of a `gave` event) rather than retrieving a stored sentence.

Do not special-case this example.

### Unusual probes (not in training)

| Input | Expected | Predicted |
| --- | ---: | ---: |
| Bob started out with five crayons. / Rebekah passed a pair of crayons over to Bob. / What's Bob's crayon total now? | 7 | 9 |
| Initially, Maya was in possession of twelve stickers. / Then Maya let Carlos take three stickers. / So how many stickers remain with Maya? | 9 | 12 |
| Priya began with 4 books. / Sam handed Priya 6 books. / Diego owns 8 marbles. / What number of books is Priya sitting on? | 10 | 8 |

Maya's 12 is the initial quantity (ignored transfer). Priya's 8 is Diego's distractor count (last mentioned number). These failures are kept.

## Interpretation

1. **Quantity reading is learnable at this size.** Held-out Stage 1 at ~79% (90.7% on the training-run held-out set at epoch 20) shows the byte encoder can map paraphrases and number words/digits to a quantity class without a parser.
2. **Reusable transfer is only partly learned.** Held-out Stage 2/3 ~44–45%, in-distribution ~70%. The Bob miss (`7` → `3`) is a role-binding error, not retrieval of a canned answer.
3. **A single working-state vector is a bottleneck** once several entities, sequential updates, or distractors appear. Stages 5–6 are unsolved, including on training templates. That is a state-capacity problem, not only a language-template problem.
4. **Template generalization is incomplete.** Gap: in-distribution 49% vs held-out 36% vs unusual 0%. Unusual transfer phrases (`passed a pair`) were withheld on purpose.
5. **Curriculum interference.** After Stage 1 train accuracy passed ~78%, mixing later stages improved some transfer scores but Stage 5/6 never took off. Extra epochs on long chains without a better memory may not be enough.

Success on this synthetic curriculum would still **not** mean general language understanding.

The compute-vs-size sweep below was the v0.3 follow-up. It is **not** v0.4.

---

## v0.3A — Recurrent compute vs state size (Stages 1–3)

Research question:

> Can additional recurrent computation compensate for a smaller latent working state when learning possession-transfer semantics?

Architecture unchanged: one shared encoder, one shared working-memory cell, quantity head. Stages 4–6 were withheld so transfer/role learning could be measured without long-chain collapse. Semantic dim stayed at 64. Bob/Rebekah, unusual probes, and held-out templates were frozen. The exact Bob/Rebekah episode was excluded from training.

### Protocol

- Grid: `state_dim ∈ {32,64,128,256}` × `inner_steps ∈ {1,2,4}` (12 configs)
- Same optimizer (AdamW 2e-3), 20 epochs, 600 episodes/stage, stages 1–3 only
- Primary seed 1337 for all 12; extra seeds 2024 and 4242 on the top 3 configs
- Results: `experiments/state_v03a_sweep.csv` and `.json`
- Sweep checkpoints are local-only (`experiments/sweep_models/`, gitignored)

```powershell
tinybrain sweep-state --extra-seeds
```

### Seed 1337, sorted by held-out Stage 2+3 transfer

| state | inner | MB | ms/ep | S1 HO | S2 HO | S3 HO | S2+3 HO | Bob | unusual | role | xfer/MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4 | 2.091 | 4.00 | 88.8 | 56.2 | 75.0 | **65.62** | 7 PASS | 2/3 | 16.7 | 0.314 |
| 256 | 1 | 2.091 | 2.53 | 81.2 | 53.8 | 61.2 | 57.50 | 7 PASS | 3/3 | 33.3 | 0.275 |
| 32 | 1 | 0.469 | 2.43 | 93.8 | 56.2 | 57.5 | 56.88 | 3 FAIL | 1/3 | 25.0 | **1.212** |
| 64 | 2 | 0.560 | 2.05 | 86.2 | 55.0 | 55.0 | 55.00 | 3 FAIL | 1/3 | 33.3 | 0.982 |
| 32 | 2 | 0.469 | 1.56 | 80.0 | 50.0 | 55.0 | 52.50 | 7 PASS | 2/3 | 33.3 | 1.119 |
| 128 | 2 | 0.883 | 3.12 | 82.5 | 52.5 | 50.0 | 51.25 | 3 FAIL | 1/3 | 16.7 | 0.580 |
| 32 | 4 | 0.469 | 1.49 | 81.2 | 41.2 | 58.8 | 50.00 | 3 FAIL | 0/3 | 33.3 | 1.066 |
| 256 | 2 | 2.091 | 2.82 | 87.5 | 47.5 | 52.5 | 50.00 | 7 PASS | 0/3 | 25.0 | 0.239 |
| 64 | 4 | 0.560 | 2.09 | 82.5 | 43.8 | 55.0 | 49.38 | 3 FAIL | 1/3 | 33.3 | 0.881 |
| 128 | 4 | 0.883 | 3.08 | 93.8 | 48.8 | 47.5 | 48.12 | 3 FAIL | 0/3 | 25.0 | 0.545 |
| 64 | 1 | 0.560 | 2.32 | 88.8 | 41.2 | 52.5 | 46.88 | 7 PASS | 2/3 | 25.0 | 0.837 |
| 128 | 1 | 0.883 | 2.61 | 76.2 | 38.8 | 55.0 | 46.88 | 3 FAIL | 1/3 | 33.3 | 0.531 |

In-distribution Stage 1 is 100% for every config. Mean recurrent updates = `inner_steps × mean events` (~1.67 events).

### Bob / Rebekah (frozen)

Same strings as v0.3. Predicted **3** is reversed transfer (`5-2`). PASS means predicted 7.

| state×inner (seed 1337) | predicted | error |
| --- | ---: | --- |
| 32×1 | 3 | reversed_transfer |
| 32×2 | 7 | exact |
| 32×4 | 3 | reversed_transfer |
| 64×1 | 7 | exact |
| 64×2 | 3 | reversed_transfer |
| 64×4 | 3 | reversed_transfer |
| 128×1 | 3 | reversed_transfer |
| 128×2 | 3 | reversed_transfer |
| 128×4 | 3 | reversed_transfer |
| 256×1 | 7 | exact |
| 256×2 | 7 | exact |
| 256×4 | 7 | exact |

### Role probes (frozen, train-like wording)

Families: A incoming/ask recipient; B outgoing/ask giver; C outgoing/ask recipient who started at 0; D two initialized people, ask the giver.

On seed 1337, **C and D are essentially unsolved** (usually 0/3). A and B sometimes reach 2/3 or 3/3. The network can apply add-or-subtract to a salient person; it does not reliably answer about the other role.

### Transfer error modes (held-out Stage 2+3, seed 1337)

The dominant miss is **reversed_transfer** (~27–45 of 160 items per run). Ignored transfers are rare. That matches the original Bob `7→3` error.

### Seed variance (top 3 configs, 3 seeds)

| config | 1337 | 2024 | 4242 | mean S2+3 HO |
| --- | ---: | ---: | ---: | ---: |
| 256×4 | 65.62 | 55.00 | 53.75 | 58.1 |
| 256×1 | 57.50 | 59.38 | 48.12 | 55.0 |
| 32×1 | 56.88 | 40.00 | 43.12 | 46.7 |

Spreads of 10–17 points. Bob also flips with seed (256×1 seed 2024 predicts 3; 32×1 seeds 2024/4242 predict 7).

### Interpretation

Averaging seed 1337 across widths:

- inner 1 / 2 / 4 transfer: 52.0% / 52.2% / 53.3% — **no material compute gain**
- state 32 / 64 / 128 / 256 transfer: 53.1% / 50.4% / 48.8% / 57.7% — 256 is the best peak, not a clean scaling curve

**CASE A** (more inner steps help at fixed size): not supported. At 32-d, extra steps *hurt*.

**CASE B** (larger state helps, compute does not): weakly, and only as a peak. 32×1 beat 64×1 and 128×1 on the same seed.

**CASE C** (neither fixes role reversal): supported for the diagnostic that matters. Reversed transfer stays the modal error. Role C/D fail in every configuration.

**CASE D** (seed variance): supported. Ranking 32 vs 256 by a single seed would be misleading.

Decision: **do not treat extra recurrent steps as a substitute for a better state representation.** Do not treat 256-d as a solved transfer model. The single-vector state still does not implement queryable source/recipient bindings. That failure is stable across the grid; the accuracy numbers are not.

v0.3B below tests whether addressable slots fix that.

---

## v0.3B — Learned multi-slot working memory

Hypothesis:

> Does a tiny learned multi-slot/associative working memory solve entity-role binding better than a single compressed latent vector, without relying on handcrafted language rules?

This is **not** v0.4. Encoder, curriculum, held-out templates, Bob/Rebekah, unusual probes, and A/B/C/D probes are unchanged. No `gave` rules, no name-to-slot wiring, no giver/recipient auxiliary labels.

### Architecture

```text
event encoder (shared)
    → learned write attention over 4 slots
    → shared GRUCell + gate writes into addressed slots
    → memory persists across events

question encoder (shared)
    → learned read attention over slots
    → quantity head
```

Each slot is 32-d. Addressing is content-based softmax. The same update module is reused for every event. Slots are not labeled as people, objects, or roles.

### Controlled comparison

| | baseline | multi-slot |
| --- | --- | --- |
| memory | 1 vector, 32-d | 4 slots × 32-d |
| inner_steps | 1 | 1 |
| params | 122,978 (0.469 MB) | 129,314 (0.493 MB) |
| epochs / per-stage / stages | 20 / 600 / 1–3 | same |
| seeds | 1337, 2024, 4242 | same |
| optimizer | AdamW 2e-3 | same |

```powershell
tinybrain compare-multislot
```

Results: `experiments/state_v03b_multislot.csv` and `.json`. Checkpoints are local-only (`experiments/multislot_models/`).

### Results (primary metric: frozen role C+D)

| arch | seed | S1 HO | S2 HO | S3 HO | S2+3 HO | C | D | C+D | Bob | OOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1337 | 93.8 | 56.2 | 57.5 | 56.9 | 0.0 | 0.0 | **0.0** | 3 | 1/3 |
| baseline | 2024 | 63.8 | 36.2 | 43.8 | 40.0 | 0.0 | 0.0 | **0.0** | 7 | 1/3 |
| baseline | 4242 | 76.2 | 46.2 | 40.0 | 43.1 | 0.0 | 0.0 | **0.0** | 7 | 1/3 |
| multislot | 1337 | 76.2 | 47.5 | 48.8 | 48.1 | 0.0 | 33.3 | **16.7** | 3 | 2/3 |
| multislot | 2024 | 72.5 | 51.2 | 41.2 | 46.2 | 0.0 | 0.0 | **0.0** | 3 | 1/3 |
| multislot | 4242 | 71.2 | 51.2 | 38.8 | 45.0 | 0.0 | 0.0 | **0.0** | 7 | 1/3 |

Mean S2+3: baseline **46.7%**, multi-slot **46.5%**. Mean C+D: baseline **0%**, multi-slot **5.6%** (one seed got 1/3 on family D).

Role C (ask the recipient who started at zero) is **0/3 on every seed of both architectures**.

### Bob / Rebekah

| arch | 1337 | 2024 | 4242 |
| --- | ---: | ---: | ---: |
| baseline | 3 reversed | 7 | 7 |
| multislot | 3 reversed | 3 reversed | 7 |

### Error types (held-out Stage 2+3 reversed_transfer counts / 160)

Baseline: 40, 39, 34. Multi-slot: 27, 31, 33. Reversal remains the modal miss. Ignored transfers stay rare (2–4).

### Slot-attention diagnostics

On trained multi-slot models, write and read attention **collapse onto a single slot** (weights ≈ 1.0 on one index, ~0 on the others). Later events overwrite that same slot. Question readout uses the same slot.

The extra slots exist in the tensor but are not used as separable entity stores. This is not interpreted as “slot 0 is Bob”; it is evidence that learned addressing did not separate entities.

### Cost

RSS ~325 MB for both. Train ~3.5–4 min per seed. Inference ~1.2–1.6 ms/episode. Multi-slot is +6.3k params (+0.024 MB).

### Interpretation

**CASE A** (C/D strongly improves across seeds): not supported.

**CASE B** (aggregate up, C/D still poor): aggregate did not rise. C/D remains poor.

**CASE C** (no meaningful role-binding gain): **supported.** Four unlabeled slots with learned attention did not solve source/recipient queries.

**CASE D** (one lucky seed): S2+3 and Bob still vary by seed. C/D failure is stable.

Do not add more slots on this evidence. The model already ignores three of four.

### Next experiment (still not v0.4)

Target **semantic role / relational binding** in the update itself: the network needs a learned distinction between who loses and who gains, not just more addresses. That must still be learned from answers, not a `gave` parser. Role-auxiliary labels are a later option if an unsupervised relational update also fails.

Keep Bob/Rebekah and C/D frozen.

v0.2 remains the intent-routing baseline.

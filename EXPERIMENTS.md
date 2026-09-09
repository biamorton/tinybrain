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

## Recommended next experiment (not v0.4)

Do not jump to free-form dialogue, pronouns as the main bet, or wrapping an LLM.

The evidence says the encoder can read quantities. The failure is **learned state update and multi-fact binding**.

Recommended v0.3-line follow-up, in order:

1. **Compute vs size sweep on Stages 2–3 only**, until in-distribution transfer is clearly saturated. Compare `inner_steps` 1/2/4 at 64-d against `state_dim` 32/64/128/256 with `inner_steps=1`. Same held-out templates. This answers the secondary research question with a task the 64-d model can at least partially learn.
2. If extra inner steps do not fix giver/recipient errors, test a **learned multi-slot / associative working memory** that still reuses one update module per event. Still no `gave` rules.
3. Keep the Bob/Rebekah item and the unusual probes frozen. If they start passing, it should be because the update operator generalized, not because those strings entered training.

v0.2 remains the intent-routing baseline. Do not spend the next iteration only improving that classifier.

# TinyBrain

Long-term goal:

> Build a continually learning personal AI capable of replacing a conventional LLM for most everyday computing tasks while running locally on ordinary consumer hardware, using dramatically less active memory and compute.

This is an experimental architecture, not a wrapper around Qwen, OpenAI, or another large language model.

TinyBrain is **not** supposed to become a pile of if/else language rules, a regex semantic parser, a lookup table of answers, a database of precomputed answers, or a RAG wrapper that pretends retrieval is intelligence.

Deterministic specialist tools are allowed where appropriate (calculator, code execution). Decisions about language and when to use a tool should ultimately be learned.

## Version line

| Version | What it tested | Status |
| --- | --- | --- |
| v0.1 | Random latent skeleton + heuristic router | Historical baseline |
| v0.2 | Learned byte-level intent / language routing | Preserved |
| v0.3 | Learned semantic state + recurrent working memory | Current experiment |

v0.3 does **not** replace the v0.2 classifier. The two experiments sit side by side so results can be compared.

---

## v0.3 — Learned semantic state

### What this version is testing

v0.2 showed that a tiny classifier can route utterances, but classification is too shallow. The motivating failure:

1. `Bob owns 5 crayons.` is stored.
2. `Rebekah gave Bob 2 crayons.` is misread (in v0.2, often as a greeting).
3. `How many crayons does Bob have?` retrieves the first sentence instead of computing **7**.

v0.3 asks a different question:

> Can a sub-megabyte / few-megabyte recurrent network learn reusable semantic state transformations from natural language well enough to solve held-out compositional tasks?

A secondary question:

> How does additional recurrent computation compare with additional model size?

The same working-memory cell is reused for every event. Sequence length is extra iterative computation, not extra parameters.

### Architecture

```text
sentence
    ↓
shared learned byte-level encoder
    ↓
sentence semantic vector
    ↓
shared working-memory cell  (same weights for event 1, 2, 3, ...)
    ↓
persistent latent working state

next sentence uses the same encoder + the same cell

question
    ↓
shared encoder + working state
    ↓
learned quantity head
    ↓
answer (integer 0..64)
```

There are **no** hand-written rules such as `if verb == "gave": recipient += quantity`.

Labels for training are produced by a symbolic world simulator. That simulator is **not** used at inference time. The network only sees text.

A small auxiliary head asks the encoder to read the quantity mentioned in each event. That is number extraction, not a transfer rule.

Pronouns are an explicit extension point and are **not** generated in v0.3.

### First permanent benchmark

Context:

- `Bob owns 5 crayons.`
- `Rebekah gave Bob 2 crayons.`

Question:

- `How many crayons does Bob have?`

Expected: `7`

This example is **not** special-cased. It uses the same `infer(events, question)` path as every other episode. The answer `7` is not stored.

### Train

```powershell
tinybrain train-state
```

Useful knobs:

```powershell
tinybrain train-state --semantic-dim 64 --state-dim 64 --inner-steps 1
tinybrain train-state --semantic-dim 128 --state-dim 128 --inner-steps 1
tinybrain train-state --semantic-dim 64 --state-dim 64 --inner-steps 4
```

Default configuration used for the bundled v0.3 checkpoint:

- semantic dim 64, working-state dim 64, inner steps 1
- 20 epochs, 600 episodes per curriculum stage
- progressive stages: possession → incoming transfer → outgoing transfer → multiple entities → sequential transfers → distractors

Held-out sentence templates are never used in training.

### Benchmark and demo

```powershell
tinybrain benchmark-state
tinybrain state-demo
```

`benchmark-state` reports exact-answer accuracy, held-out linguistic accuracy, accuracy by curriculum stage, parameter count / MB, process RSS, training time, inference latency, recurrent state updates, and accuracy per parameter MB. Results are written to `experiments/state_v03.json` and `.csv`.

### Current measured result

Bundled model: **146,850 parameters (0.56 MB)**.

| Split | Exact accuracy |
| --- | ---: |
| In-distribution templates | 49.17% |
| Held-out linguistic templates | 36.46% |
| Compositional names/objects | 27.50% |
| Unusual OOD phrasing | 0.00% |
| Stage 1 possession (held-out) | 78.75% |
| Stage 2 incoming transfer (held-out) | 43.75% |
| Stage 5 sequential transfers (held-out) | 7.50% |
| Bob / Rebekah permanent benchmark | **FAIL** (predicted 3, expected 7) |

This is **not** solved. It is useful evidence. See [EXPERIMENTS.md](EXPERIMENTS.md) for v0.3 and the v0.3A compute-vs-size sweep.

v0.3A (Stages 1–3 only, 12 configs): extra `inner_steps` did **not** reliably replace a larger state. Role-reversal remains the main transfer error. Bob/Rebekah still fails on several configs (predicted 3). Sweep: `tinybrain sweep-state`.

v0.3B (4 learned 32-d slots vs 32-d single vector, 3 seeds): **C/D role probes did not improve.** Attention collapsed onto one slot. Compare: `tinybrain compare-multislot`.

v0.3C (4 unlabeled event components + symmetric questioning, 3 seeds): **components collapsed to copies of one vector.** Symmetric questions moved C/D off zero but not stably. Compare: `tinybrain compare-relational`.

v0.3D (training-only role aux vs answer-only, 3 seeds): **aux labels did not produce stable C/D generalization.** Compare: `tinybrain compare-roleaux`.

### Current limitations

- A single latent vector does not reliably bind several people and objects at once. Four learned unlabeled slots (v0.3B) and four unlabeled event components (v0.3C) both collapsed. Training-only role labels (v0.3D) did not stabilize C/D either.
- One incoming/outgoing transfer is only partially learned; longer chains collapse.
- Unusual wording (`started out with`, `a pair of`, `crayon total`) was **not** added to training and currently fails. That failure is kept.
- Success on synthetic possession/transfer does **not** mean general language understanding.
- v0.3 does not replace v0.2's intent router, does not generate free text, and does not call an LLM.

---

## v0.2 — Learned language routing (preserved)

Natural language is encoded as UTF-8 bytes, passed through learned embeddings and a bidirectional GRU, compressed into a 96-dimensional semantic vector, and classified by a learned intent head.

A trained `tinybrain/models/semantic_router.pt` checkpoint is included.

```powershell
tinybrain ask "Hi"
tinybrain ask "what is square root of 9?"
tinybrain interpret "Bob owns 5 crayons."
tinybrain benchmark-language
tinybrain train-language
```

Held-out intent accuracy on the bundled classifier is in the mid-50% range. That is not “language solved.” Do not fix it by stuffing hidden `if` statements into the runtime.

---

## Research rules

1. No runtime semantic keyword router for the learned experiments.
2. Keep held-out wording separate from training.
3. Measure unseen-language performance, RAM, model size, and latency.
4. Do not hide failures with handcrafted semantic rules.
5. Deterministic specialist execution is allowed after a learned decision.
6. Do not call retrieval “learning.”
7. Do not call intent classification “language solved.”
8. If a benchmark exposes a weakness, improve the learning method rather than the benchmark.

import math
import random

import torch

from tinybrain.core.relational import (
    RelationalDecomposer,
    RelationalWorkingMemory,
    attention_entropy,
    collapse_metrics,
    pairwise_cosine_mean,
)
from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.training.state_data import (
    DEMO_EPISODE,
    RELATIONAL_PROBES,
    generate_curriculum,
    generate_episode,
    generate_episode_family,
    is_frozen_episode,
)


def _rel_model() -> SemanticStateModel:
    return SemanticStateModel(
        StateModelConfig(
            embed_dim=16,
            encoder_hidden=24,
            semantic_dim=32,
            state_dim=32,
            component_dim=32,
            n_components=4,
            answer_hidden=48,
            max_answer=20,
            inner_steps=1,
        )
    )


def test_component_tensor_shapes():
    dec = RelationalDecomposer(encoder_dim=16, n_components=4, component_dim=8)
    seq = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    components, attn = dec.decompose(seq, mask)
    assert components.shape == (2, 4, 8)
    assert attn.shape == (2, 4, 5)
    related, rel_w = dec.relate(components)
    assert related.shape == (2, 4, 8)
    assert rel_w.shape == (2, 4, 4)
    mem = RelationalWorkingMemory(n_components=4, component_dim=8, question_dim=16)
    memory = torch.zeros(2, 4, 8)
    updated, write_w, delta = mem.write(memory, related)
    assert updated.shape == (2, 4, 8)
    assert write_w.shape == (2, 4, 4)
    assert delta.shape == (2, 4)
    q = torch.randn(2, 16)
    read_w, readout = mem.read(updated, q)
    assert read_w.shape == (2, 4)
    assert readout.shape == (2, 8)


def test_attention_normalizes_over_tokens_and_components():
    dec = RelationalDecomposer(encoder_dim=10, n_components=4, component_dim=8)
    seq = torch.randn(3, 6, 10)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    _, attn = dec.decompose(seq, mask)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3, 4), atol=1e-5)
    assert torch.all(attn >= 0)
    related, rel_w = dec.relate(torch.randn(3, 4, 8))
    assert torch.allclose(rel_w.sum(dim=-1), torch.ones(3, 4), atol=1e-5)
    _ = related


def test_shared_update_path_independent_of_event_count():
    model = _rel_model()
    n = model.parameter_count()
    model.infer(["Bob owns 5 crayons."], "How many crayons does Bob have?")
    model.infer(DEMO_EPISODE.events, DEMO_EPISODE.question)
    model.infer(
        DEMO_EPISODE.events + ["Maya owns 3 stickers."],
        "How many crayons does Bob have?",
    )
    assert model.parameter_count() == n
    assert model.use_relational
    assert model.relational is not None
    assert not hasattr(model, "component_0")


def test_variable_episode_lengths_in_one_batch():
    model = _rel_model()
    logits, n_updates, _ = model(
        [
            ["Bob owns 5 crayons."],
            DEMO_EPISODE.events,
        ],
        [
            "How many crayons does Bob have?",
            DEMO_EPISODE.question,
        ],
    )
    assert logits.shape == (2, model.config.max_answer + 1)
    assert int(n_updates[0].item()) == 1
    assert int(n_updates[1].item()) == 2


def test_collapse_metrics_detect_identical_components():
    identical = torch.ones(1, 4, 8)
    collapsed = collapse_metrics(components=identical)
    assert collapsed["mean_pairwise_cosine"] > 0.99
    diverse = torch.eye(4).unsqueeze(0)
    diverse = torch.nn.functional.pad(diverse, (0, 4))
    spread = collapse_metrics(components=diverse)
    assert spread["mean_pairwise_cosine"] < 0.05
    masses = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    attn = collapse_metrics(read_attn=masses)
    assert attn["dominant_read_mass"] == 1.0
    assert attn["read_attn_entropy"] < 0.01
    uniform = torch.full((1, 4), 0.25)
    ent = collapse_metrics(read_attn=uniform)
    assert abs(ent["read_attn_entropy"] - 1.0) < 1e-5


def test_pairwise_cosine_and_entropy_helpers():
    eye = torch.eye(3)
    assert pairwise_cosine_mean(eye).item() < 1e-5
    uniform = torch.full((2, 4), 0.25)
    assert abs(attention_entropy(uniform).mean().item() - 1.0) < 1e-5
    assert math.isfinite(float(attention_entropy(torch.tensor([[1.0, 0.0, 0.0]])).item()))


def test_symmetric_family_queries_both_participants():
    rng = random.Random(9)
    family = generate_episode_family(2, rng, symmetric=True)
    assert len(family) == 2
    assert family[0].events == family[1].events
    assert family[0].question != family[1].question
    assert family[0].answer != family[1].answer
    names = {family[0].target_name, family[1].target_name}
    assert len(names) == 2
    # Quantity is conserved when both participants were initialized.
    if family[0].initial_qty > 0 and family[1].initial_qty > 0:
        assert family[0].answer + family[1].answer == (
            family[0].initial_qty + family[1].initial_qty
        )


def test_symmetric_curriculum_excludes_frozen_and_varies_questions():
    train = generate_curriculum(40, held_out=False, seed=202, stages=(1, 2, 3), symmetric=True)
    assert not any(is_frozen_episode(ep) for ep in train)
    demo_key = (tuple(DEMO_EPISODE.events), DEMO_EPISODE.question)
    assert all((tuple(ep.events), ep.question) != demo_key for ep in train)
    for ep in RELATIONAL_PROBES:
        assert not any((tuple(x.events), x.question) == (tuple(ep.events), ep.question) for x in train)
    s2 = [ep for ep in train if ep.stage == 2]
    assert len(s2) == 80
    questions = {ep.question for ep in s2}
    assert len(questions) > 2


def test_asymmetric_generate_episode_unchanged():
    rng = random.Random(44)
    s2 = generate_episode(2, rng)
    assert len(s2.events) == 2
    assert s2.transfer_direction == "in"


def test_relational_probes_are_frozen():
    assert all(is_frozen_episode(ep) for ep in RELATIONAL_PROBES)
    assert len(RELATIONAL_PROBES) == 8


def test_infer_trace_has_component_diagnostics():
    model = _rel_model()
    pred, n_updates, trace = model.infer_trace(DEMO_EPISODE.events, DEMO_EPISODE.question)
    assert n_updates == 2
    assert 0 <= pred <= model.config.max_answer
    assert trace["n_components"] == 4
    assert len(trace["event_write_attention"]) == 2
    assert abs(sum(trace["question_read_attention"]) - 1.0) < 1e-4
    assert "collapse" in trace
    assert "mean_pairwise_cosine" in trace["collapse"]

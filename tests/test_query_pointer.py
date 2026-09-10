import torch

from tinybrain.core.query_pointer import QueryPointerReadout, query_collapse_metrics
from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.training.state_data import (
    COUNTERFACTUAL_PROBES,
    DEMO_EPISODE,
    generate_curriculum,
    is_frozen_episode,
)


def _ptr_model() -> SemanticStateModel:
    return SemanticStateModel(
        StateModelConfig(
            embed_dim=16,
            encoder_hidden=24,
            semantic_dim=32,
            state_dim=16,
            answer_hidden=48,
            max_answer=20,
            query_pointer=True,
        )
    )


def test_retained_token_shapes():
    ptr = QueryPointerReadout(encoder_dim=10, query_dim=8)
    q_seq = torch.randn(2, 5, 10)
    q_mask = torch.ones(2, 5, dtype=torch.bool)
    query, q_attn = ptr.pool_question(q_seq, q_mask)
    assert query.shape == (2, 8)
    assert q_attn.shape == (2, 5)
    ev = torch.randn(2, 7, 10)
    ev_mask = torch.ones(2, 7, dtype=torch.bool)
    readout, ev_attn = ptr.read_events(ev, ev_mask, query)
    assert readout.shape == (2, 8)
    assert ev_attn.shape == (2, 7)


def test_cross_attention_normalizes():
    ptr = QueryPointerReadout(encoder_dim=12, query_dim=6)
    ev = torch.randn(3, 8, 12)
    mask = torch.tensor(
        [
            [True, True, True, False, False, False, False, False],
            [True, True, True, True, True, False, False, False],
            [True] * 8,
        ]
    )
    query = torch.randn(3, 6)
    _, attn = ptr.read_events(ev, mask, query)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)
    assert torch.all(attn >= 0)
    q_seq = torch.randn(3, 4, 12)
    q_mask = torch.ones(3, 4, dtype=torch.bool)
    _, q_attn = ptr.pool_question(q_seq, q_mask)
    assert torch.allclose(q_attn.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_variable_event_lengths():
    model = _ptr_model()
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


def test_same_context_different_question_is_independent_query_path():
    model = _ptr_model()
    events = COUNTERFACTUAL_PROBES[0].events
    _, _, t_bob = model.infer_trace(events, "How many crayons does Bob have?")
    _, _, t_reb = model.infer_trace(events, "How many crayons does Rebekah have?")
    assert t_bob["query_pointer"] is True
    assert t_reb["query_pointer"] is True
    assert t_bob["query"] != t_reb["query"]
    bob_q = torch.tensor(t_bob["query"])
    reb_q = torch.tensor(t_reb["query"])
    assert not torch.allclose(bob_q, reb_q, atol=1e-6)
    n = model.parameter_count()
    model.infer(events, "How many crayons does Bob have?")
    model.infer(events, "How many crayons does Rebekah have?")
    assert model.parameter_count() == n
    assert not hasattr(model, "name_table")


def test_counterfactual_probes_are_frozen():
    assert len(COUNTERFACTUAL_PROBES) == 8
    assert all(is_frozen_episode(ep) for ep in COUNTERFACTUAL_PROBES)
    train = generate_curriculum(30, held_out=False, seed=88, stages=(1, 2, 3), symmetric=True)
    keys = {(tuple(ep.events), ep.question) for ep in COUNTERFACTUAL_PROBES}
    assert all((tuple(x.events), x.question) not in keys for x in train)


def test_query_collapse_helper():
    same = torch.ones(1, 4)
    vec = torch.ones(1, 8)
    collapsed = query_collapse_metrics(same, same, vec, vec)
    assert collapsed["query_collapse"] is True
    assert collapsed["readout_cosine"] > 0.99
    other = torch.tensor([[1.0, -1.0, 0.0, 0.0]])
    other_v = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    spread = query_collapse_metrics(same, other, vec, other_v)
    assert spread["readout_cosine"] < 0.9

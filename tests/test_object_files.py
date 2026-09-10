import torch

from tinybrain.core.object_files import (
    ObjectFileMemory,
    collate_spans,
    lexical_spans,
    pool_spans,
    write_collapse,
)
from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.training.state_data import COUNTERFACTUAL_PROBES, DEMO_EPISODE


def _of_model() -> SemanticStateModel:
    return SemanticStateModel(
        StateModelConfig(
            embed_dim=16,
            encoder_hidden=24,
            semantic_dim=32,
            state_dim=16,
            answer_hidden=48,
            max_answer=20,
            object_files=True,
            n_object_files=6,
            object_dim=16,
        )
    )


def test_local_units_preserve_order_and_locality():
    text = "Rebekah gave Bob 2 crayons."
    spans = lexical_spans(text)
    tokens = [tok for tok, _, _ in spans]
    assert tokens == ["Rebekah", "gave", "Bob", "2", "crayons"]
    raw = text.encode("utf-8")
    for tok, start, end in spans:
        assert raw[start:end].decode("utf-8") == tok
    assert spans[0][1] < spans[1][1] < spans[2][1]


def test_pool_spans_uses_distinct_regions():
    seq = torch.zeros(1, 8, 4)
    seq[0, 0:3] = 1.0
    seq[0, 3:6] = 2.0
    starts = torch.tensor([[0, 3]])
    ends = torch.tensor([[3, 6]])
    mask = torch.tensor([[True, True]])
    pooled = pool_spans(seq, starts, ends, mask)
    assert pooled.shape == (1, 2, 4)
    assert torch.allclose(pooled[0, 0], torch.ones(4))
    assert torch.allclose(pooled[0, 1], torch.full((4,), 2.0))


def test_object_file_shapes_and_shared_update():
    mem = ObjectFileMemory(encoder_dim=10, object_dim=8, n_files=6)
    keys, values, usage = mem.new_memory(2, torch.device("cpu"))
    assert keys.shape == (2, 6, 8)
    assert values.shape == (2, 6, 8)
    seq = torch.randn(2, 9, 10)
    starts = torch.tensor([[0, 3, 6], [1, 4, 0]])
    ends = torch.tensor([[3, 6, 9], [4, 8, 1]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    keys, values, usage, trace = mem.write_event(keys, values, usage, seq, starts, ends, mask)
    assert keys.shape == (2, 6, 8)
    assert values.shape == (2, 6, 8)
    assert usage.shape == (2, 6)
    assert trace["write_mass"].shape == (2, 3, 6)
    assert torch.allclose(trace["write_mass"].sum(dim=-1), trace["salience"], atol=1e-5)
    n = sum(p.numel() for p in mem.parameters())
    keys, values, usage, _ = mem.write_event(keys, values, usage, seq, starts, ends, mask)
    assert sum(p.numel() for p in mem.parameters()) == n
    assert not hasattr(mem, "file_0")


def test_addressing_normalizes_and_unused_files_exist():
    mem = ObjectFileMemory(encoder_dim=12, object_dim=6, n_files=4)
    keys, values, usage = mem.new_memory(3, torch.device("cpu"))
    assert torch.all(usage == 0)
    seq = torch.randn(3, 6, 12)
    starts, ends, mask, tokens = collate_spans(
        ["Bob has 5.", "Maya owns 2 stickers", ""],
        max_bytes=32,
        max_units=8,
    )
    assert tokens[0][0] == "Bob"
    assert tokens[2] == []
    keys, values, usage, trace = mem.write_event(keys, values, usage, seq, starts, ends, mask)
    mass = trace["write_mass"]
    for b in range(3):
        for u, present in enumerate(mask[b].tolist()):
            if present:
                assert mass[b, u].sum() >= 0
    read_w, query, readout, q_attn = mem.read(
        keys, values, usage, seq, starts, ends, mask
    )
    assert read_w.shape == (3, 4)
    assert torch.allclose(read_w.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert query.shape == (3, 6)
    assert readout.shape == (3, 6)


def test_variable_units_and_multiple_events():
    model = _of_model()
    logits, n_updates, mention = model(
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
    assert mention.shape[0] == 2
    n = model.parameter_count()
    model.infer(DEMO_EPISODE.events, DEMO_EPISODE.question)
    assert model.parameter_count() == n
    assert isinstance(model.memory, ObjectFileMemory)


def test_same_context_different_question_independent_read():
    model = _of_model()
    events = COUNTERFACTUAL_PROBES[0].events
    _, _, t_bob = model.infer_trace(events, "How many crayons does Bob have?")
    _, _, t_reb = model.infer_trace(events, "How many crayons does Rebekah have?")
    assert t_bob["object_files"] is True
    assert t_bob["local_units"] == t_reb["local_units"]
    assert t_bob["query"] != t_reb["query"]
    assert not torch.allclose(torch.tensor(t_bob["query"]), torch.tensor(t_reb["query"]), atol=1e-6)
    assert t_bob["question_units"] != t_reb["question_units"]
    assert "file_usage" in t_bob
    assert len(t_bob["file_usage"]) == 6


def test_write_collapse_helper():
    mass = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    assert float(write_collapse(mass).item()) == 1.0
    spread = torch.tensor([[0.5, 0.5, 0.0]])
    assert abs(float(write_collapse(spread).item()) - 0.5) < 1e-5


def test_identity_map_is_diagnostic_only():
    from tinybrain.eval.state_object_files import identity_from_map

    separated = identity_from_map({"Bob": [2, 2], "Rebekah": [4, 4]})
    assert separated["identity_consistency"] == 1.0
    assert separated["identity_separation"] == 1.0
    assert separated["over_merging"] == 0.0
    merged = identity_from_map({"Bob": [1, 2], "Rebekah": [1, 1]})
    assert merged["identity_drift"] == 0.5
    assert merged["over_merging"] == 1.0


def test_object_files_rejected_with_pointer():
    try:
        SemanticStateModel(StateModelConfig(query_pointer=True, object_files=True))
    except ValueError as exc:
        assert "object_files" in str(exc)
    else:
        raise AssertionError("expected ValueError")


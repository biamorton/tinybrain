import torch

from tinybrain.core.slot_memory import MultiSlotMemory
from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.training.state_data import DEMO_EPISODE


def _slot_model() -> SemanticStateModel:
    return SemanticStateModel(
        StateModelConfig(
            embed_dim=16,
            encoder_hidden=24,
            semantic_dim=32,
            state_dim=32,
            slot_dim=32,
            n_slots=4,
            answer_hidden=48,
            max_answer=20,
            inner_steps=1,
        )
    )


def test_slot_tensor_shapes():
    mem = MultiSlotMemory(semantic_dim=32, slot_dim=32, n_slots=4)
    slots = torch.zeros(2, 4, 32)
    semantic = torch.randn(2, 32)
    updated, weights, delta = mem.write(slots, semantic)
    assert updated.shape == (2, 4, 32)
    assert weights.shape == (2, 4)
    assert delta.shape == (2, 4)
    read_w, readout = mem.read(updated, semantic)
    assert read_w.shape == (2, 4)
    assert readout.shape == (2, 32)


def test_attention_normalizes():
    mem = MultiSlotMemory(semantic_dim=16, slot_dim=8, n_slots=4)
    slots = torch.randn(3, 4, 8)
    query = torch.randn(3, 8)
    weights, _ = mem.attend(slots, query)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)
    assert torch.all(weights >= 0)


def test_shared_update_path_independent_of_event_count():
    model = _slot_model()
    n = model.parameter_count()
    model.infer(["Bob owns 5 crayons."], "How many crayons does Bob have?")
    model.infer(DEMO_EPISODE.events, DEMO_EPISODE.question)
    model.infer(
        DEMO_EPISODE.events + ["Maya owns 3 stickers."],
        "How many crayons does Bob have?",
    )
    assert model.parameter_count() == n
    assert isinstance(model.memory, MultiSlotMemory)
    assert not hasattr(model, "memory_slot_0")


def test_variable_episode_lengths_in_one_batch():
    model = _slot_model()
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


def test_infer_trace_has_attention():
    model = _slot_model()
    pred, n_updates, trace = model.infer_trace(DEMO_EPISODE.events, DEMO_EPISODE.question)
    assert n_updates == 2
    assert 0 <= pred <= model.config.max_answer
    assert trace["n_slots"] == 4
    assert len(trace["event_write_attention"]) == 2
    assert abs(sum(trace["event_write_attention"][0]) - 1.0) < 1e-4
    assert abs(sum(trace["question_read_attention"]) - 1.0) < 1e-4

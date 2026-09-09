import torch
from torch import nn

from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.training.state_data import DEMO_EPISODE


def _tiny_model() -> SemanticStateModel:
    cfg = StateModelConfig(
        embed_dim=16,
        encoder_hidden=24,
        semantic_dim=32,
        state_dim=32,
        answer_hidden=48,
        max_answer=20,
        inner_steps=1,
    )
    return SemanticStateModel(cfg)


def test_parameter_count_independent_of_event_length():
    model = _tiny_model()
    n = model.parameter_count()
    model.infer(["Bob owns 5 crayons."], "How many crayons does Bob have?")
    model.infer(
        [
            "Bob owns 5 crayons.",
            "Rebekah gave Bob 2 crayons.",
            "Maya owns 3 stickers.",
            "Sam handed Bob 1 crayons.",
        ],
        "How many crayons does Bob have?",
    )
    assert model.parameter_count() == n
    memory_params = sum(p.numel() for p in model.memory.parameters())
    assert memory_params > 0
    assert not hasattr(model, "memory_1")
    assert not hasattr(model, "memory_2")


def test_shared_encoder_used_for_events_and_questions():
    model = _tiny_model()
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    assert encoder_ids == {id(p) for p in model.encoder.parameters()}
    logits, n_updates, mention_logits = model(
        [["Bob owns 5 crayons.", "Rebekah gave Bob 2 crayons."]],
        ["How many crayons does Bob have?"],
    )
    assert logits.shape == (1, model.config.max_answer + 1)
    assert int(n_updates[0].item()) == 2
    assert mention_logits.shape[0] == 1


def test_inner_steps_multiply_updates_without_new_parameters():
    cfg = StateModelConfig(
        embed_dim=16,
        encoder_hidden=24,
        semantic_dim=32,
        state_dim=32,
        answer_hidden=48,
        max_answer=20,
        inner_steps=4,
    )
    model = SemanticStateModel(cfg)
    baseline = _tiny_model()
    assert model.parameter_count() == baseline.parameter_count()
    _, n_updates, _ = model(
        [["Bob owns 5 crayons.", "Rebekah gave Bob 2 crayons."]],
        ["How many crayons does Bob have?"],
    )
    assert int(n_updates[0].item()) == 8


def test_overfit_single_possession():
    torch.manual_seed(0)
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    events = ["Bob owns 5 crayons."]
    question = "How many crayons does Bob have?"
    target = torch.tensor([5], dtype=torch.long)
    for _ in range(120):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits, _, _ = model([events], [question])
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
    predicted, n_updates = model.infer(events, question)
    assert n_updates == 1
    assert predicted == 5


def test_demo_uses_generic_infer_path():
    model = _tiny_model()
    predicted, n_updates = model.infer(DEMO_EPISODE.events, DEMO_EPISODE.question)
    assert n_updates == 2
    assert isinstance(predicted, int)
    assert 0 <= predicted <= model.config.max_answer

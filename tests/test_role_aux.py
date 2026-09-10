import torch
from torch import nn

from tinybrain.core.state_model import SemanticStateModel, StateModelConfig
from tinybrain.training.state_data import DEMO_EPISODE, generate_episode
from tinybrain.training.train_state import role_targets, xfer_qty_targets
import random


def _aux_model() -> SemanticStateModel:
    return SemanticStateModel(
        StateModelConfig(
            embed_dim=16,
            encoder_hidden=24,
            semantic_dim=32,
            state_dim=32,
            answer_hidden=48,
            max_answer=20,
            role_aux=True,
        )
    )


def test_role_aux_heads_exist_only_when_enabled():
    plain = SemanticStateModel(StateModelConfig(embed_dim=16, encoder_hidden=24, semantic_dim=32, state_dim=32, answer_hidden=32, max_answer=20))
    aux = _aux_model()
    assert plain.role_head is None
    assert aux.role_head is not None
    assert aux.parameter_count() > plain.parameter_count()


def test_role_aux_shapes_and_infer_takes_raw_text_only():
    model = _aux_model()
    logits, n_updates, mention, aux = model(
        [DEMO_EPISODE.events],
        [DEMO_EPISODE.question],
        return_aux=True,
    )
    assert logits.shape[0] == 1
    assert aux["role_logits"].shape == (1, 3)
    assert aux["xfer_qty_logits"].shape == (1, model.config.max_answer + 1)
    predicted, steps = model.infer(DEMO_EPISODE.events, DEMO_EPISODE.question)
    assert steps == 2
    assert isinstance(predicted, int)


def test_role_targets_come_from_simulator_not_text():
    rng = random.Random(3)
    s1 = generate_episode(1, rng)
    s2 = generate_episode(2, rng)
    s3 = generate_episode(3, rng)
    device = torch.device("cpu")
    y = role_targets([s1, s2, s3], device)
    assert int(y[0].item()) == 0
    assert int(y[1].item()) == 1
    assert int(y[2].item()) == 2
    xfer = xfer_qty_targets([s1, s2, s3], 64, device)
    assert int(xfer[0].item()) == -100
    assert int(xfer[1].item()) == s2.transfer_qty
    assert int(xfer[2].item()) == s3.transfer_qty


def test_role_aux_not_required_for_default_forward():
    model = _aux_model()
    logits, n_updates, mention = model([DEMO_EPISODE.events], [DEMO_EPISODE.question])
    assert logits.shape[1] == model.config.max_answer + 1
    assert int(n_updates[0].item()) == 2
    _ = mention


def test_xfer_qty_loss_skips_possession_only_batches():
    device = torch.device("cpu")
    rng = random.Random(1)
    batch = [generate_episode(1, rng) for _ in range(4)]
    y = xfer_qty_targets(batch, 64, device)
    assert torch.all(y == -100)

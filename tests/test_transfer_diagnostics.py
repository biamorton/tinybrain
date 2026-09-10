from tinybrain.eval.state_sweep import SWEEP_INNER_STEPS, SWEEP_STATE_DIMS, configs
from tinybrain.eval.transfer_diagnostics import classify_prediction, reversed_quantity
from tinybrain.training.state_data import DEMO_EPISODE, ROLE_PROBES, UNUSUAL_PROBES
from tinybrain.training.train_state import stages_for_epoch


def test_bob_error_modes():
    assert classify_prediction(DEMO_EPISODE, 7) == "exact"
    assert classify_prediction(DEMO_EPISODE, 5) == "ignored_transfer"
    assert classify_prediction(DEMO_EPISODE, 3) == "reversed_transfer"
    assert reversed_quantity(DEMO_EPISODE) == 3


def test_outgoing_error_modes():
    probe = next(ep for ep in ROLE_PROBES if ep.probe_family == "B" and ep.target_name == "Bob")
    assert probe.answer == 3
    assert classify_prediction(probe, 3) == "exact"
    assert classify_prediction(probe, 5) == "ignored_transfer"
    assert classify_prediction(probe, 7) == "reversed_transfer"


def test_curriculum_never_includes_stages_above_max():
    assert stages_for_epoch(20, 20, max_stage=3) == (1, 2, 3)
    assert 4 not in stages_for_epoch(20, 20, max_stage=3)
    assert stages_for_epoch(1, 20, max_stage=3) == (1,)


def test_unusual_and_role_probes_are_distinct_from_demo():
    demo_key = (tuple(DEMO_EPISODE.events), DEMO_EPISODE.question)
    for ep in UNUSUAL_PROBES + ROLE_PROBES:
        assert (tuple(ep.events), ep.question) != demo_key


def test_role_probe_families_cover_source_and_recipient():
    families = {ep.probe_family for ep in ROLE_PROBES}
    assert families == {"A", "B", "C", "D"}
    assert all(ep.held_out for ep in ROLE_PROBES)


def test_sweep_grid_has_twelve_configs():
    assert configs(SWEEP_STATE_DIMS, SWEEP_INNER_STEPS) == [
        (32, 1),
        (32, 2),
        (32, 4),
        (64, 1),
        (64, 2),
        (64, 4),
        (128, 1),
        (128, 2),
        (128, 4),
        (256, 1),
        (256, 2),
        (256, 4),
    ]

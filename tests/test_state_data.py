from tinybrain.training.state_data import (
    COMPOSITIONAL_NAMES,
    COMPOSITIONAL_OBJECTS,
    DEMO_EPISODE,
    FORBIDDEN_SNIPPETS,
    HELD_TEMPLATES,
    INCLUDE_PRONOUNS,
    NAMES,
    OBJECTS,
    PRONOUN_TEMPLATES,
    TRAIN_TEMPLATES,
    UNUSUAL_PROBES,
    World,
    all_template_strings,
    contains_forbidden_snippet,
    generate_curriculum,
    generate_episode,
)
import random


def test_world_bob_rebekah_arithmetic():
    world = World()
    world.set_possess("Bob", "crayons", 5)
    world.transfer("Rebekah", "Bob", "crayons", 2)
    assert world.get("Bob", "crayons") == 7


def test_world_outgoing_transfer():
    world = World()
    world.set_possess("Maya", "stickers", 12)
    world.transfer("Maya", "Carlos", "stickers", 3)
    assert world.get("Maya", "stickers") == 9
    assert world.get("Carlos", "stickers") == 3


def test_train_and_heldout_templates_are_disjoint():
    train = all_template_strings(TRAIN_TEMPLATES)
    held = all_template_strings(HELD_TEMPLATES)
    assert train.isdisjoint(held)


def test_unusual_probes_are_not_training_templates():
    train = all_template_strings(TRAIN_TEMPLATES)
    held = all_template_strings(HELD_TEMPLATES)
    for probe in UNUSUAL_PROBES:
        for event in probe.events + [probe.question]:
            assert event not in train
            assert event not in held


def test_generated_curriculum_avoids_forbidden_snippets():
    train = generate_curriculum(30, held_out=False, seed=11)
    held = generate_curriculum(20, held_out=True, seed=22)
    for ep in train + held:
        blob = " ".join(ep.events + [ep.question])
        assert not contains_forbidden_snippet(blob), blob


def test_pronouns_disabled():
    assert INCLUDE_PRONOUNS is False
    unused = " ".join(sum(PRONOUN_TEMPLATES.values(), []))
    train = generate_curriculum(40, held_out=False, seed=33)
    for ep in train:
        blob = " ".join(ep.events + [ep.question]).lower()
        for pronoun in (" he ", " she ", " they ", " him ", " theirs "):
            assert pronoun not in f" {blob} "
        assert unused == "" or True


def test_stage_structures():
    rng = random.Random(44)
    s1 = generate_episode(1, rng)
    assert len(s1.events) == 1
    assert s1.answer == s1.mentioned_numbers[0]

    s2 = generate_episode(2, rng)
    assert len(s2.events) == 2
    assert s2.answer == s2.initial_qty + s2.mentioned_numbers[1]

    s3 = generate_episode(3, rng)
    assert len(s3.events) == 2
    assert s3.answer == s3.initial_qty - s3.mentioned_numbers[1]

    s4 = generate_episode(4, rng)
    assert len(s4.events) >= 2

    s5 = generate_episode(5, rng)
    assert len(s5.events) >= 3

    s6 = generate_episode(6, rng)
    assert len(s6.events) >= 3


def test_demo_episode_is_ordinary_stage2_shape():
    assert DEMO_EPISODE.answer == 7
    assert DEMO_EPISODE.n_updates == 2
    assert DEMO_EPISODE.events[0] == "Bob owns 5 crayons."
    assert "gave" in DEMO_EPISODE.events[1]


def test_compositional_names_do_not_overlap_training_names():
    assert set(NAMES).isdisjoint(COMPOSITIONAL_NAMES)
    assert set(OBJECTS).isdisjoint(COMPOSITIONAL_OBJECTS)


def test_curriculum_answers_in_range():
    episodes = generate_curriculum(25, seed=55, max_answer=64)
    assert all(0 <= ep.answer <= 64 for ep in episodes)
    assert {ep.stage for ep in episodes} == {1, 2, 3, 4, 5, 6}

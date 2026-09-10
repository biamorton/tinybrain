from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import random

# Pronoun generation is an explicit extension point for a later experiment.
# v0.3 does not emit he/she/they/him/her/theirs in event or question text.
INCLUDE_PRONOUNS = False
PRONOUN_TEMPLATES: dict[str, list[str]] = {
    # Phase 2 examples, unused:
    # "He gave {receiver} {n} {obj}."
    # "Then she received {n} {obj}."
}


NUM_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}

NAMES = [
    "Bob",
    "Alice",
    "Carlos",
    "Jenny",
    "Maya",
    "Liam",
    "Nora",
    "Sam",
    "Priya",
    "Diego",
    "Rebekah",
    "Zoe",
    "Owen",
    "Emma",
]

COMPOSITIONAL_NAMES = [
    "Hiroshi",
    "Aisha",
    "Leilani",
    "Kwame",
    "Ingrid",
    "Yusuf",
    "Soren",
    "Nia",
]

OBJECTS = {
    "crayons": "crayon",
    "books": "book",
    "apples": "apple",
    "marbles": "marble",
    "stickers": "sticker",
    "coins": "coin",
    "pencils": "pencil",
    "cards": "card",
    "cookies": "cookie",
    "shells": "shell",
    "notebooks": "notebook",
    "buttons": "button",
}

COMPOSITIONAL_OBJECTS = {
    "ribbons": "ribbon",
    "feathers": "feather",
    "postcards": "postcard",
    "magnets": "magnet",
    "beads": "bead",
    "tickets": "ticket",
}

# Linguistic forms used only in training. Held-out templates must not overlap.
TRAIN_TEMPLATES = {
    "possess": [
        "{name} has {n} {obj}.",
        "{name} owns {n} {obj}.",
        "{name} keeps {n} {obj}.",
        "{name} currently has {n} {obj}.",
        "{name} possesses {n} {obj}.",
        "There are {n} {obj} belonging to {name}.",
        "{name} is holding {n} {obj}.",
        "{name} bought {n} {obj}.",
        "{name} has a total of {n} {obj}.",
        "{name} collected {n} {obj}.",
    ],
    "transfer": [
        "{giver} gave {receiver} {n} {obj}.",
        "{giver} handed {receiver} {n} {obj}.",
        "{receiver} received {n} {obj} from {giver}.",
        "{giver} gave {n} {obj} to {receiver}.",
        "{receiver} got {n} {obj} from {giver}.",
        "{giver} handed {n} {obj} to {receiver}.",
        "{giver} sent {receiver} {n} {obj}.",
        "{receiver} obtained {n} {obj} from {giver}.",
    ],
    "question": [
        "How many {obj} does {name} have?",
        "How many {obj} belong to {name}?",
        "Tell me how many {obj} {name} owns.",
        "What is the number of {obj} {name} has?",
        "How many {obj} does {name} own?",
        "How many {obj} does {name} currently have?",
        "Count the {obj} that {name} has.",
    ],
}

# Linguistic forms that MUST never appear in training.
HELD_TEMPLATES = {
    "possess": [
        "{name}'s got {n} {obj}.",
        "The {n} {obj} are {name}'s.",
        "{name} ended up with {n} {obj}.",
        "At the moment, {n} {obj} belong to {name}.",
        "{name} is the owner of {n} {obj}.",
        "{name} now holds {n} {obj}.",
        "{name} has {n} {obj} in all.",
    ],
    "transfer": [
        "{giver} delivered {n} {obj} to {receiver}.",
        "{giver} supplied {receiver} with {n} {obj}.",
        "{receiver} was given {n} {obj} by {giver}.",
        "{giver} transferred {n} {obj} to {receiver}.",
        "{giver} provided {receiver} with {n} {obj}.",
        "{receiver} acquired {n} {obj} from {giver}.",
    ],
    "question": [
        "What's {name}'s {obj} count?",
        "How large is {name}'s collection of {obj}?",
        "Can you tell me the amount of {obj} belonging to {name}?",
        "What quantity of {obj} does {name} have?",
        "How many {obj} are in {name}'s collection?",
        "What amount of {obj} does {name} own?",
    ],
}

# Phrases reserved for unusual OOD probes. Never generate these in train or held-out.
FORBIDDEN_SNIPPETS = (
    "started out",
    "a pair of",
    "over to",
    "crayon total",
    "half a dozen",
    "in possession of",
    "let ",
    "remain with",
    "sitting on",
    "passed",
    "began with",
    "initially,",
)


@dataclass
class StateEpisode:
    events: list[str]
    question: str
    answer: int
    stage: int
    held_out: bool
    n_updates: int
    target_name: str
    target_object: str
    mentioned_numbers: list[int] = field(default_factory=list)
    initial_qty: int | None = None
    compositional: bool = False
    transfer_qty: int | None = None
    transfer_direction: str | None = None
    other_qty: int | None = None
    probe_family: str | None = None


class World:
    """Symbolic label generator. This is NOT used at inference time."""

    def __init__(self):
        self.inv: dict[tuple[str, str], int] = defaultdict(int)

    def set_possess(self, person: str, obj: str, n: int) -> None:
        self.inv[(person, obj)] = n

    def transfer(self, giver: str, receiver: str, obj: str, n: int) -> None:
        self.inv[(giver, obj)] = max(0, self.inv[(giver, obj)] - n)
        self.inv[(receiver, obj)] += n

    def get(self, person: str, obj: str) -> int:
        return self.inv[(person, obj)]


def noun(obj_plural: str, quantity: int, objects: dict[str, str]) -> str:
    if quantity == 1:
        return objects[obj_plural]
    return obj_plural


def render_number(n: int, rng: random.Random) -> str:
    if n in NUM_WORDS and rng.random() < 0.45:
        return NUM_WORDS[n]
    return str(n)


def all_template_strings(source: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    for group in source.values():
        out.update(group)
    return out


def _pick_two_names(rng: random.Random, names: list[str]) -> tuple[str, str]:
    a, b = rng.sample(names, 2)
    return a, b


def _question_text(templates: dict[str, list[str]], name: str, obj: str, rng: random.Random) -> str:
    return rng.choice(templates["question"]).format(name=name, obj=obj)


def _possess_text(
    templates: dict[str, list[str]],
    name: str,
    obj_plural: str,
    n: int,
    objects: dict[str, str],
    rng: random.Random,
) -> str:
    n_str = render_number(n, rng)
    return rng.choice(templates["possess"]).format(
        name=name, n=n_str, obj=noun(obj_plural, n, objects)
    )


def _transfer_text(
    templates: dict[str, list[str]],
    giver: str,
    receiver: str,
    obj_plural: str,
    n: int,
    objects: dict[str, str],
    rng: random.Random,
) -> str:
    n_str = render_number(n, rng)
    return rng.choice(templates["transfer"]).format(
        giver=giver,
        receiver=receiver,
        n=n_str,
        obj=noun(obj_plural, n, objects),
    )


def generate_episode(
    stage: int,
    rng: random.Random,
    held_out: bool = False,
    compositional: bool = False,
    max_answer: int = 64,
) -> StateEpisode:
    if INCLUDE_PRONOUNS:
        raise RuntimeError("pronoun generation is disabled for v0.3")
    if stage not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"unknown stage {stage}")

    templates = HELD_TEMPLATES if held_out else TRAIN_TEMPLATES
    names = list(COMPOSITIONAL_NAMES if compositional else NAMES)
    objects = dict(COMPOSITIONAL_OBJECTS if compositional else OBJECTS)
    obj_list = list(objects)

    world = World()
    events: list[str] = []
    mentioned: list[int] = []

    def possess(person: str, obj: str, n: int) -> None:
        world.set_possess(person, obj, n)
        events.append(_possess_text(templates, person, obj, n, objects, rng))
        mentioned.append(n)

    def transfer(giver: str, receiver: str, obj: str, n: int) -> None:
        world.transfer(giver, receiver, obj, n)
        events.append(_transfer_text(templates, giver, receiver, obj, n, objects, rng))
        mentioned.append(n)

    def distractors(target_name: str, target_obj: str, count: int) -> None:
        available_people = [n for n in names if n != target_name]
        available_objs = [o for o in obj_list if o != target_obj]
        for _ in range(count):
            d_name = rng.choice(available_people)
            d_obj = rng.choice(available_objs)
            d_n = rng.randint(1, 12)
            possess(d_name, d_obj, d_n)

    def insert_distractors(target_name: str, target_obj: str, count: int) -> None:
        extra: list[str] = []
        extra_nums: list[int] = []
        available_people = [n for n in names if n != target_name]
        available_objs = [o for o in obj_list if o != target_obj]
        for _ in range(count):
            d_name = rng.choice(available_people)
            d_obj = rng.choice(available_objs)
            d_n = rng.randint(1, 12)
            extra.append(_possess_text(templates, d_name, d_obj, d_n, objects, rng))
            extra_nums.append(d_n)
        for text, n in zip(extra, extra_nums):
            idx = rng.randint(0, len(events))
            events.insert(idx, text)
            mentioned.insert(idx, n)

    target_name = rng.choice(names)
    target_obj = rng.choice(obj_list)
    initial_qty: int | None = None
    transfer_qty: int | None = None
    transfer_direction: str | None = None
    other_qty: int | None = None

    if stage == 1:
        n = rng.randint(1, 15)
        possess(target_name, target_obj, n)
        initial_qty = n

    elif stage == 2:
        giver = rng.choice([n for n in names if n != target_name])
        n = rng.randint(2, 12)
        k = rng.randint(1, 6)
        possess(target_name, target_obj, n)
        initial_qty = n
        transfer(giver, target_name, target_obj, k)
        transfer_qty = k
        transfer_direction = "in"
        other_qty = world.get(giver, target_obj)

    elif stage == 3:
        receiver = rng.choice([n for n in names if n != target_name])
        n = rng.randint(5, 16)
        k = rng.randint(1, min(5, n - 1))
        possess(target_name, target_obj, n)
        initial_qty = n
        transfer(target_name, receiver, target_obj, k)
        transfer_qty = k
        transfer_direction = "out"
        other_qty = world.get(receiver, target_obj)

    elif stage == 4:
        n = rng.randint(2, 12)
        possess(target_name, target_obj, n)
        initial_qty = n
        other_count = rng.randint(1, 3)
        distractors(target_name, target_obj, other_count)
        order = list(range(len(events)))
        rng.shuffle(order)
        events[:] = [events[i] for i in order]
        mentioned[:] = [mentioned[i] for i in order]

    elif stage == 5:
        n = rng.randint(6, 16)
        possess(target_name, target_obj, n)
        initial_qty = n
        others = [n for n in names if n != target_name]
        n_transfers = rng.randint(2, 4)
        for _ in range(n_transfers):
            other = rng.choice(others)
            current = world.get(target_name, target_obj)
            incoming = current <= 1 or rng.random() < 0.55
            if incoming:
                k = rng.randint(1, 5)
                transfer(other, target_name, target_obj, k)
            else:
                k = rng.randint(1, min(4, current))
                transfer(target_name, other, target_obj, k)

    else:  # stage 6
        base = rng.choice([2, 3, 5])
        inner = generate_episode(
            base, rng, held_out=held_out, compositional=compositional, max_answer=max_answer
        )
        events = list(inner.events)
        mentioned = list(inner.mentioned_numbers)
        target_name = inner.target_name
        target_obj = inner.target_object
        initial_qty = inner.initial_qty
        world = World()
        # Rebuild world by not needed; answer comes from inner after distractors
        # which do not change the target quantity.
        insert_distractors(target_name, target_obj, rng.randint(1, 3))
        answer = inner.answer
        if not (0 <= answer <= max_answer):
            raise ValueError("generated answer out of range")
        return StateEpisode(
            events=events,
            question=_question_text(templates, target_name, target_obj, rng)
            if held_out or rng.random() < 0.7
            else inner.question,
            answer=answer,
            stage=6,
            held_out=held_out,
            n_updates=len(events),
            target_name=target_name,
            target_object=target_obj,
            mentioned_numbers=mentioned,
            initial_qty=initial_qty,
            compositional=compositional,
            transfer_qty=inner.transfer_qty,
            transfer_direction=inner.transfer_direction,
            other_qty=inner.other_qty,
        )

    answer = world.get(target_name, target_obj)
    if not (0 <= answer <= max_answer):
        raise ValueError("generated answer out of range")

    return StateEpisode(
        events=events,
        question=_question_text(templates, target_name, target_obj, rng),
        answer=answer,
        stage=stage,
        held_out=held_out,
        n_updates=len(events),
        target_name=target_name,
        target_object=target_obj,
        mentioned_numbers=mentioned,
        initial_qty=initial_qty,
        compositional=compositional,
        transfer_qty=transfer_qty,
        transfer_direction=transfer_direction,
        other_qty=other_qty,
    )


def generate_curriculum(
    per_stage: int,
    held_out: bool = False,
    seed: int = 1337,
    stages: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    compositional: bool = False,
    max_answer: int = 64,
) -> list[StateEpisode]:
    rng = random.Random(seed)
    episodes: list[StateEpisode] = []
    for stage in stages:
        made = 0
        attempts = 0
        while made < per_stage and attempts < per_stage * 20:
            attempts += 1
            try:
                ep = generate_episode(
                    stage,
                    rng,
                    held_out=held_out,
                    compositional=compositional,
                    max_answer=max_answer,
                )
            except ValueError:
                continue
            if 0 <= ep.answer <= max_answer and ep.events:
                if is_frozen_episode(ep):
                    continue
                episodes.append(ep)
                made += 1
        if made < per_stage:
            raise RuntimeError(f"could not generate {per_stage} stage-{stage} episodes")
    rng.shuffle(episodes)
    return episodes


def contains_forbidden_snippet(text: str) -> bool:
    low = text.lower()
    return any(snippet in low for snippet in FORBIDDEN_SNIPPETS)


DEMO_EPISODE = StateEpisode(
    events=[
        "Bob owns 5 crayons.",
        "Rebekah gave Bob 2 crayons.",
    ],
    question="How many crayons does Bob have?",
    answer=7,
    stage=2,
    held_out=False,
    n_updates=2,
    target_name="Bob",
    target_object="crayons",
    mentioned_numbers=[5, 2],
    initial_qty=5,
    transfer_qty=2,
    transfer_direction="in",
    other_qty=0,
    probe_family="demo",
)

UNUSUAL_PROBES = [
    StateEpisode(
        events=[
            "Bob started out with five crayons.",
            "Rebekah passed a pair of crayons over to Bob.",
        ],
        question="What's Bob's crayon total now?",
        answer=7,
        stage=99,
        held_out=True,
        n_updates=2,
        target_name="Bob",
        target_object="crayons",
        mentioned_numbers=[5, 2],
        initial_qty=5,
        transfer_qty=2,
        transfer_direction="in",
        other_qty=0,
        probe_family="unusual",
    ),
    StateEpisode(
        events=[
            "Initially, Maya was in possession of twelve stickers.",
            "Then Maya let Carlos take three stickers.",
        ],
        question="So how many stickers remain with Maya?",
        answer=9,
        stage=99,
        held_out=True,
        n_updates=2,
        target_name="Maya",
        target_object="stickers",
        mentioned_numbers=[12, 3],
        initial_qty=12,
        transfer_qty=3,
        transfer_direction="out",
        other_qty=3,
        probe_family="unusual",
    ),
    StateEpisode(
        events=[
            "Priya began with 4 books.",
            "Sam handed Priya 6 books.",
            "Diego owns 8 marbles.",
        ],
        question="What number of books is Priya sitting on?",
        answer=10,
        stage=99,
        held_out=True,
        n_updates=3,
        target_name="Priya",
        target_object="books",
        mentioned_numbers=[4, 6, 8],
        initial_qty=4,
        transfer_qty=6,
        transfer_direction="in",
        other_qty=8,
        probe_family="unusual",
    ),
]


def _role(
    events: list[str],
    question: str,
    answer: int,
    *,
    target_name: str,
    target_object: str,
    mentioned: list[int],
    initial: int,
    transfer_qty: int,
    direction: str,
    other_qty: int,
    family: str,
) -> StateEpisode:
    return StateEpisode(
        events=events,
        question=question,
        answer=answer,
        stage=98,
        held_out=True,
        n_updates=len(events),
        target_name=target_name,
        target_object=target_object,
        mentioned_numbers=mentioned,
        initial_qty=initial,
        transfer_qty=transfer_qty,
        transfer_direction=direction,
        other_qty=other_qty,
        probe_family=family,
    )


# Frozen source/recipient probes. Train-like wording, never in the training pool.
ROLE_PROBES = [
    _role(
        ["Maya has 8 stickers.", "Liam gave Maya 3 stickers."],
        "How many stickers does Maya have?",
        11,
        target_name="Maya",
        target_object="stickers",
        mentioned=[8, 3],
        initial=8,
        transfer_qty=3,
        direction="in",
        other_qty=0,
        family="A",
    ),
    _role(
        ["Priya owns 6 books.", "Sam handed Priya 4 books."],
        "How many books does Priya have?",
        10,
        target_name="Priya",
        target_object="books",
        mentioned=[6, 4],
        initial=6,
        transfer_qty=4,
        direction="in",
        other_qty=0,
        family="A",
    ),
    _role(
        ["Diego currently has 10 marbles.", "Nora gave Diego 2 marbles."],
        "How many marbles does Diego have?",
        12,
        target_name="Diego",
        target_object="marbles",
        mentioned=[10, 2],
        initial=10,
        transfer_qty=2,
        direction="in",
        other_qty=0,
        family="A",
    ),
    _role(
        ["Bob has 5 crayons.", "Bob gave Rebekah 2 crayons."],
        "How many crayons does Bob have?",
        3,
        target_name="Bob",
        target_object="crayons",
        mentioned=[5, 2],
        initial=5,
        transfer_qty=2,
        direction="out",
        other_qty=2,
        family="B",
    ),
    _role(
        ["Maya owns 9 stickers.", "Maya gave Carlos 4 stickers."],
        "How many stickers does Maya have?",
        5,
        target_name="Maya",
        target_object="stickers",
        mentioned=[9, 4],
        initial=9,
        transfer_qty=4,
        direction="out",
        other_qty=4,
        family="B",
    ),
    _role(
        ["Alice keeps 7 apples.", "Alice handed Jenny 3 apples."],
        "How many apples does Alice have?",
        4,
        target_name="Alice",
        target_object="apples",
        mentioned=[7, 3],
        initial=7,
        transfer_qty=3,
        direction="out",
        other_qty=3,
        family="B",
    ),
    _role(
        ["Bob has 5 crayons.", "Bob gave Rebekah 2 crayons."],
        "How many crayons does Rebekah have?",
        2,
        target_name="Rebekah",
        target_object="crayons",
        mentioned=[5, 2],
        initial=0,
        transfer_qty=2,
        direction="in",
        other_qty=3,
        family="C",
    ),
    _role(
        ["Maya owns 9 stickers.", "Maya gave Carlos 4 stickers."],
        "How many stickers does Carlos have?",
        4,
        target_name="Carlos",
        target_object="stickers",
        mentioned=[9, 4],
        initial=0,
        transfer_qty=4,
        direction="in",
        other_qty=5,
        family="C",
    ),
    _role(
        ["Alice keeps 7 apples.", "Alice handed Jenny 3 apples."],
        "How many apples does Jenny have?",
        3,
        target_name="Jenny",
        target_object="apples",
        mentioned=[7, 3],
        initial=0,
        transfer_qty=3,
        direction="in",
        other_qty=4,
        family="C",
    ),
    _role(
        ["Bob has 5 crayons.", "Rebekah has 4 crayons.", "Rebekah gave Bob 2 crayons."],
        "How many crayons does Rebekah have?",
        2,
        target_name="Rebekah",
        target_object="crayons",
        mentioned=[5, 4, 2],
        initial=4,
        transfer_qty=2,
        direction="out",
        other_qty=7,
        family="D",
    ),
    _role(
        ["Sam owns 6 coins.", "Priya has 8 coins.", "Priya gave Sam 3 coins."],
        "How many coins does Priya have?",
        5,
        target_name="Priya",
        target_object="coins",
        mentioned=[6, 8, 3],
        initial=8,
        transfer_qty=3,
        direction="out",
        other_qty=9,
        family="D",
    ),
    _role(
        ["Owen has 10 pencils.", "Zoe has 5 pencils.", "Zoe handed Owen 1 pencil."],
        "How many pencils does Zoe have?",
        4,
        target_name="Zoe",
        target_object="pencils",
        mentioned=[10, 5, 1],
        initial=5,
        transfer_qty=1,
        direction="out",
        other_qty=11,
        family="D",
    ),
]


def episode_key(episode: StateEpisode) -> tuple[tuple[str, ...], str]:
    return (tuple(episode.events), episode.question)


def frozen_episodes() -> list[StateEpisode]:
    return [DEMO_EPISODE, *UNUSUAL_PROBES, *ROLE_PROBES]


FROZEN_KEYS = {episode_key(ep) for ep in frozen_episodes()}


def frozen_keys() -> set[tuple[tuple[str, ...], str]]:
    return FROZEN_KEYS


def is_frozen_episode(episode: StateEpisode) -> bool:
    return episode_key(episode) in FROZEN_KEYS

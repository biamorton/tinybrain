from __future__ import annotations

from collections import defaultdict

from tinybrain.training.state_data import StateEpisode


TRANSFER_ERROR_LABELS = (
    "exact",
    "query_target_swap",
    "ignored_transfer",
    "reversed_transfer",
    "other_person_quantity",
    "wrong_arithmetic",
    "unrelated",
)


def reversed_quantity(episode: StateEpisode) -> int | None:
    """Quantity that would result from flipping giver/recipient."""
    if episode.initial_qty is None or episode.transfer_qty is None:
        return None
    if episode.transfer_direction == "in":
        return max(0, episode.initial_qty - episode.transfer_qty)
    if episode.transfer_direction == "out":
        return episode.initial_qty + episode.transfer_qty
    return None


def classify_prediction(episode: StateEpisode, predicted: int) -> str:
    """
    Post-hoc error label using generator ground truth.

    Must not be called from the model or used as a training target.
    """
    if predicted == episode.answer:
        return "exact"
    if episode.other_qty is not None and predicted == episode.other_qty:
        return "query_target_swap"
    if episode.initial_qty is not None and predicted == episode.initial_qty:
        return "ignored_transfer"
    reversed_qty = reversed_quantity(episode)
    if reversed_qty is not None and predicted == reversed_qty:
        return "reversed_transfer"
    if episode.other_qty is not None and predicted == episode.other_qty:
        return "other_person_quantity"
    mentioned = [n for n in episode.mentioned_numbers if n != episode.answer]
    if predicted in mentioned:
        return "other_person_quantity"
    if episode.initial_qty is not None and episode.transfer_qty is not None:
        if episode.transfer_direction == "in" and predicted > episode.initial_qty:
            return "wrong_arithmetic"
        if episode.transfer_direction == "out" and 0 <= predicted < episode.initial_qty:
            return "wrong_arithmetic"
    return "unrelated"


def summarize_predictions(episodes: list[StateEpisode], preds: list[int]) -> dict:
    buckets: dict[str, int] = defaultdict(int)
    for ep, pred in zip(episodes, preds):
        buckets[classify_prediction(ep, pred)] += 1
    n = max(len(episodes), 1)
    return {
        "n": len(episodes),
        "counts": {label: buckets.get(label, 0) for label in TRANSFER_ERROR_LABELS},
        "rates": {
            label: buckets.get(label, 0) / n for label in TRANSFER_ERROR_LABELS
        },
    }

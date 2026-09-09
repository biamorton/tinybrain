from tinybrain.memory.store import ExternalMemory


def test_memory_retrieval():
    memory = ExternalMemory()
    memory.add("Paris is the capital of France.")
    hits = memory.search("What is the capital of France?")
    assert hits
    assert "Paris" in hits[0][0]

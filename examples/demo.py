from tinybrain.brain import TinyBrain

brain = TinyBrain(memory_path="./demo_memory.json")

brain.remember("Paris is the capital of France.")

for prompt in [
    "37 * 14",
    "What is the capital of France?",
    "Design a compact reasoning architecture.",
]:
    response = brain.ask(prompt)
    print(prompt)
    print(" ->", response.answer)
    print(" ->", response.metrics)
    print()

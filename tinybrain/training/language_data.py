from __future__ import annotations
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class Example:
    text: str
    intent: str

NAMES = ["Bob","Alice","Carlos","Jenny","Maya","Liam","Nora","Sam","Priya","Diego","Zoe","Owen","Emma","Noah","Luca","Ava"]
OBJECTS = ["crayons","books","apples","marbles","stickers","coins","pencils","cards","cookies","shells","toy cars","notebooks"]
TOPICS = ["rainbows","volcanoes","databases","music","space","history","gardening","computers","oceans","batteries"]
LANGUAGES = ["Python","Kotlin","PHP","JavaScript","Go","Java","Rust"]

TRAIN = {
"greeting":["Hi","Hello","Hey","Hey there","Good morning","Good afternoon","Good evening","Howdy","Hi there","Hello friend"],
"math":["What is {a} plus {b}?","Calculate {a} * {b}","What is {a} divided by {b}?","Find the square root of {sq}","What is the square root of {sq}?","Add {a} and {b}","Multiply {a} by {b}","What is {a} minus {b}?","Compute {a} + {b}","How much is {a} times {b}?"],
"fact_statement":["{name} has {n} {obj}.","{name} owns {n} {obj}.","{name} keeps {n} {obj}.","{name} bought {n} {obj}.","{name} currently has {n} {obj}.","There are {n} {obj} belonging to {name}.","{name} is holding {n} {obj}.","{name} possesses {n} {obj}."],
"fact_question":["How many {obj} does {name} have?","How many {obj} belong to {name}?","What does {name} have?","What belongs to {name}?","Tell me how many {obj} {name} owns.","Do you know how many {obj} {name} has?","What is the number of {obj} owned by {name}?"],
"code":["Write a {lang} function that sorts a list.","Why is my {lang} loop crashing?","Help me debug this {lang} code.","Create a {lang} class for a user account.","How do I parse JSON in {lang}?","Refactor this {lang} function.","I have a bug in my {lang} program."],
"general_question":["Why is the sky blue?","How do volcanoes form?","Explain {topic} to me.","What causes tides?","How does a battery work?","What is {topic}?","Can you explain how {topic} works?"],
"chitchat":["Tell me a joke.","I had a long day.","What do you think about rainy days?","Let's chat.","Say something funny.","I'm bored.","Tell me something interesting."]}

HELD = {
"greeting":["Greetings","Heya","Morning!","Nice to see you"],
"math":["Can you work out the square root of {sq}?","Please work out {a} multiplied by {b}.","What's the sum of {a} and {b}?","If I subtract {b} from {a}, what do I get?"],
"fact_statement":["{name}'s got {n} {obj}.","The {n} {obj} are {name}'s.","{name} ended up with {n} {obj}.","At the moment, {n} {obj} belong to {name}."],
"fact_question":["What's {name}'s {obj} count?","How large is {name}'s collection of {obj}?","Can you tell me the amount of {obj} belonging to {name}?","Regarding {name}, how many {obj} are theirs?"],
"code":["My {lang} program throws an error; can you figure it out?","Could you implement this in {lang}?","Make this {lang} code cleaner.","Something is wrong with my {lang} application."],
"general_question":["Help me understand {topic}.","Teach me about {topic}.","I'm curious how {topic} works.","Could you walk me through {topic}?"],
"chitchat":["Keep me company for a minute.","I'm in the mood to talk.","Make me laugh.","Chat with me."]}

def _fill(t, rng):
    a=rng.randint(2,99); b=rng.randint(1,20); r=rng.randint(2,15)
    return t.format(a=a,b=b,sq=r*r,n=rng.randint(1,20),name=rng.choice(NAMES),obj=rng.choice(OBJECTS),topic=rng.choice(TOPICS),lang=rng.choice(LANGUAGES))

def generate_examples(per_intent=500, held_out=False, seed=1337):
    rng=random.Random(seed); source=HELD if held_out else TRAIN; out=[]
    for intent, templates in source.items():
        for _ in range(per_intent): out.append(Example(_fill(rng.choice(templates), rng), intent))
    rng.shuffle(out); return out

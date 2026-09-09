from tinybrain.core.semantic import INTENTS
from tinybrain.training.language_data import generate_examples

def run_language_benchmark(brain,per_intent=100):
    examples=generate_examples(per_intent,True,424242); correct=0; by={i:[0,0] for i in INTENTS}
    for ex in examples:
        p=brain.interpret(ex.text); ok=p.intent==ex.intent; correct+=int(ok); by[ex.intent][0]+=int(ok); by[ex.intent][1]+=1
    print('Held-out semantic-language benchmark'); print('='*54); print(f'overall: {correct}/{len(examples)} = {correct/len(examples)*100:.2f}%')
    for i in INTENTS:
        c,n=by[i]; print(f'{i:18} {c:4}/{n:<4} {c/n*100:6.2f}%')

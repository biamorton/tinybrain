from __future__ import annotations
import argparse, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tinybrain.core.semantic import INTENTS, SemanticLanguageModel, encode_text_batch
from tinybrain.training.language_data import generate_examples

class DS(Dataset):
    def __init__(self, ex): self.ex=ex; self.ids={n:i for i,n in enumerate(INTENTS)}
    def __len__(self): return len(self.ex)
    def __getitem__(self,i): return self.ex[i].text,self.ids[self.ex[i].intent]

def collate(batch,device):
    texts=[x[0] for x in batch]; labels=torch.tensor([x[1] for x in batch],dtype=torch.long,device=device); ids,lens=encode_text_batch(texts,device); return ids,lens,labels

@torch.inference_mode()
def evaluate(model,examples,device,batch=128):
    loader=DataLoader(DS(examples),batch_size=batch,collate_fn=lambda b:collate(b,device)); c=n=0
    for ids,lens,y in loader:
        _,z=model(ids,lens); p=z.argmax(-1); c+=int((p==y).sum()); n+=y.numel()
    return c/max(n,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--per-intent',type=int,default=500); ap.add_argument('--batch-size',type=int,default=96); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    torch.manual_seed(1337); random.seed(1337); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=generate_examples(a.per_intent,False,1337); held=generate_examples(max(80,a.per_intent//5),True,9876)
    model=SemanticLanguageModel().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss(); loader=DataLoader(DS(train),batch_size=a.batch_size,shuffle=True,collate_fn=lambda b:collate(b,device))
    print(f'device: {device} | train: {len(train)} | held-out: {len(held)}')
    for epoch in range(1,a.epochs+1):
        model.train(); total=0; count=0
        for ids,lens,y in loader:
            opt.zero_grad(set_to_none=True); _,z=model(ids,lens); loss=lossfn(z,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=float(loss)*y.numel(); count+=y.numel()
        model.eval(); acc=evaluate(model,held,device,a.batch_size); print(f'epoch {epoch:02d} loss={total/count:.4f} held-out={acc*100:.2f}%')
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state':model.state_dict(),'intents':INTENTS,'version':1},a.output); print('saved:',a.output)
if __name__=='__main__': main()

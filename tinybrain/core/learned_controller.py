from pathlib import Path
import torch
from .semantic import INTENTS, SemanticLanguageModel, SemanticPrediction, encode_text_batch

class LearnedController:
    def __init__(self, checkpoint, device=None):
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.exists(): raise FileNotFoundError(f"Semantic model not found: {self.checkpoint}. Run `tinybrain train-language` first.")
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload=torch.load(self.checkpoint,map_location=self.device)
        self.model=SemanticLanguageModel().to(self.device); self.model.load_state_dict(payload["model_state"]); self.model.eval()
    @torch.inference_mode()
    def route(self,text):
        ids,lengths=encode_text_batch([text],self.device); _,logits=self.model(ids,lengths); probs=torch.softmax(logits[0],dim=-1); idx=int(probs.argmax())
        return SemanticPrediction(INTENTS[idx],float(probs[idx]),{n:float(probs[i]) for i,n in enumerate(INTENTS)})

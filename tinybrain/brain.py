from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from .core.encoder import ByteTextEncoder
from .core.learned_controller import LearnedController
from .core.reasoner import RecurrentReasoner
from .memory.store import ExternalMemory
from .metrics import RunMetrics,current_rss_mb,model_parameter_mb
from .skills.registry import SkillRegistry

@dataclass
class BrainResponse:
    answer:str; metrics:RunMetrics; memory_hits:list[tuple[str,float]]; intent:str|None=None; confidence:float|None=None

class TinyBrain(nn.Module):
    def __init__(self,latent_dim=128,memory_path=None,semantic_model_path=None,device=None):
        super().__init__(); self.device_name=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder=ByteTextEncoder(latent_dim=latent_dim); self.reasoner=RecurrentReasoner(latent_dim=latent_dim); self.readout=nn.Linear(latent_dim,4)
        self.memory=ExternalMemory(memory_path); self.skills=SkillRegistry()
        if semantic_model_path is None: semantic_model_path=Path(__file__).resolve().parent/'models'/'semantic_router.pt'
        self.controller=LearnedController(semantic_model_path,self.device_name); self.to(self.device_name); self.eval()
    @torch.inference_mode()
    def interpret(self,text): return self.controller.route(text)
    @torch.inference_mode()
    def ask(self,text,max_steps=8,use_reasoning=True,use_memory=True,use_skills=True):
        started=time.perf_counter(); retrievals=skill_calls=0; hits=[]; route=self.controller.route(text); intent=route.intent; conf=route.confidence
        param_mb=model_parameter_mb(self)+model_parameter_mb(self.controller.model)
        if use_skills and intent=='math':
            skill=self.skills.get('math')
            if skill:
                try:
                    ans=skill.run(text); elapsed=(time.perf_counter()-started)*1000
                    return BrainResponse(ans,RunMetrics(elapsed,current_rss_mb(),param_mb,0,0,1),[],intent,conf)
                except Exception: pass
        if use_memory and intent=='fact_statement' and conf>=0.55:
            self.memory.add(text); elapsed=(time.perf_counter()-started)*1000
            return BrainResponse('I stored that as a fact.',RunMetrics(elapsed,current_rss_mb(),param_mb,0,0,0),[],intent,conf)
        if use_memory and intent=='fact_question':
            hits=self.memory.search(text,top_k=3); retrievals=len(hits)
            if hits and hits[0][1]>0.10:
                elapsed=(time.perf_counter()-started)*1000
                return BrainResponse(hits[0][0],RunMetrics(elapsed,current_rss_mb(),param_mb,0,retrievals,0),hits,intent,conf)
        state=self.encoder([text]); steps=0
        if use_reasoning: state,trace=self.reasoner(state,max_steps=max_steps); steps=trace.steps
        ans=f'[language understood as {intent}, confidence {conf:.2f}; response generation is not trained yet]'
        elapsed=(time.perf_counter()-started)*1000
        return BrainResponse(ans,RunMetrics(elapsed,current_rss_mb(),param_mb,steps,retrievals,skill_calls),hits,intent,conf)
    def remember(self,text): self.memory.add(text)

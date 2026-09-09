from __future__ import annotations
import ast, math, operator as op, re

class SafeMathSkill:
    _ops={ast.Add:op.add,ast.Sub:op.sub,ast.Mult:op.mul,ast.Div:op.truediv,ast.FloorDiv:op.floordiv,ast.Mod:op.mod,ast.Pow:op.pow,ast.USub:op.neg,ast.UAdd:op.pos}
    def can_handle(self,text):
        try:self.run(text);return True
        except Exception:return False
    def run(self,text):
        low=text.lower().strip()
        m=re.search(r'(?:square root of|sqrt\s*)\s*(\d+(?:\.\d+)?)',low)
        if m:return self._fmt(math.sqrt(float(m.group(1))))
        pats=[
            (r'(?:sum of|add)\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)',op.add),
            (r'(\d+(?:\.\d+)?)\s+plus\s+(\d+(?:\.\d+)?)',op.add),
            (r'(\d+(?:\.\d+)?)\s+minus\s+(\d+(?:\.\d+)?)',op.sub),
            (r'(\d+(?:\.\d+)?)\s+(?:times|multiplied by)\s+(\d+(?:\.\d+)?)',op.mul),
            (r'(\d+(?:\.\d+)?)\s+divided by\s+(\d+(?:\.\d+)?)',op.truediv),]
        for pat,fn in pats:
            m=re.search(pat,low)
            if m:return self._fmt(fn(float(m.group(1)),float(m.group(2))))
        expr=self._extract_expression(text); tree=ast.parse(expr,mode='eval'); return self._fmt(self._eval(tree.body))
    def _extract_expression(self,text):
        cleaned=text.replace('^','**'); cands=re.findall(r'[\d\.\s\+\-\*\/\(\)%]+',cleaned); cands=[c.strip() for c in cands if any(ch.isdigit() for ch in c)]
        if not cands: raise ValueError('No arithmetic expression found.')
        return max(cands,key=len)
    def _eval(self,node):
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)): return node.value
        if isinstance(node,ast.BinOp) and type(node.op) in self._ops:return self._ops[type(node.op)](self._eval(node.left),self._eval(node.right))
        if isinstance(node,ast.UnaryOp) and type(node.op) in self._ops:return self._ops[type(node.op)](self._eval(node.operand))
        raise ValueError('Unsupported math syntax')
    @staticmethod
    def _fmt(v): return str(int(v)) if isinstance(v,float) and v.is_integer() else str(v)

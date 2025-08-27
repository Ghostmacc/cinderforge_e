# PowerShell Script to Set Up CinderForge-E Project
# This script assumes it is run from the root directory of your new repository.
# It will create necessary files, set up a virtual environment, and install dependencies.
# Ensure Python 3.9 or later is installed on your system.

# Set the project root to the current directory
$ProjectRoot = Get-Location

# Create the bootstrap.ps1 file content
$bootstrapContent = @'
param([string]$ProjectRoot = (Get-Location).Path)

function Write-Text($Path, $Content) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
  $Content | Set-Content -Encoding UTF8 -Path $Path
}

# folders
$dirs = @("cinderforge_e","cinderforge_e\dsl","cinderforge_e\gen","cinderforge_e\nas",
  "cinderforge_e\units","cinderforge_e\materialize","cinderforge_e\trainer",
  "cinderforge_e\validate","cinderforge_e\search","tests")
$dirs | % { New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $_) | Out-Null }

# pyproject (entrypoints)
Write-Text "$ProjectRoot\pyproject.toml" @"
[project]
name = "cinderforge_e"
version = "0.0.1"
description = "CinderForge-E"
requires-python = ">=3.9"
[project.scripts]
cfe-quick = "cinderforge_e.validate.cli:main"
cfe-study = "cinderforge_e.search.cli:main"
[build-system]
requires = ["setuptools","wheel"]
build-backend = "setuptools.build_meta"
"@

# __init__
Write-Text "$ProjectRoot\cinderforge_e\__init__.py" "__all__ = []"

# dsl/spec.py
Write-Text "$ProjectRoot\cinderforge_e\dsl\spec.py" @"
from dataclasses import dataclass, field
from typing import Dict, List, Optional
@dataclass class LevelSpec: units:int; neurons_per:int
@dataclass class UnitAddr: level:int; index:int
@dataclass class Edge: src:UnitAddr; dst:UnitAddr; kind:str="lateral"
@dataclass class Futures: vertical:List[Edge]=field(default_factory=list); lateral:List[Edge]=field(default_factory=list)
@dataclass
class GraphSpec:
    id:str; seed:int=0; levels:List[LevelSpec]=field(default_factory=list)
    max_vertical_skip:int=1; max_lateral_right:int=2; lateral_gate_after:int=1
    resources:Optional[Dict]=field(default_factory=dict)
"@

# gen/topology.py
Write-Text "$ProjectRoot\cinderforge_e\gen\topology.py" @"
from random import Random
from typing import List
from cinderforge_e.dsl.spec import UnitAddr, Edge
def lateral_small_world(nodes:List[UnitAddr], beta:float, k:int, rng:Random):
    edges=[]; n=len(nodes)
    if n<=1 or k<=0: return edges
    for j in range(n):
        for step in range(1, min(k, n-(j+1))+1):
            dst=j+step
            if rng.random() < float(beta): dst=rng.randrange(j+1,n)
            edges.append(Edge(src=nodes[j], dst=nodes[dst], kind="lateral"))
    return edges
def lateral_scale_free(nodes:List[UnitAddr], m:int, rng:Random):
    edges=[]; n=len(nodes)
    if n<=1 or m<=0: return edges
    indeg=[0]*n
    for j in range(n-1):
        cands=list(range(j+1,n)); chosen=[]
        for _ in range(min(m,len(cands))):
            weights=[(indeg[idx]+1) for idx in cands]; tot=sum(weights); r=rng.random()*tot; acc=0.0; pick=None
            for idx,w in zip(cands,weights):
                acc+=w
                if acc>=r: pick=idx; break
            if pick is None: pick=cands[-1]
            chosen.append(pick); cands.remove(pick); indeg[pick]+=1
        for dst in chosen: edges.append(Edge(src=nodes[j], dst=nodes[dst], kind="lateral"))
    return edges
"@

# gen/generate.py
Write-Text "$ProjectRoot\cinderforge_e\gen\generate.py" @"
import random, networkx as nx
from cinderforge_e.dsl.spec import GraphSpec, UnitAddr, Futures, Edge
from cinderforge_e.gen.topology import lateral_small_world, lateral_scale_free
def generate_futures(spec:GraphSpec)->Futures:
    rng=random.Random(spec.seed); fut=Futures()
    levels=[[UnitAddr(level=i,index=j) for j in range(spec.levels[i].units)] for i in range(len(spec.levels))]
    for L in range(1,len(levels)):
        for u in levels[L]:
            src=[UnitAddr(level=k,index=j) for k in range(max(0,L-spec.max_vertical_skip),L) for j in range(len(levels[k]))]
            if src: fut.vertical.append(Edge(src=rng.choice(src), dst=u, kind="vertical"))
    sel={(e.src.level,e.src.index) for e in fut.vertical}
    for L in range(0,len(levels)-1):
        for u in levels[L]:
            if (u.level,u.index) not in sel:
                dst=[UnitAddr(level=k,index=j) for k in range(L+1,min(len(levels),L+1+spec.max_vertical_skip)) for j in range(len(levels[k]))]
                if dst: fut.vertical.append(Edge(src=u, dst=rng.choice(dst), kind="vertical"))
    mode=(spec.resources or {}).get("lateral_mode","affinity")
    if mode in ("ws","small_world"):
        k=int((spec.resources or {}).get("ws_k",max(1,spec.max_lateral_right))); beta=float((spec.resources or {}).get("ws_beta",0.15))
        for L in range(len(levels)): fut.lateral.extend(lateral_small_world(levels[L],beta=beta,k=k,rng=rng))
    elif mode in ("ba","scale_free"):
        m=int((spec.resources or {}).get("ba_m",1))
        for L in range(len(levels)): fut.lateral.extend(lateral_scale_free(levels[L],m=m,rng=rng))
    else:
        for L in range(len(levels)):
            for j,u in enumerate(levels[L]):
                cap=min(spec.max_lateral_right,max(0,len(levels[L])-(j+1)))
                for step in range(1,cap+1): fut.lateral.append(Edge(src=u,dst=levels[L][j+step],kind="lateral"))
    G=nx.DiGraph()
    for e in fut.vertical+fut.lateral: G.add_edge((e.src.level,e.src.index),(e.dst.level,e.dst.index))
    assert nx.is_directed_acyclic_graph(G),"generator produced a cycle"
    return fut
"@

# nas/metrics.py
Write-Text "$ProjectRoot\cinderforge_e\nas\metrics.py" @"
import math, numpy as np, networkx as nx
from typing import Dict, Optional
from cinderforge_e.dsl.spec import GraphSpec, Futures
def _graph(spec:GraphSpec,fut:Futures)->nx.Graph:
    G=nx.DiGraph()
    for L,lvl in enumerate(spec.levels):
        for j in range(lvl.units): G.add_node((L,j))
    for e in fut.vertical+fut.lateral: G.add_edge((e.src.level,e.src.index),(e.dst.level,e.dst.index))
    return G.to_undirected()
def _largest_cc_avg_path_len(Gu:nx.Graph)->Optional[float]:
    if Gu.number_of_nodes()<2 or Gu.number_of_edges()==0: return None
    H=Gu.subgraph(max(nx.connected_components(Gu),key=len)).copy()
    if H.number_of_nodes()<2: return None
    try: return nx.average_shortest_path_length(H)
    except: return None
def small_world_sigma(Gu:nx.Graph)->Optional[float]:
    N,E=Gu.number_of_nodes(),Gu.number_of_edges()
    if N<3 or E==0: return None
    C=nx.average_clustering(Gu); L=_largest_cc_avg_path_len(Gu)
    if L is None: return None
    k=(2.0*E)/N
    if k<=1.0: return None
    C_rand=k/N; L_rand=math.log(N)/math.log(k)
    if C_rand<=0 or L_rand<=0: return None
    return float((C/C_rand)/(L/L_rand))
def degree_gini(Gu:nx.Graph)->float:
    degs=np.array([d for _,d in Gu.degree()],float)
    if degs.size==0: return 0.0
    mean=degs.mean()
    if mean==0: return 0.0
    diffs=np.abs(degs[:,None]-degs[None,:])
    return float(diffs.sum()/(2.0*(degs.size**2)*mean))
def hub_fraction(Gu:nx.Graph, top_p:float=0.10)->float:
    N,E=Gu.number_of_nodes(),Gu.number_of_edges()
    if N==0 or E==0: return 0.0
    top_n=max(1,int(math.ceil(top_p*N)))
    degs=sorted([d for _,d in Gu.degree()],reverse=True)
    return float(min(1.0,(sum(degs[:top_n])/(2.0*E))))
def spectral_radius(Gu:nx.Graph)->float:
    if Gu.number_of_nodes()==0: return 0.0
    A=nx.to_numpy_array(Gu,dtype=float)
    try: return float(np.max(np.abs(np.linalg.eigvals(A))).real)
    except: return 0.0
def pack_metrics(spec:GraphSpec,fut:Futures)->Dict:
    Gu=_graph(spec,fut); sigma=small_world_sigma(Gu)
    C=float(nx.average_clustering(Gu)) if Gu.number_of_edges()>0 else 0.0
    L=_largest_cc_avg_path_len(Gu)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        from networkx.algorithms.community.quality import modularity as nx_modularity
        coms=list(greedy_modularity_communities(Gu)) if Gu.number_of_edges()>0 else [set(Gu.nodes)]
        Q=float(nx_modularity(Gu,coms)) if Gu.number_of_edges()>0 else 0.0
    except: Q=0.0
    return {"n_nodes":Gu.number_of_nodes(),"n_edges":Gu.number_of_edges(),
            "avg_clustering":C,"avg_path_len":float(L) if L is not None else None,
            "sigma_overall":sigma,"modularity":Q,"degree_gini":float(degree_gini(Gu)),
            "hub_fraction":float(hub_fraction(Gu)),"spectral_radius":float(spectral_radius(Gu))}
"@

# units/blocks.py (simple MLP only for first boot)
Write-Text "$ProjectRoot\cinderforge_e\units\blocks.py" @"
import torch.nn as nn
UNIT_REGISTRY = {}
def register(name): 
    def _fn(fn): UNIT_REGISTRY[name]=fn; return fn
    return _fn
@register("mlp_block")
def mlp_block(d_model:int):
    return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
"@

# materialize/torchnet.py
Write-Text "$ProjectRoot\cinderforge_e\materialize\torchnet.py" @"
import torch, torch.nn as nn
from cinderforge_e.units.blocks import UNIT_REGISTRY
class ModelWrapper(nn.Module):
    def __init__(self, blocks, d_model):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,1), nn.Sigmoid())
    def forward(self, x):
        if x.ndim==2: x=x.unsqueeze(1)
        h=x
        for blk in self.blocks: h = h + blk(h)
        if h.ndim==3: h=h.mean(dim=1)
        return self.head(h).squeeze(1)
def build_model_from_spec(spec, fut):
    d = spec.levels[0].neurons_per
    block = UNIT_REGISTRY["mlp_block"]
    return ModelWrapper([block(d) for _ in spec.levels], d)
"@

# trainer/train_seq.py (toy data)
Write-Text "$ProjectRoot\cinderforge_e\trainer\train_seq.py" @"
import torch, torch.nn as nn, torch.optim as optim
from cinderforge_e.materialize.torchnet import build_model_from_spec
def _make_data(dim=16, n_per=64, seed=None):
    g=torch.Generator(); 
    if seed is not None: g.manual_seed(int(seed))
    a=torch.randn(n_per,dim,generator=g)+1.0; b=torch.randn(n_per,dim,generator=g)-1.0
    x=torch.cat([a,b],dim=0); y=torch.cat([torch.ones(n_per),torch.zeros(n_per)],dim=0)
    idx=torch.randperm(x.size(0),generator=g); return x[idx], y[idx]
def train_one(spec, fut, xb, yb, epochs=6, lr=1e-3):
    model=build_model_from_spec(spec,fut); dev=torch.device("cpu")
    model.to(dev); xb=xb.to(dev); yb=yb.to(dev)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-2); loss_fn=nn.BCELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); out=model(xb); loss=loss_fn(out,yb); loss.backward(); opt.step()
    return model, None, None, float(loss.item())
"@

# validate/conformance.py + CLI
Write-Text "$ProjectRoot\cinderforge_e\validate\conformance.py" @"
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.nas.metrics import pack_metrics
from cinderforge_e.dsl.spec import GraphSpec
def determinism_by_seed(spec:GraphSpec)->bool:
    f1=generate_futures(spec); f2=generate_futures(spec)
    def edges(f): return sorted([(e.src.level,e.src.index,e.dst.level,e.dst.index,e.kind) for e in f.vertical+f.lateral])
    return edges(f1)==edges(f2)
def estimate_params(spec,fut): return sum(l.units*l.neurons_per*2 for l in spec.levels)
def run_conformance(spec):
    fut=generate_futures(spec); det=determinism_by_seed(spec); met=pack_metrics(spec,fut); p=estimate_params(spec,fut)
    return {"determinism_by_seed":bool(det),"params_estimate":p,"structure":met,"ok":bool(det)}
"@

Write-Text "$ProjectRoot\cinderforge_e\validate\cli.py" @"
import json
from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.validate.conformance import run_conformance
def main():
    spec=GraphSpec(id="quick",seed=7,
        levels=[LevelSpec(units=2,neurons_per=16),LevelSpec(units=3,neurons_per=64),LevelSpec(units=2,neurons_per=16)],
        max_vertical_skip=2,max_lateral_right=2,lateral_gate_after=1,resources={"lateral_mode":"ws","ws_k":2,"ws_beta":0.2})
    print(json.dumps(run_conformance(spec),indent=2))
"@

# search/cli.py (tiny Optuna study)
Write-Text "$ProjectRoot\cinderforge_e\search\cli.py" @"
import uuid, json, time
from pathlib import Path
import optuna, torch
from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.trainer.train_seq import _make_data, train_one
from cinderforge_e.nas.metrics import pack_metrics
def _spec_from_trial(trial, run_id):
    d0=trial.suggest_categorical("n0",[16,32]); n1=trial.suggest_categorical("n1",[32,64]); n2=trial.suggest_categorical("n2",[16,32])
    u0=trial.suggest_int("u0",1,2); u1=trial.suggest_int("u1",2,3); u2=trial.suggest_int("u2",1,2)
    vskip=trial.suggest_int("vskip",1,2); lateral=trial.suggest_categorical("lateral_mode",["affinity","ws","ba"])
    return GraphSpec(id=run_id,seed=int(trial.number+1),
        levels=[LevelSpec(units=u0,neurons_per=d0),LevelSpec(units=u1,neurons_per=n1),LevelSpec(units=u2,neurons_per=n2)],
        max_vertical_skip=vskip,max_lateral_right=2,lateral_gate_after=1,resources={"lateral_mode":lateral,"ws_k":2,"ws_beta":0.2,"ba_m":2})
def run_study(trials:int=6,epochs:int=6,out_dir:str|None=None):
    study=optuna.create_study(direction="minimize")
    def objective(trial):
        spec=_spec_from_trial(trial,f"trial-{uuid.uuid4().hex[:8]}"); fut=generate_futures(spec)
        xb,yb=_make_data(dim=spec.levels[0].neurons_per,n_per=32,seed=spec.seed)
        _,_,_,final=train_one(spec,fut,xb,yb,epochs=epochs)
        struct=pack_metrics(spec,fut); sigma=(struct.get("sigma_overall") or 1.0); mod=(struct.get("modularity") or 0.0)
        return final - 0.02*max(0.0,sigma-1.0) - 0.01*mod
    study.optimize(objective,n_trials=trials)
    out=Path(out_dir or f"reports/study_{int(time.time())}"); out.mkdir(parents=True,exist_ok=True)
    (out/"best.json").write_text(json.dumps({"value":study.best_value,"params":study.best_trial.params},indent=2))
    return str(out)
def main():
    import argparse; p=argparse.ArgumentParser(); p.add_argument("--trials",type=int,default=6); p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--out_dir",type=str,default=None); a=p.parse_args(); print(f"study complete -> {run_study(a.trials,a.epochs,a.out_dir)}")
"@

Write-Text "$ProjectRoot\tests\test_basic.py" @"
from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.nas.metrics import pack_metrics
def test_metrics_smoke():
    spec=GraphSpec(id="t",seed=1,levels=[LevelSpec(units=6,neurons_per=8)])
    fut=generate_futures(spec); m=pack_metrics(spec,fut); assert "n_nodes" in m and m["n_nodes"]>=6
"@

Write-Host "Bootstrap files written."
'@

# Write the bootstrap.ps1 file
Set-Content -Path "$ProjectRoot\bootstrap.ps1" -Value $bootstrapContent -Encoding UTF8

# Set execution policy to allow running local scripts
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# Run the bootstrap script
& "$ProjectRoot\bootstrap.ps1"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip

# Install the project in editable mode
pip install -e .

# Install torch for CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
pip install networkx optuna numpy scipy pytest

# Output completion message
Write-Host "Setup complete. You can now test the installation with:"
Write-Host "pytest -q"
Write-Host "python -m cinderforge_e.validate.cli"
Write-Host "python -m cinderforge_e.search.cli --trials 4 --epochs 4"
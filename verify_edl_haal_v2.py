"""
verify_edl_haal_v2.py  -  checks the loss / metric code that is INSIDE the notebooks.

WHY THIS REPLACES verify_edl_haal.py
  The old verifier tested its own pasted copies of the functions ("verbatim from B3 / C1").
  Fixing a notebook therefore could never change its result, so the same 8 checks kept failing.
  This version opens each .ipynb, pulls out the named functions with `ast`, runs them, and
  tests them. If a notebook is wrong, its test fails. If you fix the notebook, it passes.

HOW IT WORKS
  - Only function definitions (and the small helpers they need) are executed. Training,
    Drive mounting and data loading are never run. No GPU needed.
  - CONFIG is a small stand-in using the values the notebooks use (C=1, eps=1e-8, ...).

TESTS (each one is run on every notebook that has the function)
  T0  every code cell parses (after removing !pip / % magics)
  T1  expected Dirichlet CE vs Monte Carlo                         B3, B4, C1
  T2  Dirichlet KL vs torch.distributions                          B3, B4, C1
  T3  evidence-removal KL is KL(Dir(a_tilde)||Dir(C)), C=0.5/1/2   B3, B4, C1
      (exact match to torch.distributions + non-target evidence optimum is 0)
  T4  C1 main objective: output p_hat follows the readers' votes q   C1
      (legacy HAAL and hard-label modes are shown as INFO only, they are baselines)
  T5b C1 Dirichlet strength S stays bounded                         C1
  T6  rejection-curve AUC equals the full r=1..N definition         B1, B3, B4, C1
  T7  macro AUROC when a class is missing from the test fold        B1, B3, B4, C1
  T8  no bare np.trapz (removed in NumPy 2.4)                        all

USAGE
  python verify_edl_haal_v2.py                 # looks for B1/B3/B4/C1.ipynb next to this file or in cwd
  python verify_edl_haal_v2.py --nb-dir /content/drive/MyDrive/notebooks
  python verify_edl_haal_v2.py --fast          # fewer optimiser steps (about 1 minute)

Needs: torch, numpy, pandas, scikit-learn.
"""
import argparse
import ast
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("--nb-dir", default=None)
ap.add_argument("--fast", action="store_true")
ARGS = ap.parse_args()
STEPS = 4000 if ARGS.fast else 12000
STEPS_LONG = 20000 if not ARGS.fast else 10000

RESULTS = []   # (name, ok)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def info(msg):
    print(f"[INFO] {msg}")


# ----------------------------------------------------------------------------------------------
# Loading functions out of notebooks
# ----------------------------------------------------------------------------------------------
def find_nb(name):
    dirs = [Path(ARGS.nb_dir)] if ARGS.nb_dir else []
    dirs += [Path(__file__).resolve().parent if "__file__" in globals() else Path("."), Path(".")]
    for d in dirs:
        p = d / f"{name}.ipynb"
        if p.exists():
            return p
    return None


def code_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            out.append((i, "".join(c["source"])))
    return out


def strip_magics(src):
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))


def make_ns(config):
    ns = dict(np=np, pd=pd, torch=torch, F=F, math=math, nn=torch.nn,
              label_binarize=label_binarize, roc_auc_score=roc_auc_score,
              accuracy_score=accuracy_score, f1_score=f1_score,
              CONFIG=config, EPS=config.get("epsilon", 1e-8), NUM_CLASSES=3)
    return ns


def load_funcs(path, names, config):
    """Return a namespace holding the LAST top-level definition of each requested function/constant."""
    found = {}
    for i, src in code_cells(path):
        try:
            tree = ast.parse(strip_magics(src))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = ast.get_source_segment(strip_magics(src), node)
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
                found[node.targets[0].id] = ast.get_source_segment(strip_magics(src), node)
    ns = make_ns(config)
    for n, seg in found.items():
        exec(seg, ns)
    missing = [n for n in names if n not in found]
    return ns, missing


CFG_B3 = dict(num_classes=3, prior_weight_C=1.0, lambda_kl=1.0, kl_anneal_epochs=10, epsilon=1e-8)
CFG_B4 = dict(num_classes=3, prior_weight_C=1.0, lambda_kl=1.0, kl_anneal_epochs=10, epsilon=1e-8)
CFG_C1 = dict(num_classes=3, prior_c=1.0, evidence_reg_max=0.01, evidence_reg_warmup=10, epsilon=1e-8)

NB = {n: find_nb(n) for n in ("B1", "B3", "B4", "C1")}
print("Notebooks found:", {k: (str(v) if v else None) for k, v in NB.items()}, "\n")
for k, v in NB.items():
    if v is None:
        check(f"{k} notebook file found", False, "put it next to this script or use --nb-dir")

# ----------------------------------------------------------------------------------------------
# T0  syntax
# ----------------------------------------------------------------------------------------------
for k, p in NB.items():
    if p is None:
        continue
    bad = []
    for i, src in code_cells(p):
        try:
            ast.parse(strip_magics(src))
        except SyntaxError as e:
            bad.append(f"cell {i}: {e.msg} (line {e.lineno})")
    check(f"T0  {k}: all code cells parse", not bad, "; ".join(bad[:3]))

# ----------------------------------------------------------------------------------------------
# Load what we need from each notebook
# ----------------------------------------------------------------------------------------------
NS = {}
WANT = {
    "B1": ["safe_macro_auroc", "rejection_curve"],
    "B3": ["logits_to_dirichlet", "expected_dirichlet_ce", "dirichlet_kl", "kl_anneal_factor",
           "vanilla_edl_loss", "safe_macro_auroc", "rejection_curve", "curve_auc"],
    "B4": ["evidential", "expected_dirichlet_ce", "kl_to_prior", "kl_anneal_factor", "edl_loss",
           "auroc_macro", "rejection_curve"],
    "C1": ["edl", "expected_dirichlet_ce", "kl_q_to_p", "dirichlet_kl", "soft_target_c_consistent_kl",
           "c1_loss", "safe_macro_auroc", "rejection_curve", "rejection_auc", "OBJECTIVE_MODES", "_trapz"],
}
CFG = dict(B1={}, B3=CFG_B3, B4=CFG_B4, C1=CFG_C1)
for k, p in NB.items():
    if p is None:
        continue
    NS[k], miss = load_funcs(p, WANT[k], CFG[k])
    if miss:
        check(f"load {k}: required functions present", False, f"missing {miss}")
    else:
        check(f"load {k}: required functions present", True)


def has(k, *names):
    return k in NS and all(n in NS[k] for n in names)


# ----------------------------------------------------------------------------------------------
# T1  expected CE vs Monte Carlo
# ----------------------------------------------------------------------------------------------
def ce_fn(k):
    f = NS[k]["expected_dirichlet_ce"]
    if k == "C1":
        def g(alpha, t):
            oh = F.one_hot(t, 3).float()
            return f(alpha, alpha.sum(1, keepdim=True), oh).mean()
        return g
    return f


for k in ("B3", "B4", "C1"):
    if not has(k, "expected_dirichlet_ce"):
        continue
    g = ce_fn(k)
    worst = 0.0
    for a in ([1., 1., 1.], [5., 1., 2.], [0.5, 3., 0.7], [40., 2., 1.]):
        al = torch.tensor([a])
        code = g(al, torch.tensor([0])).item()
        mc = (-Dirichlet(al[0]).sample((300_000,))[:, 0].log()).mean().item()
        worst = max(worst, abs(code - mc) / max(1, mc))
    check(f"T1  {k}: expected CE vs Monte Carlo", worst < 0.01, f"worst rel. error={worst:.4f}")

# ----------------------------------------------------------------------------------------------
# T2  KL vs torch.distributions
# ----------------------------------------------------------------------------------------------
g2 = torch.Generator().manual_seed(0)
for k in ("B3", "C1"):
    if not has(k, "dirichlet_kl"):
        continue
    worst = 0.0
    for _ in range(6):
        al = torch.rand(1, 3, generator=g2) * 4 + 0.2
        be = torch.rand(1, 3, generator=g2) * 3 + 0.2
        code = NS[k]["dirichlet_kl"](al, be).mean().item()
        ref = kl_divergence(Dirichlet(al[0]), Dirichlet(be[0])).item()
        worst = max(worst, abs(code - ref))
    check(f"T2  {k}: dirichlet_kl vs torch.distributions", worst < 1e-4, f"worst abs error={worst:.2e}")

# ----------------------------------------------------------------------------------------------
# T3  evidence-removal KL must target Dir(C), for C = 0.5, 1, 2
# ----------------------------------------------------------------------------------------------
def ref_kl(alpha, y_idx, C):
    """Reference: mean over batch of KL(Dir(alpha_tilde) || Dir(C*1)) with target entry set to C."""
    rows = []
    for a, t in zip(alpha, y_idx):
        at = a.clone()
        at[int(t)] = C
        rows.append(kl_divergence(Dirichlet(at), Dirichlet(torch.full((3,), float(C)))))
    return torch.stack(rows).mean().item()


def kl_term(k, logits, y_idx, C):
    """Return the KL term the notebook's loss actually uses (not scaled by lambda)."""
    if k == "B3":
        f = NS[k]["vanilla_edl_loss"]
        full, _ = f(logits, y_idx, epoch=100, prior_C=C, lambda_kl=1.0, loss_mode="ece_plus_kl")
        ce_only, _ = f(logits, y_idx, epoch=100, prior_C=C, lambda_kl=1.0, loss_mode="ece_only")
        return (full - ce_only)
    if k == "B4":
        _, _, kl, _ = NS[k]["edl_loss"](logits, y_idx, C=C, lambda_kl=1.0, mode="ece_plus_kl", epoch=100)
        return kl
    if k == "C1":
        alpha = F.softplus(logits) + C
        return NS[k]["soft_target_c_consistent_kl"](alpha, F.one_hot(y_idx, 3).float(), C).mean()


def total_loss_hard(k, z_nontarget, C, big):
    """Loss as the notebook builds it, with the KL weight made very large so the KL term decides
    where the non-target evidence ends up. Target logit is fixed at 0."""
    logits = torch.cat([torch.zeros(1), z_nontarget.expand(2)]).unsqueeze(0)
    y = torch.tensor([0])
    if k == "B3":
        return NS[k]["vanilla_edl_loss"](logits, y, epoch=100, prior_C=C, lambda_kl=big, loss_mode="ece_plus_kl")[0]
    if k == "B4":
        return NS[k]["edl_loss"](logits, y, C=C, lambda_kl=big, mode="ece_plus_kl", epoch=100)[0]
    if k == "C1":
        old = NS[k]["CONFIG"].copy()
        NS[k]["CONFIG"].update(prior_c=C, evidence_reg_max=big, evidence_reg_warmup=0)
        try:
            q = F.one_hot(y, 3).float()
            V = torch.zeros(1); Hq = torch.zeros(1); m = torch.ones(1)
            return NS[k]["c1_loss"](logits, q, V, Hq, m, 100, 0.0, 0, "hard_label_edl", q)[0]
        finally:
            NS[k]["CONFIG"].clear(); NS[k]["CONFIG"].update(old)


need = {"B3": ("vanilla_edl_loss",), "B4": ("edl_loss",), "C1": ("soft_target_c_consistent_kl", "c1_loss")}
for k in ("B3", "B4", "C1"):
    if not has(k, *need[k]):
        continue
    for C in (0.5, 1.0, 2.0):
        # exact value check on random alphas
        logits = torch.randn(6, 3) * 1.5
        y_idx = torch.randint(0, 3, (6,))
        alpha = F.softplus(logits) + C
        code = float(kl_term(k, logits, y_idx, C))
        ref = ref_kl(alpha, y_idx, C)
        check(f"T3  {k}: KL term equals KL(Dir(a_tilde)||Dir(C)), C={C}", abs(code - ref) < 1e-4,
              f"code={code:.5f} ref={ref:.5f}")
    for C in (0.5, 1.0, 2.0):
        # behaviour check: non-target evidence is driven to ~0
        e = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([e], lr=0.05)
        for _ in range(3000):
            loss = total_loss_hard(k, e, C, 1000.0)
            opt.zero_grad(); loss.backward(); opt.step()
        ev = F.softplus(e).item()
        check(f"T3  {k}: non-target evidence optimum is 0, C={C}", ev < 0.05, f"optimum={ev:.3f}")

# ----------------------------------------------------------------------------------------------
# T4 / T5  C1: does p_hat follow the readers' votes?
# ----------------------------------------------------------------------------------------------
def label_rule(s):
    m = np.median(s)
    return 0 if m <= 2 else (1 if m < 4 else 2)


def votes(s):
    return np.bincount([0 if x <= 2 else 1 if x == 3 else 2 for x in s], minlength=3) / len(s)


def optimise(loss_fn, steps, lr=0.2):
    z = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        loss = loss_fn(z)
        opt.zero_grad(); loss.backward(); opt.step()
    a = (F.softplus(z) + 1.0).detach()[0]
    return (a / a.sum()).numpy(), float(a.sum())


READER_PATTERNS = ([3, 2, 3, 4], [2, 2, 3, 3], [4, 1, 3, 4], [1, 1, 5, 5], [5, 4, 5, 5])

if has("C1", "c1_loss"):
    c1 = NS["C1"]["c1_loss"]
    print("\nT4/T5  C1 converged p_hat versus reader votes q  (TV = total-variation distance, 0 = identical)")
    for s in READER_PATTERNS:
        V = torch.tensor([float(np.var(s) / 4)], dtype=torch.float32)
        q_np = votes(s)
        q = torch.tensor([q_np], dtype=torch.float32)
        Hq = torch.tensor([float(-(q_np[q_np > 0] * np.log(q_np[q_np > 0])).sum() / math.log(3))], dtype=torch.float32)
        y1 = F.one_hot(torch.tensor([label_rule(s)]), 3).float()
        mk = torch.ones(1)

        def mode_loss(mode):
            return lambda z: c1(z, q, V, Hq, mk, 100, 1.0, 10, mode, y1)[0]

        p_main, _ = optimise(mode_loss("reader_distribution"), STEPS)
        tv_main = 0.5 * np.abs(p_main - q_np).sum()
        p_leg, _ = optimise(mode_loss("legacy_v_haal"), STEPS)
        tv_leg = 0.5 * np.abs(p_leg - q_np).sum()
        print(f"   scores={s} q={np.round(q_np, 2)}  main p={np.round(p_main, 3)} TV={tv_main:.3f} | "
              f"legacy-HAAL p={np.round(p_leg, 3)} TV={tv_leg:.3f}")
        check(f"T4  C1 main objective follows reader votes (TV<0.10), scores={s}", tv_main < 0.10,
              f"TV={tv_main:.3f}")
    info("legacy_v_haal and hard_label_edl are kept as ablation baselines. The original HAAL mismatch "
         "(large TV above) is the reason the main C1 mode uses the reader distribution. Not counted as a failure.")

    # T5b strength bounded under the notebook's real objective (CE soft target + KL align + evidence KL)
    qb = torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float32)
    Vb = torch.tensor([0.1]); Hb = torch.tensor([0.9]); mb = torch.ones(1)
    Ss = []
    for steps in (2000, STEPS_LONG):
        _, S = optimise(lambda z: c1(z, qb, Vb, Hb, mb, 100, 1.0, 10, "reader_distribution", None)[0], steps)
        Ss.append(S)
    print(f"\nT5b C1 Dirichlet strength S after 2k / {STEPS_LONG // 1000}k steps = {Ss[0]:.1f} / {Ss[1]:.1f}")
    check("T5b C1 strength S stays bounded (S_long < 2*S_2k)", Ss[1] < 2 * Ss[0], f"S={Ss[0]:.1f}/{Ss[1]:.1f}")

# ----------------------------------------------------------------------------------------------
# T6  rejection-curve AUC
# ----------------------------------------------------------------------------------------------
rng = np.random.RandomState(1)
n = 40
y6 = rng.randint(0, 3, n)
logit6 = rng.randn(n, 3) + np.eye(3)[y6]
p6 = np.exp(logit6) / np.exp(logit6).sum(1, keepdims=True)
u6 = rng.rand(n)
trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

order = np.argsort(u6)
ref_acc = [accuracy_score(y6[order[:r]], p6[order[:r]].argmax(1)) for r in range(1, n + 1)]
REF_AUC = float(trapz(ref_acc, np.arange(1, n + 1) / n))

AUCS = {}
for k in ("B1", "B3", "B4", "C1"):
    if not has(k, "rejection_curve"):
        continue
    df = NS[k]["rejection_curve"](y6, p6, u6)
    d = df.sort_values("coverage")
    auc = float(trapz(d["accuracy"].to_numpy(), d["coverage"].to_numpy()))
    AUCS[k] = auc
    check(f"T6  {k}: rejection AUC equals full r=1..N definition", abs(auc - REF_AUC) < 1e-9,
          f"{k}={auc:.4f} reference={REF_AUC:.4f} (curve points={len(df)})")
for k, fn in (("B3", "curve_auc"), ("C1", "rejection_auc")):
    if has(k, "rejection_curve", fn):
        v = NS[k][fn](NS[k]["rejection_curve"](y6, p6, u6))
        check(f"T6  {k}: {fn}() agrees with reference", abs(v - REF_AUC) < 1e-9, f"{v:.4f}")

# ----------------------------------------------------------------------------------------------
# T7  macro AUROC with an absent class
# ----------------------------------------------------------------------------------------------
y7 = y6.copy(); y7[y7 == 0] = 1                       # class 0 missing from this "test fold"
yb = label_binarize(y7, classes=[0, 1, 2])
ref7 = float(np.mean([roc_auc_score(yb[:, j], p6[:, j]) for j in range(3) if len(np.unique(yb[:, j])) == 2]))
ref_full = float(roc_auc_score(y6, p6, multi_class="ovr", average="macro"))
for k, fn in (("B1", "safe_macro_auroc"), ("B3", "safe_macro_auroc"), ("B4", "auroc_macro"), ("C1", "safe_macro_auroc")):
    if not has(k, fn):
        continue
    f = NS[k][fn]
    try:
        v7 = float(f(y7, p6))
    except Exception as e:
        v7 = float("nan")
    check(f"T7  {k}: macro AUROC with a missing class", np.isfinite(v7) and abs(v7 - ref7) < 1e-9,
          f"{k}={v7:.4f} reference={ref7:.4f}")
    vf = float(f(y6, p6))
    check(f"T7  {k}: macro AUROC equals sklearn when all classes present", abs(vf - ref_full) < 1e-9,
          f"{k}={vf:.4f} sklearn={ref_full:.4f}")

# ----------------------------------------------------------------------------------------------
# T8  bare np.trapz
# ----------------------------------------------------------------------------------------------
for k, p in NB.items():
    if p is None:
        continue
    bad = []
    for i, src in code_cells(p):
        guarded = bool(re.search(r"hasattr\(np|getattr\(np", src))
        for ln, line in enumerate(src.split("\n"), 1):
            if re.search(r"\bnp\.trapz\b", line) and not line.strip().startswith("#") and not guarded:
                bad.append(f"cell {i} line {ln}")
    check(f"T8  {k}: no unguarded np.trapz", not bad, ", ".join(bad[:4]))

# ----------------------------------------------------------------------------------------------
print("\nSummary")
npass = sum(ok for _, ok in RESULTS)
nfail = len(RESULTS) - npass
print(f"  {npass} passed, {nfail} failed, {len(RESULTS)} checks.")
if nfail:
    print("  Failed:")
    for name, ok in RESULTS:
        if not ok:
            print("   -", name)
sys.exit(1 if nfail else 0)

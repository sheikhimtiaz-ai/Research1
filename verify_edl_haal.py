"""
Correctness checks for the loss and metric code in Sheik's notebooks (B1, B3, C1).

Every function under test is copied VERBATIM from the notebooks, except the ones marked
"PROPOSED FIX", which show the correction recommended in the review, so that Sheik can check
his own fix against the same test.

Tests
  T1  expected Dirichlet CE (B3, C1, and B4 once implemented) vs Monte Carlo      -> should PASS
  T2  KL(Dir(a) || Dir(1)) (B3/B4) vs torch.distributions                        -> should PASS
  T3  KL with prior C != 1: non-target evidence optimum (current code)            -> FAILS for C=0.5
  T3b same test with the PROPOSED FIX (KL to Dir(C), target entry set to C)       -> should PASS
  T4  HAAL fixed point (C1 haal_loss, verbatim): does the output move towards the readers' votes?
  T5  PROPOSED soft-label EDL target: does the output move towards the readers' votes?
  T5b PROPOSED soft-label target: is the Dirichlet strength S bounded? (expected FAIL: needs a regulariser)
  T6  rejection-curve AUC: B1/B3 implementation vs C1 implementation on identical inputs
  T7  macro AUROC when one class is absent from a test fold: B1/B3 vs C1

Expected run time: about 2-3 minutes on CPU.  Requires: torch, numpy, pandas, scikit-learn.
Usage:  python verify_edl_haal.py
"""
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)
K = 3
# np.trapz was removed in NumPy 2.4 (review item B9); the notebooks still call it.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ======================= verbatim from B3 =======================
def expected_dirichlet_ce(alpha, targets):
    y = F.one_hot(targets, num_classes=K).float()
    S = alpha.sum(1, keepdim=True)
    return (y * (torch.digamma(S) - torch.digamma(alpha))).sum(1).mean()


def dirichlet_kl_to_uniform(alpha_tilde):
    beta = torch.ones_like(alpha_tilde)
    sa = alpha_tilde.sum(1, keepdim=True)
    sb = beta.sum(1, keepdim=True)
    logB_a = torch.lgamma(alpha_tilde).sum(1, keepdim=True) - torch.lgamma(sa)
    logB_b = torch.lgamma(beta).sum(1, keepdim=True) - torch.lgamma(sb)
    dig = ((alpha_tilde - beta) * (torch.digamma(alpha_tilde) - torch.digamma(sa))).sum(1, keepdim=True)
    return (logB_b - logB_a + dig).mean()


# ======================= verbatim from C1 =======================
def haal_loss(logits, y_onehot, V, mask, prior_c, K, epoch, lambda_max, t_warmup):
    evidence = F.softplus(logits)
    alpha = evidence + prior_c
    S = alpha.sum(dim=1, keepdim=True)
    ce_per_sample = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    L_ce = ce_per_sample.mean()
    p_hat = alpha / S
    H_al = -(p_hat * torch.log(p_hat + 1e-8)).sum(dim=1)
    H_al_norm = H_al / math.log(K)
    align_per_sample = mask * (H_al_norm - V) ** 2
    L_align = align_per_sample.sum() / (mask.sum() + 1e-8)
    if float(t_warmup) <= 0:
        lambda_t = float(lambda_max)
    else:
        lambda_t = float(lambda_max) * min(1.0, float(epoch) / float(t_warmup))
    return L_ce + lambda_t * L_align, {"ce": L_ce.item(), "align": L_align.item(), "lambda_t": lambda_t}


# ======================= PROPOSED FIXES (not in the notebooks) =======================
def dirichlet_kl_to_prior(alpha_tilde, C):
    """KL(Dir(alpha_tilde) || Dir(C*1)).  Use with alpha_tilde = y*C + (1-y)*alpha."""
    beta = torch.full_like(alpha_tilde, float(C))
    sa = alpha_tilde.sum(1, keepdim=True)
    sb = beta.sum(1, keepdim=True)
    logB_a = torch.lgamma(alpha_tilde).sum(1, keepdim=True) - torch.lgamma(sa)
    logB_b = torch.lgamma(beta).sum(1, keepdim=True) - torch.lgamma(sb)
    dig = ((alpha_tilde - beta) * (torch.digamma(alpha_tilde) - torch.digamma(sa))).sum(1, keepdim=True)
    return (logB_b - logB_a + dig).mean()


def soft_label_edl_ce(logits, q, prior_c):
    """Expected Dirichlet CE with the readers' ternary vote distribution q as the target."""
    alpha = F.softplus(logits) + prior_c
    S = alpha.sum(dim=1, keepdim=True)
    return (q * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1).mean()


# ======================= helpers =======================
def label_rule(s):
    """Rule observed in the preprocessed pilot data: medians 2.5 and 3.5 -> indeterminate."""
    m = np.median(s)
    return 0 if m <= 2 else (1 if m < 4 else 2)


def votes(s):
    return np.bincount([0 if x <= 2 else 1 if x == 3 else 2 for x in s], minlength=3) / len(s)


def optimise(loss_fn, steps=20_000, lr=0.2):
    z = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        loss = loss_fn(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
    a = (F.softplus(z) + 1.0).detach()[0]
    return (a / a.sum()).numpy()


READER_PATTERNS = ([3, 2, 3, 4], [2, 2, 3, 3], [4, 1, 3, 4], [1, 1, 5, 5], [5, 4, 5, 5])

# ---------------- T1 ----------------
for a in ([1., 1., 1.], [5., 1., 2.], [0.5, 3., 0.7], [40., 2., 1.]):
    al = torch.tensor([a])
    code = expected_dirichlet_ce(al, torch.tensor([0])).item()
    mc = (-Dirichlet(al[0]).sample((400_000,))[:, 0].log()).mean().item()
    check("T1 expected CE vs Monte Carlo", abs(code - mc) < 0.01 * max(1, mc), f"alpha={a} code={code:.4f} mc={mc:.4f}")

# ---------------- T2 ----------------
for a in ([1., 1., 1.], [3., 1., 1.], [0.6, 2.5, 4.]):
    al = torch.tensor([a])
    code = dirichlet_kl_to_uniform(al).item()
    ref = kl_divergence(Dirichlet(al[0]), Dirichlet(torch.ones(3))).item()
    check("T2 KL vs torch.distributions", abs(code - ref) < 1e-4, f"alpha={a} code={code:.5f} ref={ref:.5f}")


# ---------------- T3 / T3b ----------------
def kl_optimum(C, fixed):
    e = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([e], lr=0.05)
    target = float(C) if fixed else 1.0          # notebook sets the target entry to 1
    for _ in range(3000):
        a = torch.cat([torch.tensor([target]), F.softplus(e) + C, F.softplus(e) + C]).unsqueeze(0)
        loss = dirichlet_kl_to_prior(a, C) if fixed else dirichlet_kl_to_uniform(a)
        opt.zero_grad(); loss.backward(); opt.step()
    return F.softplus(e).item()

for C in (0.5, 1.0, 2.0):
    ev = kl_optimum(C, fixed=False)
    check("T3  KL optimum of non-target evidence is 0 (current code)", ev < 0.05, f"C={C} optimum={ev:.3f}")
for C in (0.5, 1.0, 2.0):
    ev = kl_optimum(C, fixed=True)
    check("T3b KL optimum of non-target evidence is 0 (proposed fix)", ev < 0.05, f"C={C} optimum={ev:.3f}")
# Consistency: at C=1 the proposed fix must equal the notebook KL exactly.
al = torch.tensor([[1.0, 2.3, 0.7]])
check("T3b fix reduces to notebook KL when C=1",
      abs(dirichlet_kl_to_prior(al, 1.0).item() - dirichlet_kl_to_uniform(al).item()) < 1e-6)

# ---------------- T4 / T5 ----------------
print("\nT4 (HAAL as coded) vs T5 (proposed soft-label target): converged p_hat per reader pattern")
print("   'TV' = total-variation distance from the readers' ternary vote distribution q (0 = identical)")
for s in READER_PATTERNS:
    V = float(np.var(s) / 4)
    q = votes(s)
    y = F.one_hot(torch.tensor([label_rule(s)]), 3).float()
    p_haal = optimise(lambda z: haal_loss(z, y, torch.tensor([V]), torch.tensor([1.0]), 1.0, 3, 100, 1.0, 10)[0])
    p_soft = optimise(lambda z: soft_label_edl_ce(z, torch.tensor([q], dtype=torch.float32), 1.0))
    tv_h = 0.5 * np.abs(p_haal - q).sum()
    tv_s = 0.5 * np.abs(p_soft - q).sum()
    print(f"   scores={s} V={V:.3f} q={np.round(q, 2)}")
    print(f"      HAAL (lambda=1): p={np.round(p_haal, 3)} TV={tv_h:.3f} | soft-label: p={np.round(p_soft, 3)} TV={tv_s:.3f}")
    if q.max() < 1.0:   # readers disagree across ternary classes
        check("T4  HAAL output tracks readers' votes (TV < 0.10)", tv_h < 0.10, f"scores={s} TV={tv_h:.3f}")
    check("T5  soft-label output tracks readers' votes (TV < 0.10)", tv_s < 0.10, f"scores={s} TV={tv_s:.3f}")

# ---------------- T5b ----------------
# The soft target fixes the MEAN p_hat but not the Dirichlet strength S, which keeps growing
# (vacuity u -> 0) under the expected CE. Vacuity therefore still needs its own regulariser.
qb = torch.tensor([[0.25, 0.5, 0.25]])
S_trace = []
for steps in (2_000, 20_000):
    z = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.2)
    for _ in range(steps):
        loss = soft_label_edl_ce(z, qb, 1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    S_trace.append(float((F.softplus(z) + 1.0).sum().detach()))
print(f"\nT5b soft-label target: Dirichlet strength S after 2k / 20k steps = {S_trace[0]:.0f} / {S_trace[1]:.0f}")
check("T5b soft-label target alone keeps S bounded (vacuity meaningful)", S_trace[1] < 2 * S_trace[0],
      "S keeps growing -> pair the soft target with a C-consistent KL or evidence penalty")

# ---------------- T6 ----------------
rng = np.random.RandomState(1)
n = 40
y6 = rng.randint(0, 3, n)
logit6 = rng.randn(n, 3) + np.eye(3)[y6]
p6 = np.exp(logit6) / np.exp(logit6).sum(1, keepdims=True)
u6 = rng.rand(n)


def rejection_b1_b3(y, p, u):          # verbatim logic from B1 / B3
    pred = p.argmax(1); order = np.argsort(u)
    return pd.DataFrame([{"coverage": r / len(y), "accuracy": accuracy_score(y[order[:r]], pred[order[:r]])}
                         for r in range(1, len(y) + 1)])


def rejection_c1(y, p, u, n_steps=20):  # verbatim logic from C1 compute_rejection_curve
    order = np.argsort(-u); rows = []
    for frac in np.linspace(0.0, 0.9, n_steps):
        keep = order[int(round(frac * len(y))):]
        rows.append({"coverage": 1 - frac, "accuracy": float((p[keep].argmax(1) == y[keep]).mean())})
    return pd.DataFrame(rows).sort_values("coverage")


d_b = rejection_b1_b3(y6, p6, u6)
d_c = rejection_c1(y6, p6, u6)
auc_b = float(_trapz(d_b.accuracy, d_b.coverage))
auc_c = float(_trapz(d_c.accuracy, d_c.coverage))
check("T6  rejection AUC identical across B1/B3 and C1 implementations", abs(auc_b - auc_c) < 1e-6,
      f"B1/B3={auc_b:.4f} C1={auc_c:.4f}")

# ---------------- T7 ----------------
y7 = y6.copy(); y7[y7 == 0] = 1           # benign absent from this test fold
yb = label_binarize(y7, classes=[0, 1, 2])
auc_b13 = float(np.mean([roc_auc_score(yb[:, k], p6[:, k]) for k in range(3) if len(np.unique(yb[:, k])) == 2]))
try:
    auc_c1 = float(roc_auc_score(y7, p6, multi_class="ovr", average="macro", labels=[0, 1, 2]))
except ValueError:
    auc_c1 = float("nan")
check("T7  macro AUROC identical across B1/B3 and C1 when a class is absent",
      np.isfinite(auc_c1) and abs(auc_b13 - auc_c1) < 1e-6, f"B1/B3={auc_b13:.4f} C1={auc_c1}")

# ---------------- summary ----------------
print("\nSummary:")
print("  Expected PASS: T1, T2, T3 (C=1, C=2), T3b, T5.")
print("  Expected FAIL (documented defects): T3 at C=0.5 [B5], T4 [A1], T5b [A1 caveat], T6 and T7 [B4].")
print(f"  {sum(ok for _, ok in RESULTS)} passed, {sum(not ok for _, ok in RESULTS)} failed, {len(RESULTS)} checks.")

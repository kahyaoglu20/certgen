"""Tightness of the plan-pullback bound on trained decoders (Section 7.2,
Appendix B.8).

Variational autoencoders are trained on the scikit-learn digits table
(1797 x 64, standardised on a 1300-row training split) in two architectures,
latent dimension 8 with two hidden layers of width 128 and latent dimension
12 with three hidden layers of width 256, with three training seeds each
(Adam, learning rate 1e-3, 150 epochs). For each model and two evaluation
draws, the encoder means of 400 held-out rows and 400 draws from the latent
prior are decoded, and certgen.plan_pullback computes

  exact      the output-space W2 by assignment;
  PPC        the plan-pullback bound under the latent-optimal assignment;
  Lipschitz  the product of the decoder's layer spectral norms times the
             latent W2.

The largest ratio ||g(z) - g(z')|| / ||z - z'|| over sampled latent pairs, a
lower bound on the decoder's Lipschitz constant, is recorded as well.

Training is seeded and single-threaded, but neural-network training is not
bit-identical across floating-point platforms; the platform is recorded, and
--out keeps a run on a second platform beside the reference artifact.

    python experiments/plan_pullback_vae.py [--out NAME]

Needs torch (CPU). Writes results/plan_pullback_vae.json (about 3 minutes).
"""
from __future__ import annotations

import json
import time

import numpy as np

import _artifacts as art
import certgen as cg

ARCHS = {"mlp128x2_lat8": (8, (128, 128)), "mlp256x3_lat12": (12, (256, 256, 256))}
TRAIN_SEEDS = (0, 1, 2)
SAMPLE_SEEDS = (0, 1)
N_EVAL = 400
EPOCHS = 150


def load_digits_std():
    from sklearn.datasets import load_digits
    X = load_digits().data.astype(float)
    g = np.random.default_rng(20260930)
    idx = g.permutation(len(X))
    tr, te = idx[:1300], idx[1300:]
    st = cg.Standardizer().fit(X[tr])
    return st.transform(X[tr]), st.transform(X[te])


def build(torch, d_in, d_lat, hidden):
    nn = torch.nn

    def mlp(sizes):
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        return nn.Sequential(*layers[:-1])       # no activation on the output

    enc = mlp((d_in,) + hidden + (2 * d_lat,))
    dec = mlp((d_lat,) + hidden[::-1] + (d_in,))
    return enc, dec


def train_vae(torch, Xtr, d_lat, hidden, seed):
    torch.manual_seed(seed)
    enc, dec = build(torch, Xtr.shape[1], d_lat, hidden)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    X = torch.tensor(Xtr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g)
        for b in range(0, len(X), 128):
            xb = X[perm[b:b + 128]]
            h = enc(xb)
            mu, logv = h[:, :d_lat], h[:, d_lat:].clamp(-8, 8)
            z = mu + torch.exp(0.5 * logv) * torch.randn(mu.shape, generator=g)
            rec = ((dec(z) - xb) ** 2).sum(1).mean()
            kl = 0.5 * (mu ** 2 + logv.exp() - 1 - logv).sum(1).mean()
            loss = rec + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc, dec, float(loss.detach())


def spectral_product(torch, dec) -> float:
    L = 1.0
    for m in dec:
        if isinstance(m, torch.nn.Linear):
            L *= float(torch.linalg.matrix_norm(m.weight.detach(), ord=2))
    return L


def local_lipschitz_lower(dec_np, d_lat, rng, n=4000):
    Z1 = rng.normal(size=(n, d_lat))
    Z2 = Z1 + rng.normal(size=(n, d_lat)) * rng.uniform(0.01, 1.0, size=(n, 1))
    num = np.linalg.norm(dec_np(Z1) - dec_np(Z2), axis=1)
    den = np.linalg.norm(Z1 - Z2, axis=1)
    return float(np.max(num / np.maximum(den, 1e-12)))


def main(argv=None):
    import argparse
    import platform
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plan_pullback_vae.json",
                    help="artifact name under results/")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    t0 = time.time()
    Xtr, Xte = load_digits_std()
    cells = []
    for arch, (d_lat, hidden) in ARCHS.items():
        for ts in TRAIN_SEEDS:
            enc, dec, loss = train_vae(torch, Xtr, d_lat, hidden, ts)
            enc.eval()
            dec.eval()

            def dec_np(Z, dec=dec):
                with torch.no_grad():
                    return dec(torch.tensor(Z, dtype=torch.float32)).numpy().astype(float)

            with torch.no_grad():
                mu_te = enc(torch.tensor(Xte, dtype=torch.float32))[:, :d_lat].numpy()
            L = spectral_product(torch, dec)
            for ss in SAMPLE_SEEDS:
                rng = np.random.default_rng([ts, ss, 13])
                pick = rng.choice(len(mu_te), N_EVAL, replace=False)
                Zd = mu_te[pick].astype(float)
                Zg = rng.normal(size=(N_EVAL, d_lat))
                rep = cg.plan_pullback(Zd, Zg, dec_np, L_cert=L, compute_exact=True)
                llb = local_lipschitz_lower(dec_np, d_lat, rng)
                rec = {"arch": arch, "d_lat": d_lat, "hidden": list(hidden),
                       "train_seed": ts, "sample_seed": ss, "n": N_EVAL,
                       "final_train_loss": loss,
                       "w2_latent": rep.w2_latent, "ppc": rep.ppc_bound,
                       "exact": rep.exact_w2_output,
                       "lipschitz_bound": rep.lipschitz_bound,
                       "L_spectral_product": L, "L_local_lower": llb,
                       "ppc_over_exact": rep.ppc_bound / rep.exact_w2_output,
                       "lipschitz_over_exact": rep.lipschitz_bound / rep.exact_w2_output,
                       "local_lipschitz_over_exact": llb * rep.w2_latent / rep.exact_w2_output,
                       "warnings": rep.warnings}
                cells.append(rec)
                print(f"{arch:<16} ts={ts} ss={ss}  exact={rec['exact']:.3f}  "
                      f"PPC/exact={rec['ppc_over_exact']:.3f}  "
                      f"Lip/exact={rec['lipschitz_over_exact']:.1f}  "
                      f"(local L lower bound route {rec['local_lipschitz_over_exact']:.2f})",
                      flush=True)

    def agg(key):
        v = np.array([c[key] for c in cells], float)
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max())}
    out = {"_definition": ("tightness of the plan-pullback bound on trained "
                           "VAE decoders: PPC/exact and Lipschitz/exact, with "
                           "exact the output-space W2 between decoded held-out "
                           "encodings and decoded prior draws"),
           "data": "sklearn load_digits (1797 x 64), standardised on a 1300-row "
                   "training split; evaluation on the 497 held-out rows",
           "config": {"archs": {k: [v[0], list(v[1])] for k, v in ARCHS.items()},
                      "train_seeds": list(TRAIN_SEEDS),
                      "sample_seeds": list(SAMPLE_SEEDS), "n_eval": N_EVAL,
                      "epochs": EPOCHS},
           "cells": cells,
           "summary": {"ppc_over_exact": agg("ppc_over_exact"),
                       "lipschitz_over_exact": agg("lipschitz_over_exact"),
                       "local_lipschitz_over_exact": agg("local_lipschitz_over_exact"),
                       "all_ppc_valid": bool(all(c["ppc"] >= c["exact"] - 1e-9
                                                 for c in cells)),
                       "cells": len(cells)},
           "torch": torch.__version__,
           "platform": f"{platform.system()} {platform.machine()}, "
                       f"Python {platform.python_version()}",
           "elapsed_s": round(time.time() - t0, 1)}
    art.save(a.out, out)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()

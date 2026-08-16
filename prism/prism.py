"""
PRISM: Prototype-Rectified Iterative Self-supervised Manifold Denoising
CIKM 2026 | https://github.com/Ashish-1108/PRISM

Usage:
    from prism import prism, deploy
    E_denoised, W = prism(E_noisy, T, n_classes=10)  # calibrate once
    E_clean = deploy(E_new, W)                         # deploy instantly
"""

import numpy as np
import torch
import torch.nn.functional as F


def opca(E, T, n_classes, n_iters=1, top_pct=0.80):
    """Align noisy audio embeddings to text prototypes via Orthogonal Procrustes."""
    d = E.shape[1]
    E_cur = E.clone()
    for _ in range(n_iters):
        logits = E_cur @ T.t()
        confs = logits.max(dim=1).values
        labels = logits.argmax(dim=1).numpy()
        thresh = torch.quantile(confs, 1.0 - top_pct)
        M_n = torch.zeros(n_classes, d)
        valid = []
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                conf_mask = torch.tensor(mask) & (confs >= thresh)
                if conf_mask.sum() > 0:
                    w = confs[conf_mask]; w = w / w.sum()
                    M_n[c] = (E_cur[conf_mask] * w.unsqueeze(1)).sum(dim=0)
                else:
                    M_n[c] = E_cur[mask].mean(dim=0)
                valid.append(c)
        if len(valid) < 2:
            return E_cur
        Mn = M_n[valid]; Mt = T[valid]
        mu_n = Mn.mean(0, keepdim=True); mu_t = Mt.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd((Mn - mu_n).T @ (Mt - mu_t))
        R = U @ Vh
        s = min(max((Mt - mu_t).norm() / ((Mn - mu_n).norm() + 1e-8), 0.8), 1.2)
        E_cur = F.normalize(s * ((E_cur - mu_n) @ R) + mu_t, p=2, dim=1)
    return E_cur


def ccvd(E, T, n_classes, K=60, conf_pct=0.70):
    """Remove top-K noise eigenvectors via Fisher discriminant analysis."""
    d = E.shape[1]
    logits = E @ T.t()
    confs, labels = logits.max(dim=1)
    labels = labels.numpy(); confs = confs.numpy()
    thresh = np.percentile(confs, (1 - conf_pct) * 100)
    mask = confs >= thresh
    E_f = E[mask]; lf = labels[mask]
    if len(np.unique(lf)) < 2:
        return E
    mu = E_f.mean(dim=0)
    S_W = torch.zeros(d, d); S_B = torch.zeros(d, d)
    for c in range(n_classes):
        cm = lf == c
        if cm.sum() < 2: continue
        Ec = E_f[cm]; mc = Ec.mean(dim=0)
        diff = Ec - mc.unsqueeze(0); S_W += diff.T @ diff
        delta = (mc - mu).unsqueeze(1); S_B += cm.sum() * (delta @ delta.T)
    S_B_reg = S_B + 1e-4 * torch.eye(d)
    M = torch.linalg.solve(S_B_reg, S_W)
    _, eigvecs = torch.linalg.eigh(M)
    noise_dirs = eigvecs[:, -K:]
    P = torch.eye(d) - noise_dirs @ noise_dirs.T
    return F.normalize(E @ P, p=2, dim=1)


def residual_shift(E, T, n_classes, alpha=0.30):
    """Shift each sample toward its predicted text prototype centroid."""
    labels = (E @ T.t()).argmax(dim=1).numpy()
    E_shifted = E.clone()
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            mu_c = E[mask].mean(dim=0)
            E_shifted[mask] = E[mask] + alpha * (T[c] - mu_c).unsqueeze(0)
    return F.normalize(E_shifted, p=2, dim=1)


def self_cfed_v2(E_noisy, T, n_classes,
                 rounds=3, ridge_lambda=0.01,
                 K=60, alpha=0.3,
                 opca_iters=1, opca_conf=0.80):
    """
    PRISM calibration. Runs OPCA → CCVD → Shift × rounds, compiles into matrix W.

    Args:
        E_noisy:  (N, d) L2-normalized noisy CLAP audio embeddings
        T:        (C, d) L2-normalized text prototypes
        n_classes: number of classes
        rounds, ridge_lambda, K, alpha, opca_iters, opca_conf: hyperparameters

    Returns:
        E_denoised: (N, d) denoised embeddings
        W:          (d+1, d) affine projection matrix for deploy()
    """
    d = E_noisy.shape[1]
    E_cur = E_noisy.clone()
    beta_schedule = [0.0, 0.5, 1.0]

    for r in range(rounds):
        beta = beta_schedule[min(r, len(beta_schedule) - 1)]
        if beta == 0.0:
            P = T.clone()
        else:
            logits = E_cur @ P.t()
            confs, labels = logits.max(dim=1)
            thresh = torch.quantile(confs, 1.0 - opca_conf)
            A = torch.zeros_like(T)
            for c in range(n_classes):
                mask = labels == c
                if mask.sum() > 0:
                    conf_mask = mask & (confs >= thresh)
                    A[c] = E_cur[conf_mask].mean(0) if conf_mask.sum() > 0 else E_cur[mask].mean(0)
                else:
                    A[c] = T[c]
            P = F.normalize((1 - beta) * T + beta * A, p=2, dim=1)

        E_rotated = opca(E_cur, P, n_classes, n_iters=opca_iters, top_pct=opca_conf)
        E_decayed = ccvd(E_rotated, P, n_classes, K=K)
        E_target = residual_shift(E_decayed, P, n_classes, alpha=alpha) if alpha > 0 else E_decayed

        E_aug = torch.cat([E_noisy, torch.ones(E_noisy.shape[0], 1)], dim=1)
        A_mat = E_aug.T @ E_aug + ridge_lambda * torch.eye(d + 1)
        W_aug = torch.linalg.solve(A_mat, E_aug.T @ E_target)
        E_cur = F.normalize(E_aug @ W_aug, p=2, dim=1)

    return E_cur, W_aug


# Public alias
prism = self_cfed_v2


def deploy(E_noisy, W):
    """
    Apply pre-computed W to new audio samples. ~0.0009 ms/sample.

    Args:
        E_noisy: (N, d) or (d,) L2-normalized noisy CLAP embeddings
        W:       (d+1, d) matrix from prism()

    Returns:
        E_clean: (N, d) denoised embeddings
    """
    if E_noisy.dim() == 1:
        E_noisy = E_noisy.unsqueeze(0)
    E_aug = torch.cat([E_noisy, torch.ones(E_noisy.shape[0], 1)], dim=1)
    return F.normalize(E_aug @ W, p=2, dim=1)

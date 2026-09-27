"""Exact expected macro-F0.5 for a fixed decision under independent candidate truth probabilities (Poisson-binomial DP)."""
import numpy as np
KA, KR = 14, 16


def pb(P):
    S, K = P.shape; D = np.zeros((S, K + 1)); D[:, 0] = 1.0
    for j in range(K):
        p = P[:, j:j + 1]; N = D * (1 - p); N[:, 1:] += D[:, :-1] * p; D = N
    return D


def expected_f(A, k, R):
    """A: accepted-candidate truth probs (S,KA), k: #accepted per S1, R: rejected-candidate truth probs (S,KR)."""
    DA, DR = pb(A), pb(R); E = np.zeros(A.shape[0])
    for a in range(DA.shape[1]):
        for b in range(DR.shape[1]):
            if a == 0:
                f = ((k == 0) & (b == 0)).astype(float)
            else:
                P = a / np.maximum(k, a); R_ = a / (a + b)
                f = 1.25 * P * R_ / (0.25 * P + R_)
            E += DA[:, a] * DR[:, b] * f
    return E


def odds(M, w):
    return np.where(M > 0, w * M / (w * M + 1 - M + 1e-12), 0.0)

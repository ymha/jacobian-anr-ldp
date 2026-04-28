"""LDP mechanism wrappers and registries."""
import numpy as np
import torch
from scipy.stats import norm as _snorm, beta as _beta_dist, truncnorm as _truncnorm
from scipy.special import betainc as _betainc
from scipy.optimize import brentq as _brentq
from scipy.linalg import hadamard as _hadamard

PERCENTILE = 90.0
LAMBDA_N   = 1000.0
DELTA      = 1e-5


# ── Inlined base LDP primitives ───────────────────────────────────────────────

def _base(fn):
    fn.encode = fn
    fn.decode = lambda z, n=1: z
    return fn


def _project_l1(x: np.ndarray, rho: float) -> np.ndarray:
    """Project each row of x onto the L1 ball of radius ρ (Duchi et al. 2008)."""
    abs_x = np.abs(x)
    norms = abs_x.sum(axis=1)
    out   = x.copy()
    need  = norms > rho
    if not need.any():
        return out
    v    = abs_x[need]
    u    = np.sort(v, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    rng  = np.arange(1, x.shape[1] + 1)
    K    = (u > (cssv - rho) / rng).sum(axis=1) - 1
    theta = (cssv[np.arange(len(K)), K] - rho) / (K + 1)
    out[need] = np.sign(x[need]) * np.maximum(v - theta[:, None], 0)
    return out


@_base
def piecewise_ldp(x_norm: np.ndarray, epsilon: float,
                  rng: np.random.Generator = None) -> np.ndarray:
    """Piecewise mechanism (Wang et al., 2019). Input: [0,1]^d, budget ε/d per dim."""
    if rng is None:
        rng = np.random.default_rng()
    n, d  = x_norm.shape
    eps_j = epsilon / d
    x     = 2 * x_norm - 1
    s     = np.exp(eps_j / 2)
    C     = (s + 1) / (s - 1)
    l     = (C + 1) / 2 * x - (C - 1) / 2
    r     = (C + 1) / 2 * x + (C - 1) / 2
    p_c   = s / (s + 1)
    z_c   = rng.uniform(size=(n, d)) * (r - l) + l
    left_len = l + C
    u_t   = rng.uniform(0, C + 1, size=(n, d))
    z_t   = np.where(u_t < left_len, -C + u_t, r + (u_t - left_len))
    z     = np.where(rng.uniform(size=(n, d)) < p_c, z_c, z_t)
    return (z + 1) / 2


@_base
def duchi_ldp(x_norm: np.ndarray, epsilon: float,
              rng: np.random.Generator = None) -> np.ndarray:
    """Duchi et al. (2018). Full ε on one random dim, ±C·d output. Input: [0,1]^d."""
    if rng is None:
        rng = np.random.default_rng()
    n, d  = x_norm.shape
    x     = 2 * x_norm - 1
    C     = (np.exp(epsilon) + 1) / (np.exp(epsilon) - 1)
    j_sel = rng.integers(0, d, size=n)
    x_sel = x[np.arange(n), j_sel]
    p     = (x_sel * (np.exp(epsilon) - 1) + np.exp(epsilon) + 1) / (2 * (np.exp(epsilon) + 1))
    z_sel = np.where(rng.uniform(size=n) < p, C, -C)
    z     = np.zeros((n, d))
    z[np.arange(n), j_sel] = d * z_sel
    return (z + 1) / 2


@_base
def harmony_ldp(x_norm: np.ndarray, epsilon: float,
                rng: np.random.Generator = None) -> np.ndarray:
    """Harmony (Nguyen et al., 2016). Stochastic rounding + RR, ε/d per dim. Input: [0,1]^d."""
    if rng is None:
        rng = np.random.default_rng()
    n, d   = x_norm.shape
    eps_j  = epsilon / d
    b      = (rng.uniform(size=(n, d)) < x_norm).astype(float)
    p_keep = np.exp(eps_j) / (np.exp(eps_j) + 1)
    z_bit  = np.where(rng.uniform(size=(n, d)) >= p_keep, 1 - b, b)
    return (z_bit * (np.exp(eps_j) + 1) - 1) / (np.exp(eps_j) - 1)

# ── Gaussian noise calibration ────────────────────────────────────────────────

def _agm_sigma(eps: float, delta: float, sensitivity: float) -> float:
    """Optimal σ for (ε,δ)-LDP Gaussian mechanism (Balle & Wang 2018).

    Finds the smallest σ satisfying the exact DP condition:
        δ = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ)
    which is tighter than the classical σ = Δ·√(2 ln(1.25/δ)) / ε.
    """
    def _delta_of(sigma):
        v = sensitivity / (2.0 * sigma)
        return (_snorm.cdf(v - eps * sigma / sensitivity)
                - np.exp(eps) * _snorm.cdf(-v - eps * sigma / sensitivity))

    sigma_hi = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
    while _delta_of(sigma_hi) > delta:
        sigma_hi *= 2.0
    return _brentq(lambda s: _delta_of(s) - delta, 1e-10, sigma_hi,
                   xtol=1e-8, rtol=1e-8)


# ── PrivUnit2 shared helpers ──────────────────────────────────────────────────

def _sphere_from_t(u: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """z = t·u + √(1-t²)·w,  w ⊥ u uniform on S^{d-2}. u:(n,d), t:(n,)."""
    G  = rng.standard_normal(u.shape)
    G -= (G * u).sum(axis=1, keepdims=True) * u
    G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-10)
    return t[:, None] * u + np.sqrt(np.maximum(1.0 - t[:, None]**2, 0.0)) * G


def _pu2_sample_t(gamma: float, n: int, from_cap: bool, a: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample t from Beta(a,a) marginal restricted to cap or complement."""
    g   = (1 + gamma) / 2
    out = np.empty(n)
    rem = np.arange(n)
    while len(rem):
        m   = max(len(rem) * 4, 32)
        s   = _beta_dist.rvs(a, a, size=m, random_state=rng)
        acc = s[s >= g] if from_cap else s[s < g]
        k   = min(len(acc), len(rem))
        out[rem[:k]] = 2.0 * acc[:k] - 1.0
        rem = rem[k:]
    return out


def _pu2_find_gamma(eps: float, a: float) -> float:
    """Find optimal γ* for PrivUnit2 via self-consistency A(γ*,ε) = γ*."""
    def _bias(gamma):
        if gamma >= 1:
            return 0.0
        g        = (1 + gamma) / 2
        P        = 1.0 - float(_betainc(a, a, g))
        es_above = 0.5 * (1.0 - float(_betainc(a + 1, a, g)))
        M        = 2.0 * es_above - P
        Z        = np.exp(eps) * P + (1.0 - P)
        return 0.0 if Z < 1e-15 else (np.exp(eps) - 1.0) * M / Z

    return float(_brentq(lambda g: _bias(g) - g, -1 + 1e-8, 1 - 1e-8))


def _pu2_cap_prob(gamma: float, a: float) -> float:
    if gamma <= -1: return 1.0
    if gamma >=  1: return 0.0
    return 1.0 - float(_betainc(a, a, (1 + gamma) / 2))


# ── ANR-SV shared setup ───────────────────────────────────────────────────────

def _anr_sv_setup(W, features, bounds, lambda_n):
    """Shared SVD + SV-weighted λ allocation for ANR-SV variants.

    Returns (d, r, sv, U, sqrt_l, inv_sqrt_l, mu).
    """
    d      = len(features)
    scv    = np.array([bounds[f][1] - bounds[f][0] for f in features])
    W_norm = np.asarray(W, dtype=float) * scv
    W2     = W_norm.reshape(1, -1) if W_norm.ndim == 1 else W_norm

    _, sv_all, Vt = np.linalg.svd(W2, full_matrices=True)
    r  = int(np.sum(sv_all > 1e-10 * sv_all[0]))
    U  = Vt.T
    sv = sv_all[:r]
    mu = 0.5 * np.ones(d)

    if np.isinf(lambda_n):
        C_row           = float(d)
        sqrt_l_null     = np.zeros(d - r)
        inv_sqrt_l_null = np.zeros(d - r)
    else:
        C_row = d - (d - r) / np.sqrt(lambda_n)
        assert C_row > 0, f"lambda_n={lambda_n} too small for d={d}, r={r}"
        sqrt_l_null     = np.full(d - r, np.sqrt(lambda_n))
        inv_sqrt_l_null = 1.0 / sqrt_l_null

    sv23           = sv ** (2.0 / 3.0)
    inv_sqrt_l_row = C_row * sv23 / sv23.sum()
    sqrt_l_row     = 1.0 / inv_sqrt_l_row

    sqrt_l     = np.concatenate([sqrt_l_row,     sqrt_l_null])
    inv_sqrt_l = np.concatenate([inv_sqrt_l_row, inv_sqrt_l_null])

    return d, r, sv, U, sqrt_l, inv_sqrt_l, mu


# ── Simple mechanism classes ──────────────────────────────────────────────────

class NoNoise:
    """No privacy baseline (ε=∞). Returns z unchanged."""
    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        return z.copy()

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


class L1ClipLaplace:
    """Laplace mechanism with L1 norm clipping in raw latent space.

    Clips z to L1 ball of radius ρ (90th percentile of Z_tr L1 norms),
    then adds Laplace noise with scale 2ρ/ε (L1 sensitivity = 2ρ).
    """
    def __init__(self, rho: float):
        self._rho = rho

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        z_clip = _project_l1(z, self._rho)
        return z_clip + rng.laplace(0, 2 * self._rho / eps, z_clip.shape)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


class LinfClipLaplace:
    """Laplace mechanism with L∞ clipping.

    Clips each coordinate to [-ρ, ρ] (ρ = 90th percentile of Z_tr L∞ norms).
    L∞ sensitivity = 2ρ → L1 sensitivity = 2ρd → noise scale 2ρd/ε per coordinate.
    Satisfies ε-LDP for the full vector.
    """
    def __init__(self, rho: float):
        self._rho = rho

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        d = z.shape[1]
        z_clip = np.clip(z, -self._rho, self._rho)
        return z_clip + rng.laplace(0, 2 * self._rho * d / eps, z_clip.shape)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


class GaussianMech:
    """Gaussian mechanism: L2 clip + AGM noise, no preprocessing.

    ρ = 90th percentile of Z_tr L2 norms. L2 sensitivity = 2ρ.
    σ calibrated via AGM (Balle & Wang 2018) for (ε,δ)-LDP.
    """
    def __init__(self, rho: float, delta: float):
        self._rho   = rho
        self._delta = delta
        self._cache: dict[float, float] = {}

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if eps not in self._cache:
            self._cache[eps] = _agm_sigma(eps, self._delta, 2 * self._rho)
        nrm    = np.linalg.norm(z, axis=1, keepdims=True)
        z_clip = z * np.minimum(1.0, self._rho / np.maximum(nrm, 1e-10))
        return z_clip + rng.normal(0, self._cache[eps], z_clip.shape)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return z_enc


class SymClipMech:
    """[-1,1]-based mechanism (Piecewise, Duchi) with symmetric L∞ clip.

    Clips z to [-ρ, ρ], scales to [-1,1] → calls mechanism → scales back.
    ρ = L∞ norm 90th percentile of Z_tr (same scale as ANR L∞).
    """
    def __init__(self, base_mech, rho: float):
        self._enc = base_mech.encode
        self._dec = base_mech.decode
        self._rho = rho

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        z_clip = np.clip(z, -self._rho, self._rho)
        return self._enc((z_clip + self._rho) / (2 * self._rho), eps, rng)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return self._dec(z_enc) * 2 * self._rho - self._rho


class NormMech:
    """[0,1]-based mechanism (Harmony) with per-dim min-max normalization."""
    def __init__(self, base_mech, lo: np.ndarray, scale: np.ndarray):
        self._enc   = base_mech.encode
        self._dec   = base_mech.decode
        self._lo    = lo
        self._scale = scale

    def encode(self, z: np.ndarray, eps: float,
               rng: np.random.Generator = None) -> np.ndarray:
        return self._enc((z - self._lo) / self._scale, eps, rng)

    def decode(self, z_enc: np.ndarray) -> np.ndarray:
        return self._dec(z_enc) * self._scale + self._lo


# ── ANR mechanisms ────────────────────────────────────────────────────────────

def make_anr_sv(W, features, bounds, X_pub_norm,
                lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV(L1,Lap): SV-weighted λ allocation + L1 clip + Laplace noise.

    Row-space budget allocated by: 1/√λ_i ∝ s_i^{2/3} (Lagrange optimum for
    min Σ s_i²·λ_i s.t. Σ 1/√λ_i = C_row).
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)

    lambdas_row = (1.0 / inv_sqrt_l[:r]) ** 2
    print(f"  ANR-SV: r={r}, λ_i(row) min={lambdas_row.min():.4f} "
          f"max={lambdas_row.max():.4f}  "
          f"(uniform λ_r would be "
          f"{(r / d) ** 2 if np.isinf(lambda_n) else 1.0 / ((d - (d - r) / np.sqrt(lambda_n)) / r) ** 2:.4f})")

    x_pub = (X_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.abs(x_pub).sum(axis=1), percentile))

    class _ANRSV:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            x_sc = (z - mu) @ U * inv_sqrt_l
            x_cl = _project_l1(x_sc, rho)
            return x_cl + rng.laplace(0, 2.0 * rho / epsilon, x_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _ANRSV()


def make_anr_sv_linf(W, features, bounds, X_pub_norm,
                     lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV(L∞,Lap): SV-weighted ANR transform + L∞ clip + Laplace noise.

    L∞ sensitivity = 2ρ → L1 sensitivity = 2ρd → noise scale 2ρd/ε per coordinate.
    Satisfies ε-LDP.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)

    x_pub = (X_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.abs(x_pub).max(axis=1), percentile))

    class _ANRSVLinf:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            x_sc = (z - mu) @ U * inv_sqrt_l
            x_cl = np.clip(x_sc, -rho, rho)
            return x_cl + rng.laplace(0, 2.0 * rho * d / epsilon, x_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _ANRSVLinf()


def make_anr_sv_l2agm(W, features, bounds, X_pub_norm,
                      lambda_n=LAMBDA_N, percentile=PERCENTILE, delta=DELTA):
    """ANR-SV(L2,AGM): SV-weighted λ + L2 clipping + AGM noise.

    Same SV-weighted λ allocation as ANR-SV(L1,Lap).
    L2 sensitivity after clipping: Δ₂ = 2ρ.
    Noise: AGM (Balle & Wang 2018), σ = _agm_sigma(ε, δ, 2ρ) → (ε,δ)-LDP.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)

    x_pub = (X_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.linalg.norm(x_pub, axis=1), percentile))
    sens  = 2.0 * rho
    _cache: dict[float, float] = {}

    class _ANRSVL2AGM:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                _cache[epsilon] = _agm_sigma(epsilon, delta, sens)
            x_sc = (z - mu) @ U * inv_sqrt_l
            nrm  = np.linalg.norm(x_sc, axis=1, keepdims=True)
            x_cl = x_sc * np.minimum(1.0, rho / np.maximum(nrm, 1e-10))
            return x_cl + rng.normal(0, _cache[epsilon], x_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _ANRSVL2AGM()


def make_anr_sv_privunit2(W, features, bounds, X_pub_norm,
                          lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV + PrivUnit2: SV-weighted anisotropic transform + spherical step-function.

    Combines ANR's row-space identification with PrivUnit2's optimal step-function.
    ALL d dimensions participate (including null space), so z_null is protected.

    Encoding:
        x     = (z - μ) @ U ⊙ inv_sqrt_λ          (anisotropic scaling)
        u     = x / ‖x‖                              (normalize to S^{d-1})
        z̃  ~ PrivUnit2(u, ε) on S^{d-1}
        out   = (ρ/A) · z̃
    Decoding:
        z_dec = (out ⊙ sqrt_λ) @ U.T + μ

    Pure ε-LDP: inherited from PrivUnit2 on S^{d-1}.
    """
    if np.isinf(lambda_n):
        raise ValueError("lambda_n must be finite: null space needs non-zero weight for privacy.")

    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)

    x_pub = (X_pub_norm - mu) @ U * inv_sqrt_l
    rho   = float(np.percentile(np.linalg.norm(x_pub, axis=1), percentile))
    a     = (d - 1) / 2.0
    _cache: dict = {}

    class _ANRSVPrivUnit2:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma = _pu2_find_gamma(epsilon, a)
                P     = _pu2_cap_prob(gamma, a)
                p_cap = np.exp(epsilon) * P / (np.exp(epsilon) * P + (1.0 - P))
                _cache[epsilon] = (gamma, p_cap)
                print(f"  ANR-SV-PU2: ε={epsilon}, γ*={gamma:.4f}, p_cap={p_cap:.4f}, A=γ*={gamma:.4f}")
            gamma, p_cap = _cache[epsilon]

            x     = (z - mu) @ U * inv_sqrt_l
            u_vec = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)
            n     = z.shape[0]
            use   = rng.random(n) < p_cap
            out   = np.empty_like(u_vec)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sphere_from_t(u_vec[idx],
                                          _pu2_sample_t(gamma, len(idx), from_cap, a, rng),
                                          rng)
            return (rho / gamma) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _ANRSVPrivUnit2()


def make_anr_sv_cw(W, features, bounds, X_pub_norm,
                   lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV-CW(Lap): SV-weighted ANR transform + coordinate-wise i.n.i.d. Laplace.

    Combines SV-weighted anisotropic scaling with per-coordinate budget allocation
    (λ_i^{1/3} split per [25TIFS]) applied in the transformed space.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)

    x_pub   = (X_pub_norm - mu) @ U * inv_sqrt_l
    rho     = np.percentile(np.abs(x_pub), percentile, axis=0)
    lambdas = 2.0 * rho
    norm_23 = float(np.sum(lambdas ** (2.0 / 3.0)))
    beta    = lambdas ** (1.0 / 3.0) * norm_23
    active  = beta > 0

    class _ANRSVCoordWise:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            x = (z - mu) @ U * inv_sqrt_l
            noise = np.zeros_like(x)
            if active.any():
                noise[:, active] = rng.laplace(
                    0, beta[active] / epsilon, (x.shape[0], int(active.sum())))
            return x + noise

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _ANRSVCoordWise()


def make_anr_sv_cw_gaussian(W, features, bounds, X_pub_norm,
                             lambda_n=LAMBDA_N, percentile=PERCENTILE, delta=DELTA):
    """ANR-SV-CW(AGM): SV-weighted ANR transform + coordinate-wise i.n.i.d. Gaussian.

    Same transform as ANR-SV-CW(Lap); noise calibrated via AGM (Balle & Wang 2018)
    with budget split minimizing Σ σ_i² per [25TIFS]. Satisfies (ε,δ)-LDP.
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)

    x_pub   = (X_pub_norm - mu) @ U * inv_sqrt_l
    rho     = np.percentile(np.abs(x_pub), percentile, axis=0)
    lambdas = 2.0 * rho
    scale_i = np.sqrt(lambdas * lambdas.sum())
    active  = scale_i > 0
    _cache: dict[float, float] = {}

    class _ANRSVCWGaussian:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                _cache[epsilon] = _agm_sigma(epsilon, delta, 1.0)
            x = (z - mu) @ U * inv_sqrt_l
            noise = np.zeros_like(x)
            if active.any():
                noise[:, active] = rng.normal(
                    0, _cache[epsilon] * scale_i[active],
                    (x.shape[0], int(active.sum())))
            return x + noise

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (z_enc * sqrt_l) @ U.T + mu

    return _ANRSVCWGaussian()


def make_anr_sv_privunitg(W, features, bounds, X_pub_norm,
                           lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV+PrivUnitG: SV-weighted ANR transform + PrivUnitG step-function."""
    return make_anr_sv_wrap(
        lambda x: make_privunitg(x, percentile),
        W, features, bounds, X_pub_norm, lambda_n, percentile,
    )


def make_anr_sv_duchi(W, features, bounds, X_pub_norm,
                      lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV+Duchi: SV-weighted ANR transform + Duchi mechanism."""
    return make_anr_sv_wrap(
        lambda x: SymClipMech(duchi_ldp, float(np.percentile(np.abs(x).max(axis=1), percentile))),
        W, features, bounds, X_pub_norm, lambda_n, percentile,
    )


def make_anr_sv_piecewise(W, features, bounds, X_pub_norm,
                           lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV+Piecewise: SV-weighted ANR transform + Piecewise mechanism."""
    return make_anr_sv_wrap(
        lambda x: SymClipMech(piecewise_ldp, float(np.percentile(np.abs(x).max(axis=1), percentile))),
        W, features, bounds, X_pub_norm, lambda_n, percentile,
    )


def make_anr_sv_harmony(W, features, bounds, X_pub_norm,
                        lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV+Harmony: SV-weighted ANR transform + Harmony mechanism."""
    def _factory(x):
        lo = x.min(axis=0)
        return NormMech(harmony_ldp, lo, np.maximum(x.max(axis=0) - lo, 1e-8))
    return make_anr_sv_wrap(_factory, W, features, bounds, X_pub_norm, lambda_n, percentile)


def make_anr_sv_wrap(inner_factory, W, features, bounds, X_pub_norm,
                     lambda_n=LAMBDA_N, percentile=PERCENTILE):
    """ANR-SV preprocessing wrapper for any mechanism.

    Applies the ANR-SV transform (SVD rotation + SV-weighted anisotropic scaling)
    as a plug-in preprocessing, then delegates encode/decode to an inner mechanism
    built on the transformed public data.

    inner_factory(x_pub: np.ndarray) -> mechanism with .encode / .decode
    """
    d, r, sv, U, sqrt_l, inv_sqrt_l, mu = _anr_sv_setup(W, features, bounds, lambda_n)
    x_pub = (X_pub_norm - mu) @ U * inv_sqrt_l
    inner = inner_factory(x_pub)

    class _Wrapped:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            return inner.encode((z - mu) @ U * inv_sqrt_l, epsilon, rng)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return (inner.decode(z_enc) * sqrt_l) @ U.T + mu

    return _Wrapped()


# ── Gaussian preprocessing mechanisms ────────────────────────────────────────

def make_plan(X_pub_norm: np.ndarray, p: int = 2,
              percentile: float = PERCENTILE, delta: float = DELTA):
    """PLAN: Variance-Aware Private Mean Estimation [24PET].

    Known-variance scenario: σ̂²_i = var(Z_tr[:,i]), μ̃ = mean(Z_tr).
    Scaling: y = (z - μ̃) · σ̂^{-1/(p+2)}, clip to L2 ball of radius C.
    Gaussian noise: N(0, σ²I), σ = 2C·sqrt(2 log(1.25/δ)) / ε  → (ε,δ)-LDP.
    """
    mu     = X_pub_norm.mean(axis=0)
    var    = np.maximum(X_pub_norm.var(axis=0), 1e-8)
    scale  = var ** (1.0 / (p + 2))
    inv_sc = 1.0 / scale

    Y_pub        = (X_pub_norm - mu) * inv_sc
    C            = float(np.percentile(np.linalg.norm(Y_pub, axis=1), percentile))
    gauss_factor = 2.0 * C * np.sqrt(2.0 * np.log(1.25 / delta))

    class _PLAN:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            y    = (z - mu) * inv_sc
            nrm  = np.linalg.norm(y, axis=1, keepdims=True)
            y_cl = y * np.minimum(1.0, C / np.maximum(nrm, 1e-10))
            return y_cl + rng.normal(0, gauss_factor / epsilon, y_cl.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc * scale + mu

    return _PLAN()


# ── Shifted-CM ────────────────────────────────────────────────────────────────

def make_shifted_cm(X_pub_norm: np.ndarray, delta: float = DELTA):
    """Shifted-CM: instance-optimal LDP mean estimation (Huang et al., NeurIPS 2021).

    Steps (Section 3.3 + Section 5 LDP adaptation):
      1. Random Hadamard rotation: x̂ = (1/√d) H D x,  D = diag(±1) shared randomness.
      2. Per-dim median shift: c̃ = median(X̂_pub, axis=0),  x̃ = x̂ − c̃.
      3. Optimal L2 clip radius C*(ε): minimises bias-variance tradeoff on public norms.
         ∂E/∂C = 0  ⟹  fraction{‖x̃‖>C} = σ_unit(ε) · √(d/n),
         where σ_unit(ε) = AGM(ε, δ, sensitivity=2).
      4. AGM noise: z_clip + N(0, σ²I),  σ = AGM(ε, δ, 2C*).
    Decode: un-shift then un-rotate (both linear, commute with averaging).
    (ε,δ)-LDP.  Requires d to be a power of 2.
    """
    n, d = X_pub_norm.shape
    assert (d & (d - 1)) == 0, f"Shifted-CM requires d = power of 2, got {d}"

    # Shared public randomness: fixed diagonal sign flip
    d_vec = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=d)
    H = _hadamard(d).astype(float)  # H @ H = d · I, H symmetric

    # Rotate public data: X_hat[i] = (1/√d) · H @ (d_vec ⊙ x[i])
    # Batch form: X_hat = (X * d_vec) @ H / √d  (H symmetric ⟹ H = H.T)
    X_hat   = (X_pub_norm * d_vec) @ H / np.sqrt(d)
    c_tilde = np.median(X_hat, axis=0)          # per-dim shift center
    norms   = np.linalg.norm(X_hat - c_tilde, axis=1)

    _cache: dict[float, tuple[float, float]] = {}

    class _ShiftedCM:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                # σ_unit = AGM(ε, δ, 2) so that σ(C) = C · σ_unit
                sigma_unit = _agm_sigma(epsilon, delta, 2.0)
                frac_above = float(np.clip(sigma_unit * np.sqrt(d / n), 0.0, 1.0))
                q          = float(np.clip(1.0 - frac_above, 0.0, 1.0))
                C_opt      = max(float(np.quantile(norms, q)), 1e-8)
                sigma      = _agm_sigma(epsilon, delta, 2.0 * C_opt)
                _cache[epsilon] = (C_opt, sigma)
                print(f"  ShiftedCM: ε={epsilon}, C={C_opt:.4f} "
                      f"(q={100*q:.1f}th pct), σ={sigma:.4f}")
            C_opt, sigma = _cache[epsilon]

            z_hat  = (z * d_vec) @ H / np.sqrt(d)          # rotate
            z_til  = z_hat - c_tilde                        # shift
            nrm    = np.linalg.norm(z_til, axis=1, keepdims=True)
            z_clip = z_til * np.minimum(1.0, C_opt / np.maximum(nrm, 1e-10))
            return z_clip + rng.normal(0, sigma, z_clip.shape)

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            # un-shift then un-rotate: z = (1/√d) · (x̂ @ H) ⊙ d_vec
            return (z_enc + c_tilde) @ H * d_vec / np.sqrt(d)

    return _ShiftedCM()


# ── Spherical mechanisms ──────────────────────────────────────────────────────

def make_privunit2(X_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnit2: optimal step-function mechanism on S^{d-1}. Pure ε-LDP.

    q(z|v) ∝ e^ε · 1[⟨z,v⟩ ≥ γ*] + 1[⟨z,v⟩ < γ*]
    where γ* satisfies the self-consistency condition A(γ*,ε) = γ*.
    Decode scale ρ/γ* gives unbiased estimate of ρ·(z/‖z‖).
    """
    rho = float(np.percentile(np.linalg.norm(X_pub_norm, axis=1), percentile))
    d   = X_pub_norm.shape[1]
    a   = (d - 1) / 2.0
    _cache: dict = {}

    class _PrivUnit2:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma = _pu2_find_gamma(epsilon, a)
                P     = _pu2_cap_prob(gamma, a)
                p_cap = np.exp(epsilon) * P / (np.exp(epsilon) * P + (1.0 - P))
                _cache[epsilon] = (gamma, p_cap)
                print(f"  PrivUnit2: ε={epsilon}, γ*={gamma:.4f}, "
                      f"p_cap={p_cap:.4f}, A=γ*={gamma:.4f}")
            gamma, p_cap = _cache[epsilon]

            n   = z.shape[0]
            u   = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
            use = rng.random(n) < p_cap
            out = np.empty_like(z)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sphere_from_t(u[idx],
                                          _pu2_sample_t(gamma, len(idx), from_cap, a, rng),
                                          rng)
            return (rho / gamma) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _PrivUnit2()


def make_privunitg(X_pub_norm: np.ndarray, percentile: float = PERCENTILE):
    """PrivUnitG (Asi et al., ICML 2022): Gaussian step-function mechanism. Pure ε-LDP.

    Step function on ⟨g,v⟩ ~ N(0,1) in ambient Gaussian space:
        q(g→z|v) ∝ e^ε · 1[⟨g,v⟩≥γ*] + 1[⟨g,v⟩<γ*],  z = g/‖g‖.
    Optimal γ* maximizes A(γ,ε) = E_q[⟨z,v⟩], found via MC grid search.
    """

    rho = float(np.percentile(np.linalg.norm(X_pub_norm, axis=1), percentile))
    d   = X_pub_norm.shape[1]

    def _find_params(eps: float, n_mc: int = 500_000):
        g   = np.random.randn(n_mc, d)
        t   = g[:, 0]
        cos = t / np.linalg.norm(g, axis=1)

        def _A(gamma):
            mask = t >= gamma
            num  = np.exp(eps) * cos[mask].sum() + cos[~mask].sum()
            den  = np.exp(eps) * mask.sum()      + (~mask).sum()
            return float(num / den) if den > 0 else 0.0

        gammas = np.linspace(-3.0, 3.0, 600)
        best   = int(np.argmax([_A(gm) for gm in gammas]))
        lo, hi = gammas[max(0, best - 5)], gammas[min(len(gammas) - 1, best + 5)]
        gammas2 = np.linspace(lo, hi, 300)
        gamma   = float(gammas2[np.argmax([_A(gm) for gm in gammas2])])
        A       = _A(gamma)
        Z       = float(np.exp(eps) * _snorm.sf(gamma) + _snorm.cdf(gamma))
        p_cap   = float(np.exp(eps) * _snorm.sf(gamma) / Z)
        return gamma, p_cap, A

    def _sample_g_batch(u: np.ndarray, gamma: float, from_cap: bool,
                        rng: np.random.Generator) -> np.ndarray:
        n = u.shape[0]
        t = (_truncnorm.rvs(gamma,    np.inf, size=n, random_state=rng) if from_cap else
             _truncnorm.rvs(-np.inf, gamma,   size=n, random_state=rng))
        G  = rng.standard_normal((n, d))
        G -= (G * u).sum(axis=1, keepdims=True) * u
        g  = t[:, None] * u + G
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-10)

    _cache: dict = {}

    class _PrivUnitG:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if rng is None:
                rng = np.random.default_rng()
            if epsilon not in _cache:
                gamma, p_cap, A = _find_params(epsilon)
                _cache[epsilon] = (gamma, p_cap, A)
                print(f"  PrivUnitG: ε={epsilon}, γ*={gamma:.4f}, "
                      f"p_cap={p_cap:.4f}, A={A:.4f}")
            gamma, p_cap, A = _cache[epsilon]

            n   = z.shape[0]
            u   = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
            use = rng.random(n) < p_cap
            out = np.empty_like(z)

            for from_cap, mask in [(True, use), (False, ~use)]:
                idx = np.where(mask)[0]
                if not len(idx): continue
                out[idx] = _sample_g_batch(u[idx], gamma, from_cap, rng)
            return (rho / A) * out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            return z_enc

    return _PrivUnitG()


# ── Jacobian row-space computation ────────────────────────────────────────────

def compute_jacobian_row_space(model: torch.nn.Module, Z_pub: torch.Tensor,
                               n_samples: int = 500,
                               sv_gap_ratio: float = 1e-2) -> np.ndarray:
    """Aggregate Jacobians and extract a clean row-space basis matrix.

    Stacks per-sample Jacobians into B ∈ ℝ^{(n_samples*K) × D}, then uses SVD
    with a gap-based rank threshold to separate row space from numerical null space.

    Returns W_eff ∈ ℝ^{r × D}  (rank-r).
    """
    model.eval()
    device = next(model.parameters()).device
    idx    = torch.randperm(len(Z_pub))[:n_samples]
    Z_sub  = Z_pub[idx].to(device)
    K      = model(Z_sub[:1]).shape[1]

    rows = []
    for i in range(len(Z_sub)):
        z   = Z_sub[i:i+1].clone().float().requires_grad_(True)
        out = model(z)
        for k in range(K):
            grad = torch.autograd.grad(out[0, k], z,
                                       retain_graph=(k < K - 1))[0]
            rows.append(grad.detach().cpu().numpy().flatten())

    B = np.array(rows, dtype=np.float64)
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    r = int(np.sum(s > sv_gap_ratio * s[0]))
    print(f"  Jacobian: B{B.shape}, effective rank={r} "
          f"(sv gap at {s[r-1]:.2f} → {s[r]:.4f}, ratio={s[r]/s[0]:.2e})")
    return np.diag(s[:r]) @ Vt[:r]


# ── Cheng et al. (ICML 2022) ─────────────────────────────────────────────────

def _cheng_water_fill(eigvals: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """Water-filling for optimal σ²_i (Prop 4.6). Returns (sigma2[:Zp], Zp)."""
    sqrt_lam = np.sqrt(eigvals)
    for Zp in range(len(eigvals), 0, -1):
        S = sqrt_lam[:Zp].sum()
        if S < 1e-12:
            continue
        sigma2 = sqrt_lam[:Zp] / S * (1.0 + Zp * alpha) - alpha
        if sigma2[-1] >= 0.0:
            return np.maximum(sigma2, 0.0), Zp
    return np.array([1.0]), 1


def _cheng_setup(W, features, bounds, X_pub_norm, percentile):
    """Cholesky whitening setup for make_cheng.

    Returns (d, mu, L, L_inv, r, P) where:
      mu:    public data mean
      L:     Cholesky factor of Cov(X_pub)
      L_inv: inverse of L
      r:     `percentile`-th percentile of ||h||_2 over public data
      P:     task matrix in whitened space (W_norm @ L)
    """
    d      = len(features)
    mu     = X_pub_norm.mean(axis=0)
    X_c    = X_pub_norm - mu
    Cov    = (X_c.T @ X_c) / len(X_pub_norm) + 1e-8 * np.eye(d)
    L      = np.linalg.cholesky(Cov)
    L_inv  = np.linalg.inv(L)
    r      = float(np.percentile(np.linalg.norm(X_c @ L_inv.T, axis=1), percentile))
    scale  = np.array([bounds[features[j]][1] - bounds[features[j]][0] for j in range(d)])
    W_norm = np.asarray(W, dtype=float) * scale
    W2     = W_norm.reshape(1, -1) if W_norm.ndim == 1 else W_norm
    P      = W2 @ L
    return d, mu, L, L_inv, r, P


def make_cheng(W, features, bounds, X_pub_norm, percentile=PERCENTILE):
    """Cheng et al. (2022, ICML) task-aware LDP mechanism for linear MSE settings.

    Encoder-decoder framework (Section 4.1):
      1. Whiten x via Cholesky of Cov(X_pub): h = (x - μ) @ L^{-T}.
      2. Rotate h into the task-relevant basis Q (eigenvectors of P^T P,
         P = W_norm @ L).
      3. Scale each dimension by σ_i from the water-filling solution (Prop. 4.6):
         σ²_i = (√λ_i / ΣZ'√λ_j)(1 + Z'·8r²/ε²) − 8r²/ε²  for i ≤ Z', 0 o.w.
      4. Add Laplace noise with sensitivity 2r (scale = 2r/ε) per encoded dim.
      5. Decode: x̂ = L · D · φ̃ + μ  where D = E^T(EE^T + 8r²/ε² · I)^{-1}.

    Warning: assumes features are on a small, normalized scale. When applied to
    VAE latent vectors the Cholesky-whitened clip radius r is large (~4.9 for
    D=16), making α = 8r²/ε² >> 1 for all practical ε. Water-filling then
    collapses to Zp=1 active dimension (verified for ε ≤ 5), concentrating
    decoded outputs to near-constant vectors and degrading classification utility.
    """
    d, mu, L, L_inv, r, P = _cheng_setup(W, features, bounds, X_pub_norm, percentile)

    PtP                = P.T @ P
    eigvals_raw, Q_raw = np.linalg.eigh(PtP)
    idx     = np.argsort(eigvals_raw)[::-1]
    eigvals = np.maximum(eigvals_raw[idx], 0.0)
    Q       = Q_raw[:, idx]

    _cache:    dict[float, tuple] = {}
    _last_eps: list               = [None]

    class _Cheng:
        def encode(self, z: np.ndarray, epsilon: float,
                   rng: np.random.Generator = None) -> np.ndarray:
            if epsilon not in _cache:
                alpha      = 8.0 * r ** 2 / epsilon ** 2
                sigma2, Zp = _cheng_water_fill(eigvals, alpha)
                E          = np.sqrt(sigma2)[:, None] * Q[:, :Zp].T  # (Zp, d)
                D          = E.T @ np.linalg.inv(E @ E.T + alpha * np.eye(Zp))
                _cache[epsilon] = (D, Zp, E)
            _last_eps[0] = epsilon
            D, Zp, E     = _cache[epsilon]

            if rng is None:
                rng = np.random.default_rng()
            h     = (z - mu) @ L_inv.T
            norms = np.linalg.norm(h, axis=1, keepdims=True)
            h     = h * np.minimum(1.0, r / np.maximum(norms, 1e-12))
            phi   = h @ E.T
            phi_noisy     = phi + rng.laplace(0.0, 2.0 * r / epsilon, phi.shape)
            z_out         = np.zeros((len(z), d))
            z_out[:, :Zp] = phi_noisy
            return z_out

        def decode(self, z_enc: np.ndarray) -> np.ndarray:
            if _last_eps[0] is None:
                raise RuntimeError("decode called before encode")
            D, Zp, _ = _cache[_last_eps[0]]
            return z_enc[:, :Zp] @ D.T @ L.T + mu

    return _Cheng()


# ── Mechanism registry ────────────────────────────────────────────────────────

def build_latent_mechs(dim: int, W: np.ndarray, Z_np: np.ndarray) -> dict:
    """Build LDP mechanism dict."""
    lo    = Z_np.min(axis=0)
    hi    = Z_np.max(axis=0)
    scale = np.maximum(hi - lo, 1e-8)

    rho_l1   = float(np.percentile(np.abs(Z_np).sum(axis=1), PERCENTILE))
    rho_linf = float(np.percentile(np.abs(Z_np).max(axis=1), PERCENTILE))
    rho_l2   = float(np.percentile(np.linalg.norm(Z_np, axis=1), PERCENTILE))

    features = [str(j) for j in range(dim)]
    bounds   = {str(j): (0.0, 1.0) for j in range(dim)}
    anr_kw   = dict(W=W, features=features,
                    bounds=bounds,
                    X_pub_norm=Z_np, lambda_n=LAMBDA_N, percentile=PERCENTILE)

    mechs = {
        "NoNoise":          NoNoise(),
        "Laplace(L∞)":      LinfClipLaplace(rho_linf),
        "ANR-SV(L∞,Lap)":   make_anr_sv_linf(**anr_kw),
        "Laplace(L1)":      L1ClipLaplace(rho_l1),
        "ANR-SV(L1,Lap)":   make_anr_sv(**anr_kw),
        "AGM":              GaussianMech(rho_l2, DELTA),
        "ANR-SV(L2,AGM)":   make_anr_sv_l2agm(**anr_kw, delta=DELTA),
        "Duchi":            SymClipMech(duchi_ldp,     rho_linf),
        "ANR-SV+Duchi":     make_anr_sv_duchi(**anr_kw),
        "Piecewise":        SymClipMech(piecewise_ldp, rho_linf),
        "ANR-SV+Piecewise": make_anr_sv_piecewise(**anr_kw),
        "Harmony":          NormMech(harmony_ldp,      lo, scale),
        "ANR-SV+Harmony":   make_anr_sv_harmony(**anr_kw),
        "PrivUnit2":        make_privunit2(Z_np),
        "ANR-SV+PrivUnit2": make_anr_sv_privunit2(**anr_kw),
        "PrivUnitG":        make_privunitg(Z_np),
        "ANR-SV+PrivUnitG": make_anr_sv_privunitg(**anr_kw),
        "ANR-SV-CW(Lap)":   make_anr_sv_cw(**anr_kw),
        "ANR-SV-CW(AGM)":   make_anr_sv_cw_gaussian(**anr_kw, delta=DELTA),
        "PLAN":             make_plan(Z_np, delta=DELTA),
        # "TASK(Cheng22)":    make_cheng(W=W, features=features, bounds=bounds, X_pub_norm=Z_np),
    }
    # Shifted-CM requires d = power of 2 (Hadamard transform)
    if (dim & (dim - 1)) == 0:
        mechs["Shifted-CM"] = make_shifted_cm(Z_np, delta=DELTA)
    return mechs

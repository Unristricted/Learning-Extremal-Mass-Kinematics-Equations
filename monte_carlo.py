import numpy as np
from preprocess import preprocess_support_set

class evaluations():
    def __init__(self, y_val, jacobian, density, exponent, local_sum, inner_exponent, g_val):
        self.y_val = y_val
        self.jacobian = jacobian
        self.density = density
        self.exponent = exponent
        self.local_sum = local_sum
        self.inner_exponent = inner_exponent
        self.g_val = g_val

    def __lt__(self, other):
        return self.y_val < other.y_val

    def __le__(self, other):
        return self.y_val <= other.y_val

    def __eq__(self, other):
        return self.y_val == other.y_val

    def __ne__(self, other):
        return self.y_val != other.y_val

    def __gt__(self, other):
        return self.y_val > other.y_val

    def __ge__(self, other):
        return self.y_val >= other.y_val

    def __str__(self):
        return f"""Evaluated Value: {self.y_val}\n                   Density:         {self.density}\n                   Exponent:        {self.exponent}\n                   Local Sum:       {self.local_sum}\n                   New Exponent:    {np.exp(-(self.y_val**2))}\n                   Inner Exponent:  {self.inner_exponent}\n                   g_val:           {self.g_val}"""

# -----------------------------------------------------------------------------
# Numerical utility helpers
# -----------------------------------------------------------------------------

def taylor_exp(x: float, tol: float = 1e-12, max_terms: int = 50) -> float:
    """Compute e**x using a truncated Taylor series to improve stability.

    Parameters
    ----------
    x : float
        Exponent argument.
    tol : float, optional
        Absolute tolerance for early‑stopping the series. Defaults to 1e‑12.
    max_terms : int, optional
        Maximum number of summation terms. Defaults to 50.

    Notes
    -----
    *   For moderately large |x|, the direct call ``np.exp`` can overflow/underflow.
    *   This implementation accumulates terms until the incremental contribution is
        smaller than *tol* **or** *max_terms* is reached.
    *   When |x| is very large (|x| > 50), numerical issues are unavoidable; in such
        cases we fall back to ``np.exp`` so behaviour is explicit.
    """
    if abs(x) > 50:
        # We are outside the region where a truncated series is numerically useful.
        return float(np.exp(x))

    term = 1.0  # first term (x^0 / 0!)
    result = 1.0
    n = 1
    while n < max_terms:
        term *= x / n
        result += term
        if abs(term) < tol:
            break
        n += 1
    return result

# -----------------------------------------------------------------------------
# Geometry helpers (unchanged)
# -----------------------------------------------------------------------------

def project_onto_plane(vec, normal):
    denom = np.dot(normal, normal)
    if denom < 1e-14:
        return vec
    return vec - (np.dot(vec, normal) / denom) * normal


def sample_x_ij(a_i, a_j, t, n):
    normal = a_i - a_j
    norm_sq = np.dot(normal, normal)
    if norm_sq < 1e-14:
        print("norm_sq is 0 – assigning random value")
        return np.random.randn(n)
    L = 2.0 * (np.log(t + 1e-9) + np.log(n + 1e-9))
    limit_alpha = L / norm_sq
    alpha = np.random.uniform(-limit_alpha, limit_alpha)
    Y = np.random.randn(n)
    Y_proj = project_onto_plane(Y, normal)
    x = Y_proj + alpha * normal
    return x


def sample_random_matrix(C, delta):
    X = C + delta * np.abs(C) * np.random.randn(*C.shape)
    return X

# -----------------------------------------------------------------------------
# Core Monte‑Carlo helpers – **modified compute_g uses taylor_exp**
# -----------------------------------------------------------------------------

def compute_g(Xmat, A, x):
    """Compute the vector g(x) with a stable exponential implementation."""
    n = Xmat.shape[0]
    t_plus_1 = Xmat.shape[1]
    g_val = np.zeros(n)
    for i in range(n):
        for j in range(1, t_plus_1):
            exponent = float(np.dot(A[j], x))
            exp_val = taylor_exp(exponent)  # *** numerically stable exp ***
            g_val[i] -= Xmat[i, j] * exp_val
    return g_val


def compute_jacobian(Xmat, A, x):
    n = Xmat.shape[0]
    t_plus_1 = Xmat.shape[1]
    dim = A.shape[1]
    J = np.zeros((n, dim))
    for i in range(n):
        for j in range(1, t_plus_1):
            exponent = float(np.dot(A[j], x))
            exp_val = taylor_exp(exponent)
            J[i, :] += -Xmat[i, j] * exp_val * A[j]
    return J

# -----------------------------------------------------------------------------
# Statistical densities & polynomial evaluation (unchanged)
# -----------------------------------------------------------------------------

def density_v_of_g(g_val, C0, delta, sigma_scale=1000.0):
    g_val = np.asarray(g_val)
    C0 = np.asarray(C0)
    log_density = 0.0
    for i in range(len(g_val)):
        mu = C0[i]
        sigma = delta * abs(mu) * sigma_scale if abs(mu) >= 1e-14 else delta * 1e-6 * sigma_scale
        log_coeff = -np.log(np.sqrt(2 * np.pi) * sigma)
        log_exponent = -0.5 * (((g_val[i] - mu) / sigma) ** 2)
        log_density += log_coeff + log_exponent
    density = np.exp(log_density)
    return density, log_density, 0.0


def compute_polynomial(A, C, x):
    total = 0.0
    for p in range(C.shape[0]):
        inner_sum = 0.0
        for i in range(A.shape[0]):
            inner_sum += C[p][i] * x[0] ** A[i][0] * x[1] ** A[i][1]
        total += inner_sum * inner_sum
    return total

# -----------------------------------------------------------------------------
# Main Monte Carlo routine (unchanged except it now picks up new compute_g)
# -----------------------------------------------------------------------------

def monte_carlo_kac_rice(A, C, delta, n_samples=50, M=30, seed=None, domain=None):
    if seed is not None:
        np.random.seed(seed)

    t_plus_1 = A.shape[0]
    t = t_plus_1 - 1
    n = C.shape[0]
    dim = A.shape[1]
    c0 = C[:, 0]
    Z0 = 0.0

    # Store the 10 smallest polynomial evaluations for debugging
    min_poly_values = []
    indices = list(range(t_plus_1))

    for i in indices:
        for j in indices:
            if j == i:
                continue
            for _ in range(n_samples):
                if domain is not None:
                    lower, upper = domain
                    x = np.random.uniform(lower, upper, size=dim)
                else:
                    x = sample_x_ij(A[i], A[j], t, dim)

                local_sum = 0.0
                for _m in range(M):
                    Xmat = sample_random_matrix(C, delta)
                    g_val = compute_g(Xmat, A, x)
                    J = compute_jacobian(Xmat, A, x)
                    detJ = abs(np.linalg.det(J))
                    density_val, log_exponent, inner_exponent = density_v_of_g(g_val, c0, delta)
                    local_sum += detJ * density_val

                    # Diagnostics: track polynomial values
                    y = compute_polynomial(A, C, x)
                    poly_val = evaluations(y, J, density_val, log_exponent, local_sum, inner_exponent, g_val)
                    if len(min_poly_values) < 10:
                        min_poly_values.append(poly_val)
                    else:
                        current_max = max(min_poly_values)
                        if poly_val < current_max:
                            max_idx = min_poly_values.index(current_max)
                            min_poly_values[max_idx] = poly_val

                Z0 += local_sum / M / n_samples

    num_pairs = t_plus_1 * (t_plus_1 - 1)
    Z0_final = (2.0 / (t * (t + 1))) * Z0 if t >= 1 else (2.0 / num_pairs) * Z0

    print("10 least values of compute_polynomial:")
    for ev in sorted(min_poly_values):
        print(ev)

    return Z0_final

# -----------------------------------------------------------------------------
# Wrapper with preprocessing (unchanged)
# -----------------------------------------------------------------------------

def run_monte_carlo_with_preprocessing(A, C, delta, n_samples=50, M=30, seed=None, domain=None):
    A_tilde, U, v = preprocess_support_set(A)
    estimate = monte_carlo_kac_rice(A_tilde, C, delta, n_samples, M, seed, domain)
    return estimate, A_tilde, U, v

# -----------------------------------------------------------------------------
# Script entry‑point example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    d = 3
    A = np.array([
        [2 * d, 0],
        [0, 2 * d],
        [0, d],
        [d, 0],
        [0, 1],
        [1, 0],
    ], dtype=float)

    s = t = 44 / 31
    C = np.array(
        [
            [1.0, 1.0, s, t, -1, -1],
            [1.0, -1.0, s, -t, -1, 1],
        ]
    )
    delta = 0.3
    domain = (np.array([0.1, 0.1]), np.array([1.0, 1.0]))

    for i in range(5):
        est = monte_carlo_kac_rice(A, C, delta, n_samples=1000, M=5, domain=None)
        print(f"Run {i + 1}: Monte Carlo Kac‑Rice estimate = {est}")

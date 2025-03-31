import numpy as np
from preprocess import preprocess_support_set

def project_onto_plane(vec, normal):
    denom = np.dot(normal, normal)
    if denom < 1e-14:
        return vec
    return vec - (np.dot(vec, normal)/denom) * normal

def sample_x_ij(a_i, a_j, t, n):
    normal = a_i - a_j
    norm_sq = np.dot(normal, normal)
    if norm_sq < 1e-14:
        return np.random.randn(n)
    L = 2.0 * (np.log(t+1e-9) + np.log(n+1e-9))
    limit_alpha = L / norm_sq
    alpha = np.random.uniform(-limit_alpha, limit_alpha)
    Y = np.random.randn(n)
    Y_proj = project_onto_plane(Y, normal)
    x = Y_proj + alpha * normal
    return x

def sample_random_matrix(C, delta):
    X = C + delta * np.abs(C) * np.random.randn(*C.shape)
    return X

def compute_g(Xmat, A, x):
    n = Xmat.shape[0]
    t_plus_1 = Xmat.shape[1]
    g_val = np.zeros(n)
    for i in range(n):
        for j in range(1, t_plus_1):
            exponent = np.dot(A[j], x)
            g_val[i] -= Xmat[i, j] * np.exp(exponent)
    return g_val

def compute_jacobian(Xmat, A, x):
    n = Xmat.shape[0]
    t_plus_1 = Xmat.shape[1]
    dim = A.shape[1]
    J = np.zeros((n, dim))
    for i in range(n):
        for j in range(1, t_plus_1):
            exponent = np.dot(A[j], x)
            J[i, :] += -Xmat[i, j] * np.exp(exponent) * A[j]
    return J

def density_v_of_g(g_val, C0, delta):
    n = len(g_val)
    val = 1.0
    for i in range(n):
        mu = C0[i]
        sigma = delta * abs(mu)
        if abs(mu) < 1e-14:
            sigma = delta * 1e-6
        coeff = 1.0 / (np.sqrt(2.0*np.pi) * sigma)
        exponent = -0.5 * ((g_val[i] - mu)/sigma)**2
        val *= coeff * np.exp(exponent)
    return val

def monte_carlo_kac_rice(A, C, delta, n_samples=50, M=30, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t_plus_1 = A.shape[0]
    t = t_plus_1 - 1
    n = C.shape[0]
    dim = A.shape[1]
    c0 = C[:, 0]
    Z0 = 0.0
    indices = list(range(t_plus_1))
    for i in indices:
        for j in indices:
            if j == i:
                continue
            for _ in range(n_samples):
                x_ij = sample_x_ij(A[i], A[j], t, dim)
                local_sum = 0.0
                for _m in range(M):
                    Xmat = sample_random_matrix(C, delta)
                    g_val = compute_g(Xmat, A, x_ij)
                    J = compute_jacobian(Xmat, A, x_ij)
                    detJ = np.abs(np.linalg.det(J))
                    density_val = density_v_of_g(g_val, c0, delta)
                    local_sum += detJ * density_val
                Z0 += local_sum / M / n_samples
    num_pairs = t_plus_1 * (t_plus_1 - 1)
    if t >= 1:
        Z0_final = (2.0 / (t*(t+1))) * Z0
    else:
        Z0_final = (2.0 / num_pairs) * Z0
    return Z0_final

def run_monte_carlo_with_preprocessing(A, C, delta, n_samples=50, M=30, seed=None):
    A_tilde, U, v = preprocess_support_set(A)
    estimate = monte_carlo_kac_rice(A_tilde, C, delta, n_samples, M, seed)
    return estimate, A_tilde, U, v

def run_monte_carlo(A, C, delta, n_samples=50, M=30, seed=None):
    estimate = monte_carlo_kac_rice(A, C, delta, n_samples, M, seed)
    return estimate

if __name__ == "__main__":
    A = np.array([
        [0, 0],
        [3, 0],
        [0, 3],
        [2, 2]
    ], dtype=float)
    C = np.array([
        [1.0, 2.0, 1.5, 0.8],
        [0.5, 1.2, 2.2, 1.1],
    ])
    delta = 0.3
    estimate, A_tilde, U, v = run_monte_carlo_with_preprocessing(
        A, C, delta, n_samples=10, M=5, seed=123
    )
    print("With PreProcessing")
    print("Monte Carlo Kac-Rice estimate:", estimate)
    print("Transformed support set A_tilde:\n", A_tilde)
    print("Transformation matrix U:\n", U)
    print("Chosen vertex v:\n", v)

    print("Without PreProcessing")
    estimate = run_monte_carlo(A, C, delta, n_samples=1000, M=5, seed=123)
    print("Monte Carlo Kac-Rice estimate:", estimate)

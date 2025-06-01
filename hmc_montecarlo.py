import numpy as np
from preprocess import preprocess_support_set
from scipy.special import logsumexp
from scipy.special import expm1
from scipy.spatial import ConvexHull
import jax
import jax.numpy as jnp
import blackjax

class evaluations():
    def __init__(self, y_val, jacobian, density, g_val):
        self.y_val = y_val
        self.jacobian = jacobian
        self.density = density
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
        return f"""Value of Polynomial System at sampled point: {self.y_val}\n
                  Density:         {self.density}\n
                  Jacobian Determinant:        { abs(np.linalg.det(self.jacobian)) }\n
                 g_val:           {self.g_val}"""

# -----------------------------------------------------------------------------
# Numerical and Geometry Helpers
# -----------------------------------------------------------------------------

def taylor_exp(x: float, tol: float = 1e-12, max_terms: int = 50) -> float:
    if abs(x) > 50:
        return float(np.exp(x))
    term, result = 1.0, 1.0
    n = 1
    while n < max_terms:
        term *= x / n
        result += term
        if abs(term) < tol:
            break
        n += 1
    return result


def project_onto_plane(vec, normal):
    denom = np.dot(normal, normal)
    if denom < 1e-14:
        return vec
    return vec - (np.dot(vec, normal) / denom) * normal


def sample_x_ij(a_i, a_j, n):
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


def compute_polynomial(A, C, x):
    total = 0.0
    for p in range(C.shape[0]):
        inner_sum = 0.0
        for i in range(A.shape[0]):
            inner_sum += C[p][i] *  np.dot(A[i], x)
        total += inner_sum * inner_sum
    return total


def compute_jacobian(Xmat, A, x):
    n = Xmat.shape[0]
    t_plus_1 = Xmat.shape[1]
    dim = A.shape[1]
    J = np.zeros((n, dim))
    for i in range(n):
        for j in range(1, t_plus_1):
            exponent = float(np.dot(A[j], x))
            if exponent**2 > 1:
                exp_val = expm1(exponent) + 1.0
            else:
                t = exponent
                exp_val = 1- t + t**2/2-t**3/6+t**4/24-t**5/120+t**6/720-t**7/5040+t**8/40320
            J[i, :] += -Xmat[i, j] * exp_val * A[j]
    return J


def compute_g(Xmat, A, x):
    n = Xmat.shape[0]
    t_plus_1 = Xmat.shape[1]
    g_val = jnp.zeros(n)

    def compute_exp(exponent):
            return jnp.where(
                exponent**2 > 1,
                jnp.expm1(exponent) + 1.0,
                1 - exponent + exponent**2 / 2 - exponent**3 / 6 +
                exponent**4 / 24 - exponent**5 / 120 +
                exponent**6 / 720 - exponent**7 / 5040 +
                exponent**8 / 40320
            )
    for i in range(n):
        for j in range(1, t_plus_1):
            exponent = jnp.dot(A[j], x)
            exp_val = compute_exp(exponent)
            g_val = g_val.at[i].add(Xmat[i, j] * exp_val)

    return g_val


def density_v_of_g(g_val, C0, delta):
    #g_val = np.asarray(g_val)
    C0 = np.asarray(C0)
    density = 1.00
    density_exp = np.arange(len(g_val))
    for i in range(len(g_val)):
        mu = C0[i]
        t = - 0.5 * (  (g_val[i]/mu - 1) / delta  )**2
        if t < -1e-14:
            density_exp[i] = min(np.log(0.0001), t* (delta**3))
        else:
            density_exp[i] = t
    log_density = logsumexp(density_exp)
    density = np.exp(log_density)
    return density


def monte_carlo_kac_rice(A, C0, delta, n_samples=100, M=100, seed=None):
    B = preprocess_support_set(A)
    hull = ConvexHull(B)
    hull_points = B[hull.vertices]
    affine_basis = hull_points - hull_points[0]
    dim = np.linalg.matrix_rank(affine_basis)
    c0 = C[:, 0]
    Z_0 = 0.000

    # Store the 5 smallest polynomial evaluations for debugging
    min_poly_values = []
    for i in hull.vertices:
        for j in hull.vertices:
            if i == j:
                continue
            if np.linalg.norm(A[i] - A[j]) < 1e-14:
                continue
            else:
                for t in range(n_samples):
                    x = sample_x_ij(B[i], B[j], dim)
                    Z_0 = 0.000
                    local_sum = 0.0
                    for m in range(M):
                            Xmat = sample_random_matrix(C, delta)
                            g_val = compute_g(Xmat, B, x)
                            J = compute_jacobian(Xmat, B, x)
                            detJ = abs(np.linalg.det(J))
                            density_val = density_v_of_g(g_val, c0, delta)
                            local_sum =  (m/(m+1))* local_sum +  (1/(m+1)) * detJ * density_val
                    Z_0 = (t/(t+1))*Z_0 + (1/(t+1))* local_sum

                # Diagnostics: track polynomial values
                y = compute_polynomial(B, C, x)
                poly_val = evaluations(y, J, density_val, g_val)
                if len(min_poly_values) < 5:
                    min_poly_values.append(poly_val)
                else:
                    current_max = max(min_poly_values)
                    if poly_val < current_max:
                        max_idx = min_poly_values.index(current_max)
                        min_poly_values[max_idx] = poly_val

                print("5 least values of compute_polynomial:")
                for ev in sorted(min_poly_values):
                    print(ev)

    return Z_0

def preprocess_routine(A):
    B = preprocess_support_set(A)
    hull = ConvexHull(B)
    hull_points = B[hull.vertices]
    affine_basis = hull_points - hull_points[0]
    dim = np.linalg.matrix_rank(affine_basis)

    return B, hull, dim

def hmc_monte_carlo_kac_rice(A, C, delta,
                              n_samples=100,
                              M=30,
                              hmc_params=None):

    Xmat = sample_random_matrix(C, delta)

    B, hull, dim = preprocess_routine(A)

    def log_density_fn(x):
        x = jnp.asarray(x, dtype=jnp.float32)
        g = compute_g(Xmat, B, x)
        return -1*density_v_of_g(g, C[:, 0], delta)

    if hmc_params is None:
        raise ValueError("hmc_params must be provided for HMC sampling")

    #Add options for other params at later stage if user wants to use them
    rng_key = hmc_params['rng_key']

    for i in hull.vertices:
        for j in hull.vertices:
            if i == j:
                continue
            if np.linalg.norm(A[i] - A[j]) < 1e-14:
                continue
            else:
                for t in range(n_samples):
                    #burn in for inverse mass matrix
                    initial_position = sample_x_ij(B[0], B[1], B.shape[1])
                    warmup = blackjax.window_adaptation(blackjax.nuts, log_density_fn)
                    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
                    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1000)
                    kernel = blackjax.nuts(log_density_fn, **parameters).step
                    states = blackjax.inference_loop(sample_key, kernel, state, 1_000)
                    inverse_mass = parameters['inverse_mass_matrix']
                    step_size = parameters['step_size']

                    #HMC sampling
                    nuts = blackjax.nuts(log_density_fn, step_size, inverse_mass)

                    n_steps = nuts.num_integration_steps

                    sample_hmc = blackjax.hmc(log_density_fn, step_size, inverse_mass, n_steps)
                    state = states[-1]
                    hmc_step = jax.jit(sample_hmc.step)

                    def one_hmc(state, k):
                        state, _ = hmc_step(k, state)
                        return state, np.array(state.position)

                    keys = jax.random.split(rng_key, n_samples)
                    _, chain = jax.lax.scan(one_hmc, state, keys)

                    # Kac-Rice estimate over HMC chain
                    c0 = C[:, 0]
                    Z = 0.0

                    for x in chain:
                        local = 0.0
                        for _ in range(M):
                            g_val = compute_g(Xmat, A, x)
                            J = compute_jacobian(Xmat, A, x)
                            local += abs(np.linalg.det(J)) * density_v_of_g(g_val, c0, delta)
                        Z += local / M

                    return Z


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
    delta = 0.0003

    hmc_params = {
        'rng_key': jax.random.PRNGKey(123),
    }

    estimate = hmc_monte_carlo_kac_rice(
        A, C, delta,
        n_samples=200, M=10,
        hmc_params=hmc_params
    )
    print("HMC-based Kac-Rice estimate:", estimate)

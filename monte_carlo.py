import numpy as np
from scipy.special import logsumexp
from scipy.special import expm1
from scipy.spatial import ConvexHull
from preprocess import preprocess_support_set

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

# -----------------------------------------------------------------------------
# Sampling Functions
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
#   Monte‑Carlo  and Polynomial Evaluation Helper Functions 
# -----------------------------------------------------------------------------

def compute_polynomial(A, C, x):
    total = 0.0
    for p in range(C.shape[0]):
        inner_sum = 0.0
        for i in range(A.shape[0]):
            inner_sum += C[p][i] *  np.dot(A[i], x)
        total += inner_sum * inner_sum
    return total


#We should replace all exmp1 with log-exp-sum trick. Just tried now to fix it.

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
    g_val = np.zeros(n)
    for i in range(n):
        g_val[i] = 0.0
        for j in range(1, t_plus_1):
            exponent = float(np.dot(A[j], x))
            if exponent**2 > 1:
                exp_val = expm1(exponent) + 1.0
            else:
                t = exponent
                exp_val = 1- t + t**2/2-t**3/6+t**4/24-t**5/120+t**6/720-t**7/5040+t**8/40320
            g_val[i] += Xmat[i, j] * exp_val 
    return g_val


def density_v_of_g(g_val, C0, delta):
    g_val = np.asarray(g_val)
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



# -----------------------------------------------------------------------------
# Main Monte Carlo routine 
# -----------------------------------------------------------------------------

def monte_carlo_kac_rice(A, C, delta, n_samples=100, M=100, seed=None):
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


    delta = 0.0001
 
    final_estimate = 0.0
    for i in range(10):
        est = monte_carlo_kac_rice(A, C, delta, n_samples=100, M=100)
        final_estimate =  (  i / (i+1)) * final_estimate +  (1/ (i+1)) * est
        print(f"My current estimate is {final_estimate}")

 
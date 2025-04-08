import numpy as np
from scipy.spatial import ConvexHull



def preprocess_support_set(A):
    hull = ConvexHull(A)
    vertex_indices = hull.vertices
    hull_points = A[hull.vertices]
    affine_basis = hull_points - hull_points[0]
    affine_dim = np.linalg.matrix_rank(affine_basis)


    for i in vertex_indices:
         pivot = A[i]
         active_constraints = []
         selected_normals = []
         tolerance = 1e-10  # Adjust tolerance if needed
         for j, equation in enumerate(hull.equations):
            if abs(np.dot(equation[:-1], pivot) + equation[-1]) < tolerance:
                active_constraints.append(equation[:-1])    
         
         dimconstraint = np.linalg.matrix_rank(active_constraints)   
         if dimconstraint == affine_dim:
             for normal in active_constraints:
                 candidate = np.vstack(selected_normals + [normal])
                 if np.linalg.matrix_rank(candidate) == len(selected_normals) + 1:
                    selected_normals.append(normal)
                 elif len(selected_normals) == affine_dim:
                     break
    H = np.vstack(selected_normals)
    U = H
    B = (U @ A.T).T
    v_transformed = U @ pivot
    A_tilde = B - v_transformed
    return A_tilde, U, pivot

def build_support_set_for_system(d):
    A = np.array([
        [2*d, 0],
        [0, 2*d],
        [d,   0],
        [0,   d],
        [1,   0],
        [0,   1],
    ], dtype=float)
    return A


A = build_support_set_for_system(3)
A_tilde, U, v = preprocess_support_set(A)
print("Original support set A (exponents):")
print(A)
print("\nChosen vertex:")
print(v)
print("\nTransformation matrix U:")
print(U)
print("\nPreprocessed support set A_tilde:")
print(A_tilde)

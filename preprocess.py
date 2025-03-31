import numpy as np
from scipy.spatial import ConvexHull

def preprocess_support_set(A):
    #TODO: Will need to figure out how to handle N vertices not only 2
    hull = ConvexHull(A)
    vertex_index = np.argmin(A.sum(axis=1))
    v = A[vertex_index]
    facet_normals = []
    for i, simplex in enumerate(hull.simplices):
        if vertex_index in simplex:
            eq = hull.equations[i]
            normal = eq[:-1]
            facet_normals.append(normal)
    facet_normals = np.array(facet_normals)
    selected_normals = []
    for normal in facet_normals:
        if len(selected_normals) == 0:
            selected_normals.append(normal)
        else:
            candidate = np.vstack(selected_normals + [normal])
            if np.linalg.matrix_rank(candidate) == len(selected_normals) + 1:
                selected_normals.append(normal)
        #Has to be N vertices not only 2
        if len(selected_normals) == 2:
            break
    #TODO: Has to be N vertices not only 2
    if len(selected_normals) < 2:
        raise ValueError("Could not find 2 linearly independent facet normals.")
    H = np.vstack(selected_normals)
    U = H
    B = (U @ A.T).T
    v_transformed = U @ v
    A_tilde = B - v_transformed
    return A_tilde, U, v

def build_support_set_for_system(d=3):
    A = np.array([
        [2*d, 0],
        [0, 2*d],
        [d,   0],
        [0,   d],
        [1,   0],
        [0,   1],
    ], dtype=float)
    return A

def example_preprocessing(d=3):
    A = build_support_set_for_system(d)
    A_tilde, U, v = preprocess_support_set(A)
    print("Original support set A (exponents):")
    print(A)
    print("\nChosen vertex v:")
    print(v)
    print("\nTransformation matrix U:")
    print(U)
    print("\nPreprocessed support set A_tilde:")
    print(A_tilde)

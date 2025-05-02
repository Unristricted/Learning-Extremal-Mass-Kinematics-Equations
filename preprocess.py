import numpy as np
from scipy.spatial import ConvexHull



def preprocess_support_set(A):
    hull = ConvexHull(A)
    hull_points = A[hull.vertices]
    affine_basis = hull_points - hull_points[0]
    dim = np.linalg.matrix_rank(affine_basis)


    for i in hull.vertices:
         pivot = A[i]
         active_constraints = []
         selected_normals = []
         tolerance = 2**(-17)  # Adjust tolerance if needed
         for j, equation in enumerate(hull.equations):
            if abs(np.dot(equation[:-1], pivot) + equation[-1]) < tolerance:
                active_constraints.append(equation[:-1])    
         
         dimconstraint = np.linalg.matrix_rank(active_constraints)   
         if dimconstraint != dim:
           print('This pivot didnt work')
         else:
             for normal in active_constraints:
                 candidate = np.vstack(selected_normals + [normal])
                 if np.linalg.matrix_rank(candidate) == len(selected_normals) + 1:
                    selected_normals.append(normal)
                 elif len(selected_normals) == dim:
                     break

    H = np.vstack(selected_normals)
    U = -H
    B = (U @ A.T).T
    v_transformed = U @ pivot
    A_tilde = B - v_transformed
    return A_tilde


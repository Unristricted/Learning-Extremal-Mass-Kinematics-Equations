import numpy as np
import numpy as np
from monte_carlo import monte_carlo_kac_rice
from preprocess import preprocess_support_set



#Example 1 from Extremal Real AG by Rojas, Dickenstein, and Rusek 

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
A_C = np.array(
        [
            [1.0, 1.0, s, t, -1, -1],
            [1.0, -1.0, s, -t, -1, 1],
        ]
    )


    
# Randomly generated example
D = np.random.randint(1, 10, size=(20, 5))

mean = 1
standard_deviation = 4

D_C = np.random.normal(loc=mean, scale=standard_deviation, size=(5, 20))


 

    
    
    
delta = 0.0001
 
final_estimate = np.arange(7)
for i in range(7):
      est = monte_carlo_kac_rice(A, A_C, delta, n_samples=100, M=100)
      final_estimate[i] = est
    
print(f"My five experiments gave me:{final_estimate}")
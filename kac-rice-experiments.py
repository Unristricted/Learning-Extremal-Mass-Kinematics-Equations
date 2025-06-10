import numpy as np
from hmc_montecarlo import hmc_monte_carlo_kac_rice
from preprocess import preprocess_support_set
import jax


class MonteCarloExperiment:
    def __init__(self, A, A_C, delta, n_samples=100, M=100):
        self.A = A
        self.A_C = A_C
        self.delta = delta
        self.n_samples = n_samples
        self.M = M


    def run_hmc_mc(self, num_experiments, seed=123):
        self.estimates = np.arange(num_experiments)
        hmc_params = {'rng_key': jax.random.PRNGKey(123)}
        for i in range(num_experiments):
            est = hmc_monte_carlo_kac_rice(self.A, self.A_C, self.delta, n_samples=self.n_samples, M=self.M, hmc_params=hmc_params)
            self.estimates[i] = est

        print(f'My {num_experiments} experiment(s) gave me:{self.estimates}')
        return self.estimates


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

def main():
    Extremal_Real_AG_by_Rojas = MonteCarloExperiment(A, A_C, delta)
    exp1_results = Extremal_Real_AG_by_Rojas.run_hmc_mc(10)

    Randomly_generated_example = MonteCarloExperiment(D, D_C, delta)
    exp2_results = Randomly_generated_example.run_hmc_mc(10)

    print(exp1_results)
    print(exp2_results)

if __name__ == "__main__":
    main()

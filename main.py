
import numpy as np
from deepga import TruncatedRealMutatorGA

def objective(x):

  result = -np.sum(x**2)

  final_result = np.empty(1, dtype=np.float32)
  final_result[0] = result
  return final_result


a = TruncatedRealMutatorGA(objective=objective,
                           initial_guess=np.array([1, 1, 1, 1, 1, 1]),
                           workers_per_rank=2,
                           num_elite=2,
                           num_parents=2,
                           verbose=True,
                           seed=2
                           )

a(10, save=2)

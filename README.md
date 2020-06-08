# deep-neuroevolution-mpi4py
Deep Neuroevolution paper implemented using mpi4py

Status: Maintenance (expect bug fixes and minor updates)

Implementation of Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning (https://arxiv.org/abs/1712.06567)

How to install?

python3 -m pip install -e .

import deep_neuroevolution


How to use?

mpiexec -n 4 python3 main.py

    optimizer = TruncatedRealMutatorGA(objective, initial_guess, workers_per_rank, num_parents, num_elite, mutation_rate, save)

    # It is going to save the results on the folder results/experiment
    optimizer(steps=10)
    
python3 results.py
    
    



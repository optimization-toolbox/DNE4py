# DNE4py: Deep-Neuroevolution with mpi4py

Status: Maintenance (expect bug fixes and major updates)

DNE4py is a python library that aims to run and visualize many different evolutionary algorithms with high performance using mpi4py. It allows easy evaluation of evolutionary algorithms in high dimension (e.g. neural networks for reinforcement learning) 

Implementation available:

* [Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/pdf/1712.06567.pdf)

## Installation

```console
foo@bar:~$ git clone https://github.com/optimization-toolbox/DNE4py
foo@bar:~$ cd deep-neuroevolution-mpi4py/
foo@bar:~$ python3 -m pip install -e .
```

## Running

Create main.py:

```python
from DNE4py.optimizers.deepga import TruncatedRealMutatorGA

def objective(x):
    result = np.sum(x**2)
    return result

initial_guess = [1.0, 1.0]

optimizer = TruncatedRealMutatorGA(objective=objective,
                                   initial_guess=initial_guess,
                                   workers_per_rank=10,
                                   num_elite=1,
                                   num_parents=3,
                                   sigma=0.01,
                                   seed=100,
                                   save=1,
                                   verbose=1,
                                   output_folder='results/experiment')

optimizer(100)
```

Execute main.py (relies on MPI):

```console
foo@bar:~$ mpiexec -n 4 python3 main.py
```

This will create a result folder based on output_folder

##### DeepGA

![DeepGA: population over generations](https://github.com/optimization-toolbox/DNE4py/blob/master/tutorials/2_postprocessing_the_results/gif/deepga_truncatedrealmutatorga.gif)



##### CMA-ES

![CMA-ES: population over generations](https://github.com/optimization-toolbox/DNE4py/blob/master/tutorials/2_postprocessing_the_results/gif/cmaes.gif)

##### RandomSearch

![RandomSearch: population over generations](https://github.com/optimization-toolbox/DNE4py/blob/master/tutorials/2_postprocessing_the_results/gif/randomsearch.gif)

## Post-processing

You can import and generate some visualizations:
```python
from DNE4py import load_mpidata, get_best_phenotype, get_best_phenotype_generator
```




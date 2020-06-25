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

## Post-processing

You can import from utils and generate some visualizations:
```python
from DNE4py.utils import plot_cost_over_generation, plot_best_cost_over_generation, render_population_over_generation
```

##### plot\_cost\_over\_generation

![cost over generation](https://github.com/optimization-toolbox/DNE4py/blob/master/DNE4py/examples/images/deepga_example1.png)

##### plot\_best\_cost\_over\_generation

![best cost over generation](https://github.com/optimization-toolbox/DNE4py/blob/master/DNE4py/examples/images/deepga_example2.png)

##### render\_population\_over\_generation

![render population over generation](https://github.com/optimization-toolbox/DNE4py/blob/master/DNE4py/examples/images/deepga_example3.gif)

## Documentation
TODO






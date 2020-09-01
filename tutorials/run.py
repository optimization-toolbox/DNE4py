import yaml
import numpy as np
from DNE4py import load_optimizer


def objective_function(x):
    return np.sum(x**2)


if "__main__" == __name__:

    # Read config:
    with open(f'TruncatedRealMutatorGA.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    optimizer_config = config.get('optimizer')

    # Declare Optimizer:
    optimizer_config['initial_guess'] = np.array([-0.3, 0.7])
    optimizer = load_optimizer(optimizer_config)

    # Run Optimizer:
    optimizer.run(objective_function, 20)

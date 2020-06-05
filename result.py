import pickle
import numpy as np

def join_mpidata(name, nb_workers, nb_generations):

    path = f'results/experiment/{name}'

    #Load:

    full_data = [[]] * nb_generations
    for w in range(nb_workers):
        with open(f'{path}_w{w}.pkl', 'rb') as f:
            for g in range(nb_generations):
                # Load data:
                data = pickle.load(f)
                worker_data = list(data.values())[0]
                # Append data:
                full_data[g] = full_data[g] + worker_data
    full_data = np.array(full_data)

    #Save:
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(full_data, f)

join_mpidata('genealogy', 3, 5)

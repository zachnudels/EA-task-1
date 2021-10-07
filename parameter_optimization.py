import multiprocessing
import os
import platform
from pathlib import Path
from datetime import datetime
from hyperopt import hp, fmin, tpe
import numpy as np
from sklearn.model_selection import ParameterGrid
import neat
from hyperopt.pyll.stochastic import sample
from functools import partial

from generalist_experiment import run_experiment


def objective(args, method, cpus, group, generations, run_path):
    local_dir = os.path.dirname('evoman')
    weight_mutate_rate = args['weight_mutate_rate']
    conn_add_prob = args['conn_add_prob']
    conn_delete_prob = args['conn_delete_prob']
    node_add_prob = args['node_add_prob']
    node_delete_prob = args['node_delete_prob']
    compatibility_threshold = args['compatibility_threshold']
        # = args
        # compatibility_disjoint_coefficient, compatibility_weight_coefficient \

    config_path = os.path.join(local_dir, f"{method}.cfg")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path
                         )
    config.genome_config.weight_mutate_rate = weight_mutate_rate
    config.genome_config.conn_add_prob = conn_add_prob
    config.genome_config.conn_delete_prob = conn_delete_prob
    config.genome_config.node_add_prob = node_add_prob
    config.genome_config.node_delete_prob = node_delete_prob
    config.species_set_config.compatibility_threshold = compatibility_threshold
    # config.genome_config.compatibility_disjoint_coefficient = compatibility_disjoint_coefficient
    # config.genome_config.compatibility_weight_coefficient = compatibility_weight_coefficient


    config_path = Path(f"configs/{method}")
    if not config_path.exists():
        config_path.mkdir(parents=True, exist_ok=True)
    config.save(filename=os.path.join(config_path, f"{method}_{datetime.now().strftime('%Y-%M-%d-%H-%M%S%f')}.cfg"))

    _, _, _, best, _, winner = run_experiment(method, cpus, generations, run_path, group, config)
    obj = 100 - best.fitness
    if best.fitness != winner.fitness:
        obj += 10
        
    return obj

# def coarse_search_space():
#     weight_mutate_rates = np.linspace(0.1, 1, 5)
#     conn_add_prob = np.linspace(0, 0.5, 5)
#     conn_delete_prob = np.linspace(0, 0.5, 5)
#     node_add_prob = np.linspace(0, 0.5, 5)
#     node_delete_prob = np.linspace(0, 0.5, 5)
#     # compatibility_disjoint_coefficient = np.linspace(0, 1, 10)
#     # compatibility_weight_coefficient = np.linspace(0, 1, 10)
#     compatibility_threshold = np.linspace(1.5, 4, 6)
#     params = ParameterGrid({'weight_mutate_rates': weight_mutate_rates,
#                            'conn_add_prob': conn_add_prob,
#                             'conn_delete_prob': conn_delete_prob,
#                             'node_add_prob': node_add_prob,
#                             'node_delete_prob': node_delete_prob,
#                             'compatibility_threshold': compatibility_threshold
#                             })
#     print(len(params))
#     return params


def search_space():
    space = {
        'weight_mutate_rate': hp.uniform("weight", 0.1, 0.9),
        'conn_add_prob': hp.uniform("conn_add", 0, 0.5),
        'conn_delete_prob': hp.uniform("conn_del", 0, 0.5),
        'node_add_prob': hp.uniform("node_add", 0, 0.5),
        'node_delete_prob': hp.uniform("node_del", 0, 0.5),
        # 'compatibility_disjoint_coefficient': ,
        # 'compatibility_weight_coefficient': ,
        'compatibility_threshold': hp.uniform("comp_thresh", 1, 4),
    }
    return space


def optimize(method, generations, cpus, group):
    return fmin(partial(objective, method=method, generations=generations, cpus=cpus, group=group, run_path='/tmp'),
                search_space(), algo=tpe.suggest, max_evals=100)


if __name__ == '__main__':
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')  # Comment this if not on MACOS

    # space = search_space()
    # print(objective("FS_NEAT", 2, 2, 2, "/tmp", sample(space)))
    optimize("FS_NEAT", 2, 2, [1, 2, 4, 5])

    # print(sample(space))

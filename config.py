SIMULATED_ANNEALING = {
    'SearchSpace': {
        'multiple': True,
        'range': [100, 500, 1000, 5000],
        'num_households': 150,
        },
    'Annealing': {
        'delta_temp': 500,
        'energy_factor': 0.85,
    },
    'Experiment': {
        'iterations': [10, 50, 200, 500, 1000, 5000, 10000]
    }
}

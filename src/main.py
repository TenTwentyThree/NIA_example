import numpy as np
import pandas as pd
import logging
from data import generate_searchspace as generator
from src import optimizers, visualize_results


def execute():
    logging.getLogger('Main').setLevel(logging.INFO)
    annealer = optimizers.SimulatedAnnealing()
    for households in [100,500,1000,5000]:
        generator.model_power_distribution(num_households=households,save=True,save_prefix=f'{households}_')
        annealer._set_new_search_space(prefix=f'{households}_')
        logging.info(f'Maximum score: {annealer.max_inhabitants}')
        history = [annealer.delta_anneal(delta_temp=item,cold_start=True,label=item)[0] for item in [10,
                                                                                                     50,
                                                                                                     200,
                                                                                                     500,
                                                                                                     1000,
                                                                                                     5000,
                                                                                                     10000]]
        visualize_results.visualize_total_score(pd.concat(history), households)

if __name__ == '__main__':
    execute()

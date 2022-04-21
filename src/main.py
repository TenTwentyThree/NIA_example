from config import SIMULATED_ANNEALING as cnfg
import pandas as pd
import logging
from data import generate_searchspace as generator
from src import optimizers, visualize_results


def execute():
    logging.getLogger('Main').setLevel(logging.INFO)
    annealer = optimizers.SimulatedAnnealing()
    if cnfg['SearchSpace']['multiple']:
        household_range = cnfg['SearchSpace']['range']
    else:
        household_range = [cnfg['SearchSpace']['num_households']]
    for households in household_range:
        generator.model_power_distribution(num_households=households, save=True, save_prefix=f'{households}_')
        annealer._set_new_search_space(prefix=f'{households}_')
        logging.info(f'Maximum score: {annealer.max_inhabitants}')
        history = [annealer.delta_anneal(delta_temp=item, cold_start=True, label=item)[0]
                   for item in cnfg['Experiment']['iterations']]
        visualize_results.visualize_total_score(pd.concat(history), households)


if __name__ == '__main__':
    execute()  # depending on your os, you may need to call this function from python console to make it work

import numpy as np
import os
import logging
import pandas as pd

class SimulatedAnnealing:
    def __init__(self, delta_temp = None, energy_availability_factor = None):
        logging.getLogger('Annealer').setLevel('INFO')
        if not delta_temp:
            logging.info('Variables not found in init call, loading from config file')
            from config import SIMULATED_ANNEALING as cnfg
            self.delta_temp = cnfg['Annealing']['delta_temp']
            self.energy_availability_factor = cnfg['Annealing']['energy_factor']
        else:
            self.delta_temp = delta_temp
            self.energy_availability_factor = energy_availability_factor
        self.households, self.consumption = self._load_search_space()
        self.num_households = len(self.households)
        self.max_inhabitants = self.households.sum()
        self.available_energy = np.round(self.consumption * self.energy_availability_factor).astype('int').sum()
        logging.info(f'Annealer loaded successfully with delta temp {self.delta_temp}, {self.max_inhabitants} '
                     f'inhabitants and energy factor {self.energy_availability_factor}, happy annealing :)')

    def _load_search_space(self):
        path = os.path.join(os.path.abspath('.'),'data')
        households = np.load(os.path.join(path,'households.npy'))
        consumption = np.load(os.path.join(path,'household_consumption.npy'))
        return households, consumption

    def _total_score(self, candidate_solution):
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        return self.households[indices].sum(), self.consumption[indices].sum()

    def score(self, candidate_solution):
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        if np.sum(self.consumption[indices]) > self.available_energy:
            return 0
        else:
            return np.sum(self.households[indices]) / self.max_inhabitants

    def select_state(self, s1, s2, temp):
        if self.score(s2) > self.score(s1):
            return s2
        elif self.score(s1) >= self.score(s2) and self.score(s2) != 0:
            if np.random.rand() < temp:
                return s2
        return s1

    def generate_neighbor(self, candidate):
        change_index = np.random.randint(0,len(candidate))
        neighbor = candidate.copy()
        if neighbor[change_index] == 1:
            neighbor[change_index] = 0
        else:
            neighbor[change_index] = 1
        return neighbor

    def delta_anneal(self, delta_temp = 1, label=0):
        history = []
        if delta_temp != 1:
            annealing_schedule = delta_temp
        else:
            annealing_schedule = self.delta_temp
        candidate = np.random.choice([1,0],self.num_households)
        for n in range(annealing_schedule):
            temp = 1 - (n+1) / delta_temp
            neighbor = self.generate_neighbor(candidate)
            candidate = self.select_state(candidate, neighbor, temp)
            history.append({'Label': label,
                            'Iteration': n, 'Relative Score': self.score(candidate),
                            'Temperature': temp,
                            'Total Score': self._total_score(candidate)[0],
                            'Max score': self.max_inhabitants,
                            'Energy utilized': np.sum(self.consumption[candidate]),
                            'Energy remaining':self.available_energy - self._total_score(candidate)[1],
                            }
                           )
        return pd.DataFrame(history)

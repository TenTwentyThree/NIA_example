import numpy as np
import os
import logging
import pandas as pd

class SimulatedAnnealing:
    def __init__(self, delta_temp = 500):
        self.delta_temp = delta_temp
        self.households, self.consumption = self._load_search_space()
        self.num_households = len(self.households)
        self.max_inhabitants = self.households.sum()
        self.energy_availability_factor = 0.75
        self.available_energy = np.round(self.consumption * self.energy_availability_factor).astype('int')

    def _load_search_space(self):
        path = os.path.join(os.path.abspath('.'),'data')
        households = np.load(os.path.join(path),'household.npy')
        consumption = np.load(os.path.join(path),'household_consumption.npy')
        return households, consumption

    def _total_score(self, candidate_solution):
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        return self.households[indices].sum()

    def score(self, candidate_solution):
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        if self.consumption[indices] > self.available_energy:
            return 0
        else:
            return self.households[indices].sum() / self.max_inhabitants

    def select_state(self, s1, s2, temp):
        if self.score(s2) > self.score(s1):
            return s2
        elif self.score(s1) >= self.score(s2):
            if np.random.rand() < temp:
                return s2
        else:
            return s1

    def generate_neighbor(self, candidate):
        change_index = np.random.randint(0,len(candidate))
        neighbor = candidate.copy()
        if neighbor[change_index] == 1:
            neighbor[change_index] = 0
        else:
            neighbor[change_index] = 1
        return neighbor

    def delta_anneal(self, delta_temp = 1):
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
            history.append({'Iteration': n, 'Relative Score': self.score(candidate),
                            'Total Score': self._total_score(candidate),
                            'Max score': self.max_inhabitants})
        return pd.DataFrame(history)










import numpy as np
import os
import logging
import pandas as pd


class SimulatedAnnealing:
    def __init__(self, delta_temp=None, energy_availability_factor=None, load_prefix=''):
        logging.getLogger('Annealer').setLevel('INFO')
        if not delta_temp:
            logging.info('Variables not found in init call, loading from config file')
            from config import SIMULATED_ANNEALING as cnfg
            self.delta_temp = cnfg['Annealing']['delta_temp']
            self.energy_availability_factor = cnfg['Annealing']['energy_factor']
        else:
            self.delta_temp = delta_temp
            self.energy_availability_factor = energy_availability_factor
        self._set_new_search_space(prefix=load_prefix)
        logging.info(f'Annealer loaded successfully with delta temp {self.delta_temp}, {self.max_inhabitants} '
                     f'inhabitants and energy factor {self.energy_availability_factor}, happy annealing :)')

    def _set_new_search_space(self, prefix):
        """
        Loads search space parameters from file and sets them as object attributes
        :param prefix: filename
        :return: None
        """
        path = os.path.join(os.path.abspath('.'), 'data')
        self.households = np.load(os.path.join(path, prefix + 'households.npy'))
        self.consumption = np.load(os.path.join(path, prefix + 'household_consumption.npy'))
        self.max_inhabitants = self.households.sum()
        self.available_energy = np.round(self.consumption * self.energy_availability_factor).astype('int').sum()
        self.num_households = len(self.households)

    def _total_score(self, candidate_solution):
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        return self.households[indices].sum(), self.consumption[indices].sum()

    def _get_household_types(self):
        return pd.Series(self.households).value_counts()

    def _count_household_types(self, candidate_solution):
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        return pd.Series(self.households[indices]).value_counts()

    def score(self, candidate_solution):
        """
        Computes the score of a candidate solution
        :param candidate_solution: binary array representing a candidate solution
        :return: score relative to maximum
        """
        indices = [n for n, i in enumerate(candidate_solution) if i == 1]
        if np.sum(self.consumption[indices]) > self.available_energy:
            return 0
        else:
            return np.sum(self.households[indices]) / self.max_inhabitants

    def select_state(self, s1, s2, temp):
        if self.score(s2) > self.score(s1):
            return s2
        elif self.score(s1) >= self.score(s2) != 0:
            if np.random.rand() < temp:
                return s2
        return s1

    def generate_neighbor(self, candidate):
        """
        Randomly generates a neighboring state to an existing state
        :param candidate: solution candidate binary array
        :return: neighboring state binary array
        """
        change_index = np.random.randint(0, len(candidate))
        neighbor = candidate.copy()
        if neighbor[change_index] == 1:
            neighbor[change_index] = 0
        else:
            neighbor[change_index] = 1
        return neighbor

    def delta_anneal(self, cold_start=True, delta_temp=1, label=0):
        """
        Perform simulated annealing
        :param cold_start: Whether the initial state starts with only 0s selected
        :param delta_temp: steps the annealing process will take
        :param label: label for dumping statistics of the annealing process into a pandas Dataframe
        :return: 2 pandas dataframe objects that contain the results of the annealing process at every step
        """
        history = []
        household_assign = []
        if delta_temp != 1:
            annealing_schedule = delta_temp
        else:
            annealing_schedule = self.delta_temp
        if cold_start:
            candidate = np.repeat(0, self.num_households)
        else:
            candidate = np.random.choice([1, 0], self.num_households)
        for n in range(annealing_schedule):
            temp = 1 - (n+1) / delta_temp
            neighbor = self.generate_neighbor(candidate)
            candidate = self.select_state(candidate, neighbor, temp)

            # record results for visualization
            history.append({'Label': label,
                            'Iteration': n, 'Relative Score': self.score(candidate),
                            'Temperature': temp + annealing_schedule,
                            'Total Score': self._total_score(candidate)[0],
                            'Max score': self.max_inhabitants,
                            'Energy utilized': np.sum(self.consumption[candidate]),
                            'Energy remaining': self.available_energy - self._total_score(candidate)[1],
                            }
                           )
            household_assign.append(self._count_household_types(candidate))
        assign_frame = pd.DataFrame(household_assign)
        assign_frame = assign_frame[assign_frame.columns.sort_values()]
        assign_frame = assign_frame.fillna(0).astype('int')
        return pd.DataFrame(history), assign_frame

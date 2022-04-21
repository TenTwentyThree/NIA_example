import numpy as np
import os.path as path
from scipy import stats


def model_power_distribution(num_households=150, sigma=1.45, save=False, save_prefix=''):
    """
    Generates a search space for the simulated annealing example: Power distribution of renewable energies
    over a number of households with a different number of residents.
    A household with multiple residents experiences a total energy discount (diminishing returns)
    :param num_households: number of households to be simulated
    :param sigma: smoothness of distribution to sample for to determine residents per household.
    :param save: whether to save to file
    :param save_prefix: file prefix
    :return: household array and energy consumption array
    """
    average_persons_household = 1.99  # on average, 1.99 people live in one household in germany (2019)
    average_kwh_consumption = 3561  # average electricity consumption per person is 3561 kw/h annually (2020)
    resident_norm = stats.truncnorm(a=0, b=8, loc=average_persons_household, scale=sigma)  # Min: 0, Max: 8 residents
    consumption_norm = stats.norm(loc=average_kwh_consumption)
    household_residents = resident_norm.rvs(num_households)
    household_residents = household_residents.astype('int')
    e_consumption = np.zeros(num_households)  # place-holder array for number of households to model
    for n, index in enumerate(household_residents):
        if index == 1:
            e_consumption[n] = np.round(consumption_norm.rvs(1))  # No discount applied
        elif index == 0:  # this is modelled just in case.
            e_consumption[n] = 0
        else:
            dc = np.arange(2, index)
            average_consumption = consumption_norm.rvs(index).mean()
            if len(dc) <= 1:
                e_consumption[n] = (average_consumption * 0.75 * index)  # roughly modelled after real-life data
            else:
                e_consumption[n] = np.round(average_consumption * np.prod(1 - (1 / (dc ** dc))))
    e_consumption = e_consumption.astype('int')
    if save:
        np.save(path.join(path.abspath('.'), 'data', save_prefix+'households.npy'), arr=household_residents)
        np.save(path.join(path.abspath('.'), 'data', save_prefix+'household_consumption.npy'), arr=e_consumption)
    return household_residents, e_consumption

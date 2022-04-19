import numpy as np
import os.path as path
from scipy import stats


def generate_searchspace(search_space_dim = 500, num_peaks = 3):
    """
    pull this into a practical domain
    :param search_space_dim:
    :param num_peaks:
    :return:
    """
    scale_peaks_x = np.append(np.sort(np.random.randint(1, 5, num_peaks)), 0)
    local_optima_indices_x = np.append(np.sort(np.random.randint(0, search_space_dim, num_peaks)), search_space_dim)
    scale_peaks_y = np.append(np.sort(np.random.randint(1, 5, num_peaks)), 0)
    local_optima_indices_y = np.append(np.sort(np.random.randint(0, search_space_dim, num_peaks)), search_space_dim)
    for index_x in range(len(local_optima_indices_x)):
        if index_x == 0:
            x = np.linspace(0, scale_peaks_x[index_x] * np.pi, local_optima_indices_x[0])
        elif index_x != len(local_optima_indices_x) - 1:
            x = np.append(x, np.linspace(scale_peaks_x[index_x - 1] * np.pi, scale_peaks_x[index_x] * np.pi,
                                                   local_optima_indices_x[index_x] - local_optima_indices_x[index_x - 1]))
        else:
            x = np.append(x, np.linspace(scale_peaks_x[index_x - 1] * np.pi, 0,
                                                   search_space_dim - local_optima_indices_x[index_x - 1]))

    for index_y in range(len(local_optima_indices_y)):
        if index_y == 0:
            y = np.linspace(0, scale_peaks_y[index_y] * np.pi, local_optima_indices_y[0])
        elif index_y != len(local_optima_indices_y) - 1:
            y = np.append(y, np.linspace(scale_peaks_y[index_y - 1] * np.pi, scale_peaks_y[index_y] * np.pi,
                                                   local_optima_indices_y[index_y] - local_optima_indices_y[index_y - 1]))
        else:
            y = np.append(y, np.linspace(scale_peaks_y[index_y - 1] * np.pi, 0,
                                                   search_space_dim - local_optima_indices_y[index_y - 1]))

    z = np.sin(y).reshape((search_space_dim, 1)) * np.sin(x).reshape((1, search_space_dim))
    return x, y, z

def model_power_distribution(num_households=150, gamma=1.45, save=False):
    average_persons_household = 1.99
    average_kwh_consumption = 3561
    inhab_norm = stats.truncnorm(a=0,b=8,loc=average_persons_household,scale=gamma)
    consumption_norm = stats.norm(loc=average_kwh_consumption)
    household_inhabitants = inhab_norm.rvs(num_households)
    household_inhabitants = household_inhabitants.astype('int')
    l = np.zeros(num_households)
    for n, index in enumerate(household_inhabitants):
        if index == 1:
            l[n] = np.round(consumption_norm.rvs(1))
        else:
            dc = np.arange(2, index)
            average_consumption = consumption_norm.rvs(index).mean()
            if len(dc) <= 1:
                l[n] = (average_consumption * 0.75 * index)
            else:
                l[n] = np.round(average_consumption * np.prod(1 - (1 / (dc ** dc))))
    l = l.astype('int')
    if save:
        np.save(path.join(path.abspath('.'),'data','households.npy'),arr=household_inhabitants)
        np.save(path.join(path.abspath('.'), 'data', 'household_consumption.npy'), arr=l)
    return household_inhabitants, l
















import matplotlib.pyplot as plt
import numpy as np


def visualize_search_space(x,y):
    """
    Visualizes the simple, 2D search space. Not suitable for more complex problems
    :param x: Z-values at coordinate X
    :param y: Z-values at coordinate >
    :return: None
    """
    xs, ys = np.meshgrid(x, y, sparse=True)
    zs = np.sqrt(xs ** 2 + ys ** 2)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xs, ys, zs, cmap='viridis', edgecolor='none')
    plt.show()

def visualize_path(dataframe):
    """
    Visualizes a hill climbing path within a given search space.
    :param dataframe: pandas dataframe object containing X and Y coordinates for each iteration.
    :return: None
    """
    x = np.load('./data/x.npy')
    y = np.load('./data/y.npy')
    xs, ys = np.meshgrid(x, y, sparse=True)
    zs = np.sqrt(xs ** 2 + ys ** 2)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x_values = x[dataframe['X'].values]
    y_values = y[dataframe['Y'].values]
    z_values = zs[dataframe['X'], dataframe['Y']]
    ax.plot(x_values, y_values, z_values, color='red')
    ax.plot_surface(xs, ys, zs, cmap='viridis', edgecolor='none', color='grey',alpha=0.025)
    plt.show()

def visualize_iterations(dataframes):
    pass

import matplotlib.pyplot as plt
import numpy as np


def visualize_homes():
    pass

def visualize_total_score(results_frame):
    frame_indices = results_frame['Label'].unique()
    frame_lengths = results_frame['Label'].value_counts()
    longest_frame = frame_lengths.iloc[0]
    x = np.arange(longest_frame)
    #y = np.arange(results_frame['Total Score'].max())
    fig, ax = plt.subplots(figsize=(8,15))
    ax.set_ylim(results_frame['Total Score'].values.min()-10,results_frame['Max score'].values.max())
    for index in frame_indices:
        sub_frame = results_frame.loc[results_frame['Label'] == index]
        x_val = sub_frame['Total Score'].values
        x_val = np.append(x_val,np.repeat(x_val[-1],longest_frame - len(x_val)))
        ax.plot(x, x_val,label=str(index))
    plt.legend()
    plt.tight_layout()
    plt.show()


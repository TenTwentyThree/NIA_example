import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_homes(household_series, household_assign):
    max_values = household_series.sort_index().values
    percentage = household_assign / max_values
    values = percentage.values
    fig, ax = plt.subplots(figsize=(35, 2))
    ax.matshow(values.transpose())
    plt.tight_layout()
    plt.show()


def visualize_total_score(results_frame, num_households):
    frame_indices = results_frame['Label'].unique()
    frame_lengths = results_frame['Label'].value_counts()
    longest_frame = frame_lengths.iloc[0]
    x = np.arange(longest_frame)
    #y = np.arange(results_frame['Total Score'].max())
    fig, ax = plt.subplots(figsize=(8, 15))
    #ax.set_ylim(results_frame['Total Score'].values.min()-10, results_frame['Max score'].values.max())

    for index in frame_indices:
        sub_frame = results_frame.loc[results_frame['Label'] == index]
        x_val = sub_frame['Total Score'].values
        x_val = np.append(x_val, np.repeat(x_val[-1], longest_frame - len(x_val)))
        ax.plot(x, x_val, label=str(index))
    plt.title(f'Energy distribution for {results_frame["Total Score"].max()} people')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.abspath('.'),'results',
                             f'{num_households}_{results_frame["Total Score"].max()}_results.png'),dpi=300)
    plt.show()
    plt.clf()

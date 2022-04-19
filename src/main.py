from src import optimizers, visualize_results

def main():
    hc = optimizer.HillClimbing()
    history = hc.hill_climb_n_times(10)
    for i in history:
        visualize_results.visualize_path(i)
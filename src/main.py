from src import optimizers, visualize_results

def main():
    history = []
    annealer = optimizers.SimulatedAnnealing()
    history = [annealer.delta_anneal(n) for n in [10,50,200,500,1000,5000,10000]]

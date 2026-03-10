import numpy as np
import pickle
import os
from multiprocessing import Pool
from model import NeuralNetwork
from evolution import EvolutionEngine
from env_utils import evaluate_network
from config import POP_SIZE, GENERATIONS, SAVE_BEST, INITIAL_MUTATION_RATE

NUM_WORKERS = 6

def evaluate_one(args):
    net, worker_id = args
    window_name = f"Mario Agent {worker_id + 1}"
    return evaluate_network(net, render=True, window_name=window_name)

if __name__ == "__main__":
    model_path = "models/best_network.pkl"

    # Seed population from saved model if it exists, else start random
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            best_saved = pickle.load(f)
        print(f"✓ Loaded saved model from {model_path}")
        print(f"  Seeding population of {POP_SIZE} from it with small mutations...\n")

        # Make a population of mutated clones of the best saved network
        population = []
        for _ in range(POP_SIZE):
            clone = best_saved.clone()
            clone.W1 += INITIAL_MUTATION_RATE * np.random.randn(*clone.W1.shape)
            clone.W2 += INITIAL_MUTATION_RATE * np.random.randn(*clone.W2.shape)
            population.append(clone)
    else:
        print("No saved model found — starting from scratch with random networks.\n")
        population = [NeuralNetwork() for _ in range(POP_SIZE)]

    engine = EvolutionEngine(population)
    best_global = None
    best_score_global = -np.inf

    print(f"Training for {GENERATIONS} generations with {NUM_WORKERS} parallel windows...\n")

    for gen in range(GENERATIONS):
        tasks = [(net, i % NUM_WORKERS) for i, net in enumerate(population)]

        with Pool(NUM_WORKERS) as pool:
            scores = pool.map(evaluate_one, tasks)

        fitness_scores = list(zip(population, scores))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_score = fitness_scores[0][1]

        if best_score > best_score_global:
            best_score_global = best_score
            best_global = fitness_scores[0][0]
            if SAVE_BEST:
                with open(model_path, "wb") as f:
                    pickle.dump(best_global, f)
                print(f"  ✓ New best saved! Fitness: {best_score_global:.2f}")

        population = engine.next_generation(fitness_scores)
        print(f"Gen {gen:>3} | Best Fitness: {best_score:.2f} | Mutation: {engine.mutation_rate:.5f}")

    print(f"\nTraining complete. Best fitness: {best_score_global:.2f}")
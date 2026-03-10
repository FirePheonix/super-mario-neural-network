import random
import numpy as np
from config import ELITE_SIZE, INITIAL_MUTATION_RATE, DIVERSITY_KEEP

class EvolutionEngine:
    def __init__(self, population):
        self.population = population
        self.mutation_rate = INITIAL_MUTATION_RATE
        self.best_fitness = -np.inf
        self.stagnation = 0

    def crossover(self, p1, p2):
        child = p1.clone()
        mask1 = np.random.rand(*p1.W1.shape) < 0.5
        mask2 = np.random.rand(*p1.W2.shape) < 0.5
        child.W1 = np.where(mask1, p1.W1, p2.W1)
        child.W2 = np.where(mask2, p1.W2, p2.W2)
        return child

    def mutate(self, net):
        net.W1 += self.mutation_rate * np.random.randn(*net.W1.shape)
        net.W2 += self.mutation_rate * np.random.randn(*net.W2.shape)
        return net

    def adapt_mutation(self, best_score):
        if best_score > self.best_fitness:
            self.best_fitness = best_score
            self.stagnation = 0
            self.mutation_rate *= 0.9
        else:
            self.stagnation += 1
            if self.stagnation > 4:
                self.mutation_rate *= 1.3
                self.stagnation = 0
        # Clamp mutation rate to a safe range
        self.mutation_rate = max(0.005, min(self.mutation_rate, 0.05))

    def tournament_select(self, fitness_scores, k=3):
        """Pick k random agents, return the best one. More diverse than pure elitism."""
        contestants = random.sample(fitness_scores, min(k, len(fitness_scores)))
        return max(contestants, key=lambda x: x[1])[0]

    def next_generation(self, fitness_scores):
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_score = fitness_scores[0][1]
        self.adapt_mutation(best_score)

        # Always keep the best performers unchanged (elitism)
        elites = [net.clone() for net, _ in fitness_scores[:ELITE_SIZE]]
        new_pop = elites

        # Add a few random diversity agents from anywhere in the population
        for _ in range(DIVERSITY_KEEP):
            rand_net = random.choice(fitness_scores)[0]
            new_pop.append(rand_net.clone())

        # Fill the rest via tournament selection — FAR more diverse than pure top-6 breeding
        while len(new_pop) < len(self.population):
            p1 = self.tournament_select(fitness_scores, k=3)
            p2 = self.tournament_select(fitness_scores, k=3)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_pop.append(child)

        return new_pop
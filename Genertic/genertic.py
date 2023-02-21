from collections import deque
import random
import numpy as np
from torch import nn
import torch
from torch.functional import F

from project_model import Model
           
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def initialize_population(self, num_params):
        # Initialize population
        self.population = np.random.randn(self.population_size, num_params)

    def select_parents(self, fitness_scores):
        # Select parents using tournament selection
        tournament_size = 2
        num_parents = self.population_size // 2
        parents = np.empty((num_parents, self.population[0].shape[0]))
        for i in range(num_parents):
            idx = np.random.randint(0, self.population_size, size=tournament_size)
            tournament_scores = fitness_scores[idx]
            winner_idx = idx[np.argmax(tournament_scores)]
            parents[i] = self.population[winner_idx]
        return parents
    
    def crossover(self, parents):
        # Perform uniform crossover
        children = np.empty_like(parents)
        for i in range(0, parents.shape[0], 2):
            parent1, parent2 = parents[i], parents[i+1]
            mask = np.random.randint(0, 2, size=parent1.shape).astype(np.bool)
            children[i][mask] = parent1[mask]
            children[i][~mask] = parent2[~mask]
            children[i+1][mask] = parent2[mask]
            children[i+1][~mask] = parent1[~mask]
        return children
    
    def mutate(self, population):
        # Perform Gaussian mutation
        num_mutations = int(self.mutation_rate * population.size)
        mutations = np.random.randn(num_mutations)
        mutation_indices = np.random.randint(0, population.size, size=num_mutations)
        population = population.flatten()
        population[mutation_indices] += mutations
        population = population.reshape(-1, mutations.shape[0])
        return population
           
    def compute_fitness_scores(self, q_network, env, experiences, epsilon):
        # Compute fitness scores for the population
        fitness_scores = np.empty(self.population_size)
        for i, weights in enumerate(self.population):
            # Load weights into Q-network
            q_network.load_state_dict({'fc1.weight': torch.tensor(weights[:512].reshape(32, 4 * 4)),
                                       'fc1.bias': torch.tensor(weights[512:544]),
                                       'fc2.weight': torch.tensor(weights[544:608].reshape(4, 32)),
                                       'fc2.bias': torch.tensor(weights[608:])})
            
            # Evaluate Q-network on experiences
            episode_rewards = []
            for _ in range(experiences):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    # Epsilon-greedy action
                    if np.random.rand() < epsilon:
                        action = env.action_space.sample()
                    else:
                        state_tensor = torch.tensor(state).unsqueeze(0).float()
                        with torch.no_grad():
                            q_values = q_network(state_tensor)
                            action = q_values.argmax().item()
                    
                    # Take action and add experience to replay buffer
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                
                episode_rewards.append(episode_reward)
            
            fitness_scores[i] = np.mean(episode_rewards)
        
        return fitness_scores

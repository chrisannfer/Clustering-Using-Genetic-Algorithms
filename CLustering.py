import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_POINTS = 100  # Number of random data points
NUM_CLUSTERS = 3  # Number of clusters (centroids)
POP_SIZE = 50     # Population size (number of centroid candidates)
MUTATION_RATE = 0.1  # Mutation rate
GENERATIONS = 20  # Number of generations
BIAS_FACTOR = 0.5  # Bias factor for selection and mutation

# Generate random data points
data_points = np.random.rand(NUM_POINTS, 2)

# Initialize centroids randomly
centroids = np.random.rand(NUM_CLUSTERS, 2)

# Function to assign clusters based on the nearest centroid
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Fitness function: Calculate the total distance of points to their nearest centroid
def fitness(centroids, data):
    clusters = assign_clusters(data, centroids)
    distances = np.linalg.norm(data[:, np.newaxis] - centroids[clusters], axis=2)
    return -np.sum(np.min(distances, axis=1))  # We want to maximize fitness (minimize distance)

# Selection: Tournament selection with bias
def selection(population, fitness_scores):
    selected = []
    mean_centroid = np.mean(population, axis=0)
    for _ in range(POP_SIZE):
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        # Introduce bias towards centroids closer to the mean
        bias1 = np.linalg.norm(population[idx1] - mean_centroid)
        bias2 = np.linalg.norm(population[idx2] - mean_centroid)
        if fitness_scores[idx1] + (BIAS_FACTOR / (1 + bias1)) > fitness_scores[idx2] + (BIAS_FACTOR / (1 + bias2)):
            selected.append(population[idx1])
        else:
            selected.append(population[idx2])
    return np.array(selected)

# Crossover: Create new centroids from selected parents with bias
def crossover(selected):
    offspring = []
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected):
            parent1, parent2 = selected[i], selected[i + 1]
            # Create a child by averaging the parents
            child = (parent1 + parent2) / 2
            offspring.append(child)
    return np.array(offspring)

# Mutation: Randomly adjust centroids with bias
def mutate(centroids, fitness_scores):
    for i in range(len(centroids)):
        if np.random.rand() < MUTATION_RATE:
            # Ensure mutation strength is non-negative
            mutation_strength = max(0.01, 0.05 * (1 + fitness_scores[i]))  # Set a minimum mutation strength
            centroids[i] += np.random.normal(0, mutation_strength, size=centroids[i].shape)  # Small random adjustment
    return centroids

# Main loop for the genetic algorithm
def genetic_algorithm(data, initial_centroids, generations):
    population = np.array([initial_centroids + np.random.normal(0, 0.1, size=initial_centroids.shape) for _ in range(POP_SIZE)])

    for generation in range(generations):
        # Calculate fitness scores
        fitness_scores = np.array([fitness(individual, data) for individual in population])

        # Selection
        selected = selection(population, fitness_scores)

        # Crossover
        offspring = crossover(selected)

        # Mutation
        population = mutate(offspring, fitness_scores)

        # Plotting the current generation with centroids
        plt.figure(figsize=(8, 6))
        clusters = assign_clusters(data, population.mean(axis=0))  # Use the mean of the population for plotting
        for k in range(NUM_CLUSTERS):
            plt.scatter(data[clusters == k][:, 0], data[clusters == k][:, 1], label=f'Cluster {k + 1}')
        plt.scatter(population.mean(axis=0)[:, 0], population.mean(axis=0)[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.title(f'Generation {generation + 1}')
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        plt.show()

# Run the genetic algorithm
genetic_algorithm(data_points, centroids, GENERATIONS)

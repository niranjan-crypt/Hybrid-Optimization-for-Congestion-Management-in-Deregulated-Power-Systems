import numpy as np

# Define the objective function (cost function, congestion reduction, etc.)
def objective_function(x):
    # Example: Quadratic cost function for power generation
    return sum(0.01 * x**2 + 5 * x + 100)

# Initialize population
def initialize_population(size, dim, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (size, dim))

# OPA Exploration Step (Orca Predation)
def orca_predation(pop, best, alpha=0.5):
    new_pop = np.copy(pop)
    for i in range(len(pop)):
        R = np.random.uniform(-1, 1)
        new_pop[i] = pop[i] + alpha * R * (best - pop[i])
    return new_pop

# KHA Refinement Step (Krill Herd)
def krill_herd_movement(pop, best, beta=0.1):
    new_pop = np.copy(pop)
    for i in range(len(pop)):
        random_index = np.random.randint(len(pop))
        new_pop[i] += beta * (pop[random_index] - pop[i]) + beta * (best - pop[i])
    return new_pop

# SHO Fine-tuning Step (Spotted Hyena)
def spotted_hyena_optimization(pop, best, gamma=0.3):
    new_pop = np.copy(pop)
    for i in range(len(pop)):
        R = np.random.uniform(0, 1)
        new_pop[i] = best - gamma * R * (best - pop[i])
    return new_pop

# Main Hybrid Algorithm (OPA + KHA + SHO)
def hybrid_optimization(dim=5, pop_size=20, generations=50, lower_bound=10, upper_bound=100):
    # Initialize population
    pop = initialize_population(pop_size, dim, lower_bound, upper_bound)
    best_solution = pop[0]
    best_fitness = objective_function(best_solution)

    for gen in range(generations):
        # Apply OPA (Exploration)
        pop = orca_predation(pop, best_solution)
        
        # Apply KHA (Refinement)
        pop = krill_herd_movement(pop, best_solution)
        
        # Apply SHO (Fine-tuning)
        pop = spotted_hyena_optimization(pop, best_solution)

        # Keep best solution
        for i in range(pop_size):
            fitness = objective_function(pop[i])
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = pop[i]
        
        print(f"Generation {gen+1}: Best Cost = {best_fitness}")

    return best_solution, best_fitness

# Run the hybrid optimization
best_sol, best_cost = hybrid_optimization()
print("\nFinal Best Solution:", best_sol)
print("Final Best Cost:", best_cost)

#optimum without price constraint  till the first drcimal
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time  # Import the time module for performance tracking


# Define the Meta-Heuristic Hybrid Optimizer with Memory
class HybridPowerFlowOptimizer:
    def __init__(self, A_matrix, line_limits, memory_size=100, max_sum_diff=4.0):
        """
        Initialize the optimizer.

        Parameters:
        - A_matrix: The power flow matrix.
        - line_limits: The line capacity limits.
        - memory_size: The size of the solution memory.
        - max_sum_diff: The maximum allowed sum of differences between original and optimized injections.
        """
        self.A = A_matrix
        self.line_limits = line_limits
        self.memory = deque(maxlen=memory_size)  # Memory of successful solutions
        self.num_buses = A_matrix.shape[1]
        self.max_sum_diff = max_sum_diff  # Maximum allowed sum of differences
        self.algorithm_weights = {'OOA': 0.3, 'KHA': 0.3, 'SHO': 0.4}  # Initial algorithm weights
        self.adaptive_weights = True  # Control adaptive weights.

    def optimize(self, B, iterations=200, population_size=20, alpha=0.7):
        """
        Optimize power flow using a hybrid of OOA, KHA, and SHO algorithms with memory.

        Parameters:
        - B: Original power injections.
        - iterations: Maximum number of iterations.
        - population_size: Size of the population.
        - alpha: Weight for minimizing changes to original injections (higher = more emphasis).
        """
        start_time = time.time()  # Record the start time

        # Check memory for similar problems first
        similar_solution = self._check_memory(B)
        if similar_solution is not None:
            # Start with a promising solution from memory
            print("Using solution from memory as starting point")
            initial_population = [similar_solution.copy() for _ in range(population_size // 2)]
            # Add random solutions for diversity
            initial_population.extend([self._generate_random_solution(B) for _ in range(population_size // 2)])
        else:
            # Generate random initial population
            initial_population = [self._generate_random_solution(B) for _ in range(population_size)]

        # Ensure the original B is in the population
        initial_population[0] = B.copy()

        # Initialize best solution
        best_solution = B.copy()
        best_fitness = self._calculate_fitness(best_solution, B, alpha)

        # Initialize tracking variables
        fitness_history = []
        algorithm_contributions = {'OOA': 0, 'KHA': 0, 'SHO': 0}
        algorithm_successes = {'OOA': 0, 'KHA': 0, 'SHO': 0}  # Track algorithm success
        stagnation_counter = 0
        previous_best_fitness = best_fitness

        # Main optimization loop
        for iteration in range(iterations):
            if self.adaptive_weights:
                self._update_algorithm_weights(algorithm_successes)  # Update weights adaptively

            new_population = []

            # Apply the algorithms to generate new solutions
            for i in range(population_size):
                # Select algorithm based on adaptive weights
                algorithm = random.choices(list(self.algorithm_weights.keys()),
                                           weights=list(self.algorithm_weights.values()), k=1)[0]

                if algorithm == 'OOA':
                    new_solution = self._apply_orcas_optimization(initial_population, i, best_solution, iteration,
                                                                 iterations)
                    algorithm_contributions['OOA'] += 1
                elif algorithm == 'KHA':
                    new_solution = self._apply_krill_herd(initial_population, i, best_solution, iteration, iterations)
                    algorithm_contributions['KHA'] += 1
                else:  # SHO
                    new_solution = self._apply_spotted_hyena(initial_population, i, best_solution, iteration, iterations)
                    algorithm_contributions['SHO'] += 1

                # Ensure power balance
                new_solution -= np.mean(new_solution)

                # Add to new population
                new_population.append(new_solution)

                # Check if new solution is better
                fitness = self._calculate_fitness(new_solution, B, alpha)

                # Only update best solution if it meets sum of differences constraint
                sum_diff = np.sum(np.abs(new_solution - B))
                if fitness < best_fitness and sum_diff <= self.max_sum_diff:
                    best_fitness = fitness
                    best_solution = new_solution.copy()
                    algorithm_successes[algorithm] += 1  # Increment algorithm success counter

                    # Debug output for tracking sum of differences
                    if iteration % 10 == 0 and i == 0:
                        print(f"Iteration {iteration}: Sum of differences = {sum_diff:.6f}, Best Fitness: {best_fitness:.6f}")
                else:
                    algorithm_successes[algorithm] *= 0.9  # Decay success if not improved

            # Replace old population with new population
            initial_population = new_population

            # Record fitness history
            fitness_history.append(best_fitness)

            # Check for stagnation and adapt algorithms if necessary
            if abs(best_fitness - previous_best_fitness) < 1e-6:
                stagnation_counter += 1
                if stagnation_counter >= 10:  # Increased stagnation threshold
                    print("Stagnation detected.  Re-initializing some solutions.")
                    # Re-initialize a portion of the population to increase diversity
                    for i in range(population_size // 4):  # Re-initialize 25% of population
                        initial_population[random.randint(0, population_size - 1)] = self._generate_random_solution(B)
                    stagnation_counter = 0  # Reset counter
            else:
                stagnation_counter = 0
            previous_best_fitness = best_fitness

            # Early stopping if we found a good solution
            if self._is_feasible(best_solution) and iteration > iterations // 4:
                # Check if the changes to injections are minimal
                deviation = np.linalg.norm(best_solution - B) / np.linalg.norm(B)
                sum_diff = np.sum(np.abs(best_solution - B))
                if deviation < 0.1 and sum_diff <= self.max_sum_diff:
                    print(f"Early stopping at iteration {iteration} - found good solution with minimal changes")
                    print(f"Sum of differences: {sum_diff:.6f} (max allowed: {self.max_sum_diff:.8f})")
                    break

        # Final correction to enforce power balance
        best_solution -= np.mean(best_solution)

        # Minimize changes to original injections while keeping line constraints and sum diff constraint satisfied
        best_solution = self._minimize_changes(best_solution, B)

        # Store successful solution in memory
        if self._is_feasible(best_solution):
            self._store_in_memory(B, best_solution)

        # Print algorithm contribution stats
        total = sum(algorithm_contributions.values())
        print("\nAlgorithm Contributions:")
        for algo, count in algorithm_contributions.items():
            print(f"{algo}: {count / total * 100:.1f}%")

        # Calculate and display final sum of differences
        final_sum_diff = np.sum(np.abs(best_solution - B))
        print(f"\nFinal sum of differences: {final_sum_diff:.6f} (max allowed: {self.max_sum_diff:.8f})")

        end_time = time.time()  # Record the end time
        print(f"Optimization time: {end_time - start_time:.2f} seconds")

        return best_solution, fitness_history, np.dot(self.A, best_solution)

    def _update_algorithm_weights(self, algorithm_successes):
        """
        Dynamically update algorithm weights based on their recent success.
        """
        total_success = sum(algorithm_successes.values())
        if total_success > 0:
            for algo in self.algorithm_weights:
                # Update weight based on success rate, with a smoothing factor
                self.algorithm_weights[algo] = 0.8 * self.algorithm_weights[algo] + 0.2 * (
                        algorithm_successes[algo] / total_success)
        else:
            # If no success, keep weights as they are
            pass

        # Normalize weights to sum to 1
        total_weight = sum(self.algorithm_weights.values())
        for algo in self.algorithm_weights:
            self.algorithm_weights[algo] /= total_weight

    def _minimize_changes(self, solution, original_B, steps=50):
        """
        Further minimize changes to original injections while maintaining feasibility
        and sum of differences constraint.

        Parameters:
        - solution: The current solution.
        - original_B: The original power injections.
        - steps: Number of steps to take in the minimization process.
        """
        best_solution = solution.copy()
        original_dist = np.linalg.norm(solution - original_B)

        # Try to gradually move the solution closer to the original B
        for step in range(steps):
            # Move solution slightly toward original B
            alpha = (steps - step) / steps  # Decreasing step size
            new_solution = best_solution + alpha * 0.02 * (original_B - best_solution)

            # Ensure power balance
            new_solution -= np.mean(new_solution)

            # Check if still feasible and within sum difference constraint
            sum_diff = np.sum(np.abs(new_solution - original_B))
            if self._is_feasible(new_solution) and sum_diff <= self.max_sum_diff:
                best_solution = new_solution.copy()
                new_dist = np.linalg.norm(best_solution - original_B)
                improvement = (original_dist - new_dist) / original_dist * 100
                if step % 10 == 0:
                    print(f"Minimizing changes: Step {step}, Improvement: {improvement:.2f}%, Sum diff: {sum_diff:.6f}")

        return best_solution

    def _generate_random_solution(self, B):
        """
        Generate a random solution based on the original B that respects the sum diff constraint.

        Parameters:
        - B: The original power injections.
        """
        max_attempts = 10

        for _ in range(max_attempts):
            # Use smaller perturbation to stay closer to original injections
            random_solution = B.copy() + np.random.uniform(-0.3, 0.3, size=B.shape) * np.abs(B).mean()
            random_solution -= np.mean(random_solution)  # Ensure power balance

            # Check sum of differences constraint
            sum_diff = np.sum(np.abs(random_solution - B))
            if sum_diff <= self.max_sum_diff:
                return random_solution

        # If we couldn't generate a solution within constraints, return a minimally perturbed one
        random_solution = B.copy() + np.random.uniform(-0.1, 0.1, size=B.shape) * np.abs(B).mean()
        random_solution -= np.mean(random_solution)  # Ensure power balance
        return random_solution

    def _calculate_fitness(self, solution, original_B, alpha=0.7):
        """
        Calculate fitness with penalty for constraint violations, deviation from original B,
        and exceeding sum difference constraint.

        Parameters:
        - solution: Current solution to evaluate.
        - original_B: Original power injections.
        - alpha: Weight for minimizing changes (higher = more emphasis).
        """
        C = np.dot(self.A, solution)

        # Penalty for line limit violations
        violation_penalty = np.sum(np.maximum(0, np.abs(C.flatten()) - self.line_limits))

        # Penalty for deviation from original injections
        deviation_penalty = np.linalg.norm(solution - original_B) / np.linalg.norm(original_B)

        # Additional penalty for exceeding sum difference constraint
        sum_diff = np.sum(np.abs(solution - original_B))
        sum_diff_penalty = max(0, sum_diff - self.max_sum_diff) * 10

        # Combined fitness with weighted penalties
        return (1 - alpha) * np.linalg.norm(C) + 10 * violation_penalty + alpha * deviation_penalty * 5 + sum_diff_penalty

    def _is_feasible(self, solution):
        """Check if a solution is feasible (satisfies all line constraints)."""
        # Check line flow constraints
        C = np.dot(self.A, solution)
        # Round to 1 decimal place for comparison
        C_rounded = np.round(np.abs(C.flatten()), 1)
        line_limits_rounded = np.round(self.line_limits, 1)
        line_constraints_met = np.all(C_rounded <= line_limits_rounded)

        return line_constraints_met

    def _check_memory(self, B):
        """Check memory for similar problems and return best matching solution."""
        if not self.memory:
            return None

        # Find solution with most similar initial conditions
        best_match = None
        best_similarity = float('inf')

        for initial_B, optimal_B in self.memory:
            similarity = np.linalg.norm(initial_B - B)

            # Also check if the solution satisfies sum difference constraint
            sum_diff = np.sum(np.abs(optimal_B - B))

            if similarity < best_similarity and sum_diff <= self.max_sum_diff:
                best_similarity = similarity
                best_match = optimal_B

        # Only use memory if it's reasonably similar
        if best_similarity < 0.5 * np.linalg.norm(B):
            return best_match
        return None

    def _store_in_memory(self, initial_B, optimal_B):
        """Store successful solutions in memory."""
        self.memory.append((initial_B.copy(), optimal_B.copy()))
        print(f"Solution stored in memory. Memory size: {len(self.memory)}")

    def _apply_orcas_optimization(self, population, index, best_solution, iteration, max_iterations):
        """Apply Orcas Optimization Algorithm (OOA) strategy."""
        solution = population[index].copy()

        # Parameters
        a = 2 * (1 - iteration / max_iterations)  # Decreases linearly from 2 to 0
        r1 = random.random()
        r2 = random.random()

        # Random solution for interaction
        random_index = random.choice([i for i in range(len(population)) if i != index])
        random_solution = population[random_index]

        # Binary success indicator (simplified)
        success = 1 if self._calculate_fitness(solution, solution) < self._calculate_fitness(random_solution,
                                                                                          solution) else 0

        # Bubble-net attacking strategy (exploitation)
        if random.random() < 0.5:
            # Spiral updating position
            distance = abs(random_solution - solution)
            spiral_factor = 1 * np.exp(1 * r1) * np.cos(2 * np.pi * r1)

            new_solution = best_solution - a * r2 * distance * spiral_factor
        # Search for prey strategy (exploration)
        else:
            new_solution = solution + a * r1 * (best_solution - solution) * success + \
                           a * r2 * (random_solution - solution) * (1 - success)

        return new_solution

    def _apply_krill_herd(self, population, index, best_solution, iteration, max_iterations):
        """Apply Krill Herd Algorithm (KHA) strategy."""
        solution = population[index].copy()

        # Parameters
        N_max = 0.01  # Maximum induced speed
        alpha = 0.5  # Inertia weight
        c_best = 2  # Best krill effect
        c_rand = 1  # Random krill effect

        # Motion induction (other krill effect)
        N_old = 0  # Previous motion
        neighbors = random.sample(population, min(3, len(population) - 1))

        # Calculate direction based on food attraction and neighbors
        food_attraction = c_best * best_solution
        neighbors_effect = c_rand * sum(neighbors) / len(neighbors)

        N_new = N_max * (alpha * N_old + food_attraction + neighbors_effect)

        # Foraging motion
        F = 0.02  # Maximum foraging speed
        foraging_factor = F * random.random()

        # Diffusion (random motion)
        D_max = 0.005 * (1 - iteration / max_iterations)  # Decreases over time
        D = D_max * (2 * np.random.random(solution.shape) - 1)

        # Combine all motions
        new_solution = solution + N_new + foraging_factor * (best_solution - solution) + D

        return new_solution

    def _apply_spotted_hyena(self, population, index, best_solution, iteration, max_iterations):
        """Apply Spotted Hyena Optimizer (SHO) strategy."""
        solution = population[index].copy()

        # Parameters
        h = 5 - iteration * ((5) / max_iterations)  # Decreases from 5 to 0

        # Select three random solutions
        random_indices = random.sample([i for i in range(len(population)) if i != index],
                                       min(3, len(population) - 1))
        random_solutions = [population[i] for i in random_indices]

        # Encircling prey mechanism
        E = h * (2 * random.random() - 1)

        if abs(E) >= 1:  # Exploration
            # Search for prey - select a random hyena
            random_hyena = random_solutions[0]
            D_h = abs(random_hyena - solution)
            new_solution = random_hyena - E * D_h
        else:  # Exploitation
            # Encircle prey - use the best hyena (alpha)
            D_alpha = abs(best_solution - solution)
            new_solution = best_solution - E * D_alpha

            # Add hunting behavior (get closer to prey)
            if random.random() < 0.5:
                # Group hunting strategy (average position of pack members)
                avg_position = sum(random_solutions) / len(random_solutions)
                new_solution = (new_solution + avg_position) / 2

        return new_solution


# Main Power Flow Optimization Function
def optimize_power_flow(A, B, line_limits, iterations=200, alpha=0.7, max_sum_diff=4.0):
    """
    Optimize power flow using the HybridPowerFlowOptimizer.

    Parameters:
    - A: The power flow matrix.
    - B: The initial power injections.
    - line_limits: The line capacity limits.
    - iterations: The maximum number of iterations.
    - alpha: The weighting factor for minimizing changes.
    - max_sum_diff: Maximum allowed sum of differences.

    Returns:
    - B_optimized: The optimized power injections.
    - fitness_history: The history of fitness values.
    - C_optimized: Optimized line flows.
    - C_unoptimized: Unoptimized line flows
    """
    optimizer = HybridPowerFlowOptimizer(A, line_limits, max_sum_diff=max_sum_diff)

    # Compute initial (unoptimized) power flows
    C_unoptimized = np.dot(A, B)

    # Run the Hybrid Optimization with Memory
    B_optimized, fitness_history, C_optimized = optimizer.optimize(B, iterations, alpha=alpha)

    return B_optimized, fitness_history, C_optimized, C_unoptimized


# New function to incrementally find minimum sufficient sum_diff
def find_minimum_sum_diff(A, B, line_limits, alpha=0.7, start_sum_diff=0.0,
                          step_size=0.001, max_attempts=10000, min_step_size=0.0001):
    """
    Incrementally find the minimum sufficient sum_diff that leads to a feasible solution.

    Parameters:
    - A: The power flow matrix.
    - B: The initial power injections.
    - line_limits: The line capacity limits.
    - alpha: Weighting factor.
    - start_sum_diff: Starting sum difference.
    - step_size: Initial step size.
    - max_attempts: Maximum number of attempts.
    - min_step_size: Minimum step size allowed.

    Returns:
    - B_optimized: The optimized power injections.
    - fitness_history: The history of fitness values.
    - C_optimized: Optimized line flows.
    - C_unoptimized: Unoptimized line flows
    - best_feasible_sum_diff: The minimum sum difference found.
    """
    print("Starting incremental search for minimum sum difference...")

    # Initialize
    current_sum_diff = start_sum_diff
    attempt = 0
    best_feasible_solution = None
    best_feasible_sum_diff = float('inf')

    # Start with the unoptimized power flows to check if they already satisfy constraints
    C_unoptimized = np.dot(A, B)
    if np.all(np.round(np.abs(C_unoptimized.flatten()), 1) <= np.round(line_limits, 1)):
        print("Initial power injections already satisfy all constraints!")
        return B, [], C_unoptimized, C_unoptimized, 0.0

    # Adapt step size based on problem size
    adaptive_step_size = max(min_step_size, np.mean(np.abs(B)) * 0.0001)
    current_step_size = adaptive_step_size
    print(f"Using adaptive step size: {current_step_size:.8f}")

    # Keep track of last failing and first succeeding sum_diff
    last_failing_sum_diff = None
    first_success_sum_diff = None

    while attempt < max_attempts:
        attempt += 1
        print(f"\nAttempt {attempt}: Testing max_sum_diff = {current_sum_diff:.8f}")

        try:
            optimizer = HybridPowerFlowOptimizer(A, line_limits, max_sum_diff=current_sum_diff)
            B_optimized, fitness_history, C_optimized = optimizer.optimize(B, iterations=150, alpha=alpha)

            # Check if constraints are satisfied
            is_feasible = np.all(np.round(np.abs(C_optimized.flatten()), 1) <= np.round(line_limits, 1))

            # Calculate actual sum of differences
            actual_sum_diff = np.sum(np.abs(B_optimized - B))

            if is_feasible:
                print(f"✓ FEASIBLE solution found with sum_diff = {actual_sum_diff:.8f}")

                # Record this as a feasible solution
                if actual_sum_diff < best_feasible_sum_diff:
                    best_feasible_solution = (B_optimized, fitness_history, C_optimized)
                    best_feasible_sum_diff = actual_sum_diff

                # Record first success if not set yet
                if first_success_sum_diff is None:
                    first_success_sum_diff = current_sum_diff
                    print(f"First successful sum_diff: {first_success_sum_diff:.8f}")

                    # After first success, do binary search between last failure and first success
                    if last_failing_sum_diff is not None:
                        # Switch to binary search for fine-tuning
                        print(
                            f"Switching to binary search between {last_failing_sum_diff:.8f} and {first_success_sum_diff:.8f}")
                        low = last_failing_sum_diff
                        high = first_success_sum_diff

                        for _ in range(20):  # Binary search iterations
                            mid = (low + high) / 2
                            print(f"Binary search testing sum_diff = {mid:.8f}")

                            optimizer = HybridPowerFlowOptimizer(A, line_limits, max_sum_diff=mid)
                            B_opt, fit_hist, C_opt = optimizer.optimize(B, iterations=150, alpha=alpha)

                            if np.all(np.round(np.abs(C_opt.flatten()), 1) <= np.round(line_limits, 1)):
                                high = mid
                                actual_sum = np.sum(np.abs(B_opt - B))
                                print(f"✓ Binary search found feasible with sum_diff = {actual_sum:.8f}")

                                if actual_sum < best_feasible_sum_diff:
                                    best_feasible_solution = (B_opt, fit_hist, C_opt)
                                    best_feasible_sum_diff = actual_sum
                            else:
                                low = mid
                                print(f"✗ Binary search solution infeasible at sum_diff = {mid:.8f}")

                        # Return best solution found through binary search
                        print(f"\nOptimal solution found through binary search!")
                        print(f"Minimum required sum_diff: {best_feasible_sum_diff:.8f}")

                        B_optimized, fitness_history, C_optimized = best_feasible_solution
                        return B_optimized, fitness_history, C_optimized, C_unoptimized, best_feasible_sum_diff

                # If we're already using very small increments, we're done
                if current_step_size <= min_step_size:
                    print(f"\nOptimal solution found with minimum sum_diff = {best_feasible_sum_diff:.8f}")
                    return best_feasible_solution[0], best_feasible_solution[1], best_feasible_solution[2], C_unoptimized, best_feasible_sum_diff

                # Reduce step size for more precise search.  Important for fine tuning.
                current_step_size = max(min_step_size, current_step_size * 0.1)
                current_sum_diff = max(0, current_sum_diff - current_step_size * 5) # Subtract a few steps worth
                print(f"Reducing step size to {current_step_size:.8f} and backtracking")

            else:
                print(f"✗ INFEASIBLE solution with sum_diff = {current_sum_diff:.8f}")

                # Record last failing sum_diff
                last_failing_sum_diff = current_sum_diff

                # Increase sum_diff to try to find a feasible solution
                current_sum_diff += current_step_size

        except Exception as e:
            print(f"Error during optimization: {e}")
            current_sum_diff += current_step_size

    # If we couldn't find any feasible solution
    if best_feasible_solution is None:
        print("\nFailed to find a feasible solution within the maximum attempts.")
        optimizer = HybridPowerFlowOptimizer(A, line_limits, max_sum_diff=10.0)  # Try with a large value
        B_optimized, fitness_history, C_optimized = optimizer.optimize(B, iterations=200, alpha=alpha)
        return B_optimized, fitness_history, C_optimized, C_unoptimized, 10.0

    print(f"\nBest feasible solution found with sum_diff = {best_feasible_sum_diff:.8f}")
    return best_feasible_solution[0], best_feasible_solution[1], best_feasible_solution[2], C_unoptimized, best_feasible_sum_diff



# Visualization and Results Reporting
def visualize_results(A, B, B_optimized, C_optimized, C_unoptimized, line_limits, fitness_history, max_sum_diff=4.0):
    """
    Visualize and report the results of the power flow optimization.

    Parameters:
    - A: The power flow matrix.
    - B: The initial power injections.
    - B_optimized: The optimized power injections.
    - C_optimized: The optimized line flows.
    - C_unoptimized: The unoptimized line flows.
    - line_limits: The line capacity limits.
    - fitness_history: The history of fitness values during optimization.
    - max_sum_diff: The maximum allowed sum of differences.
    """
    # Plot line flow comparison
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 8), C_unoptimized, marker='o', linestyle='-', label='Unoptimized Line Flows')
    plt.plot(range(1, 8), C_optimized, marker='s', linestyle='--', label='Optimized Line Flows')

    # Plot line limits
    for i in range(7):
        plt.axhline(y=line_limits[i], color='g', linestyle=':', alpha=0.5)
        plt.axhline(y=-line_limits[i], color='g', linestyle=':', alpha=0.5)

    plt.axhline(y=line_limits.mean(), color='r', linestyle='dotted', label='Average Line Limit')
    plt.xlabel('Line Index')
    plt.ylabel('Power Flow')
    plt.title('Power Flow Comparison Before and After Optimization')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot fitness score history
    if len(fitness_history) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(fitness_history)), fitness_history, marker='o', linestyle='-', color='b',
                 label='Fitness Score')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Score')
        plt.title('Fitness Score Convergence Over Iterations')
        plt.legend()
        plt.grid()
        plt.show()

    # Plot power injection comparison
    plt.figure(figsize=(10, 5))
    indices = range(1, 6)
    plt.bar(indices, B.flatten(), width=0.4, label='Initial Injections', alpha=0.7, color='blue')
    plt.bar([i + 0.4 for i in indices], B_optimized.flatten(), width=0.4, label='Optimized Injections', alpha=0.7,
            color='green')
    plt.xlabel('Bus Index')
    plt.ylabel('Power Injection')
    plt.title('Bus Power Injection Comparison')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    # Plot injection changes
    plt.figure(figsize=(10, 5))
    changes = B_optimized.flatten() - B.flatten()
    plt.bar(range(1, 6), changes, color=['green' if x >= 0 else 'red' for x in changes])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xlabel('Bus Index')
    plt.ylabel('Injection Change')
    plt.title('Difference Between Optimized and Initial Injections')
    plt.grid(axis='y')
    plt.show()

    # Calculate average change percentage
    abs_changes = np.abs(changes)
    abs_original = np.abs(B.flatten())
    pct_changes = abs_changes / abs_original * 100
    avg_pct_change = np.mean(pct_changes)

    # Calculate sum of differences
    sum_diff = np.sum(abs_changes)

    # Print results
    print("\nMatrix A (7x5):\n", A)
    print("\nInitial Power Injections (B, 5x1):\n", B)
    print("\nResultant Line Flows Before Optimization (7x1):\n", C_unoptimized)
    print("\nOptimized Power Injections (B_optimized, 5x1):\n", B_optimized)
    print("\nSum of Optimized Power Injections (Should be 0): ", np.sum(B_optimized))
    print("\nResultant Line Flows After Optimization (7x1):\n", C_optimized)

    # Print change metrics
    print("\nChanges to Power Injections:")
    print(f"Absolute Changes: {abs_changes}")
    print(f"Sum of Absolute Changes: {sum_diff:.8f}")
    print(f"Percentage Changes: {pct_changes}%")
    print(f"Average Percentage Change: {avg_pct_change:.2f}%")
    print(f"Total Deviation: {np.linalg.norm(B_optimized - B):.4f}")
    print(f"Relative Deviation: {np.linalg.norm(B_optimized - B) / np.linalg.norm(B) * 100:.2f}%")

    # Check sum difference constraint
    print(f"\nOptimal sum of differences: {sum_diff:.8f}")

    # Check that constraints are satisfied
    violations = np.where(np.round(np.abs(C_optimized.flatten()), 1) > np.round(line_limits, 1))[0]
    if len(violations) > 0:
        print(f"\nWarning: {len(violations)} line constraints still violated!")
        for line in violations:
            print(f"  Line {line + 1}: Flow = {abs(C_optimized[line][0]):.4f}, Limit = {line_limits[line]:.4f}")
    else:
        print("\nAll line constraints satisfied!")

    # Check total power flow
    print("\nTotal Line Power Flow Before Optimization:", np.sum(abs(C_unoptimized)))
    print("Total Line Power Flow After Optimization:", np.sum(abs(C_optimized)))
    print(f"Difference: {np.sum(abs(C_optimized)) - np.sum(abs(C_unoptimized)):.6f}")
    print(f"Percentage Difference: {(np.sum(abs(C_optimized)) - np.sum(abs(C_unoptimized))) / np.sum(abs(C_unoptimized)) * 100:.6f}%")



# Main program
if __name__ == "__main__":
    # Define the given 7x5 matrix (Power Flow Coefficients)
    A = np.array([
        [0.5857, -0.2571, -0.0428, -0.0857, -0.2000],
        [0.2143, 0.0571, -0.1572, -0.1143, -0.0000],
        [0.0905, 0.1619, -0.1953, -0.1238, 0.0666],
        [0.1080, 0.1651, -0.1206, -0.1968, 0.0444],
        [0.1872, 0.2158, 0.0730, 0.0349, -0.5109],
        [0.1048, 0.0191, 0.4476, -0.4380, -0.1334],
        [0.0128, -0.0158, 0.1270, 0.1651, -0.2891]
    ])

    # User input for Line Power Limits (7 lines)
    print("Enter power limits for each line:")
    line_limits = np.array([float(input(f"Enter power limit for Line {i + 1}: ")) for i in range(7)])

    # User input for Bus Injection Matrix B (5x1)
    print("\nEnter power injections for each bus:")
    B = np.array([float(input(f"Enter power injection for Bus {i + 1}: ")) for i in range(5)]).reshape(5, 1)

    # User input for weighting factor (how much to prioritize minimal changes)
    alpha = float(input("\nEnter alpha value (weight for minimal changes, 0.0-1.0, recommended: 0.9): "))

    # Run incremental optimization with increasingly small step sizes to find minimal sum_diff
    print("\nStarting incremental optimization to find minimum required sum of differences...")
    B_optimized, fitness_history, C_optimized, C_unoptimized, optimal_sum_diff = find_minimum_sum_diff(A, B,
                                                                                                      line_limits,
                                                                                                      alpha=alpha)
    print("\nVisualization of optimization results with minimum required sum of differences...")
    visualize_results(A, B, B_optimized, C_optimized, C_unoptimized, line_limits,
                      fitness_history, max_sum_diff=optimal_sum_diff)

    print("\nOptimization complete!")
    print(f"Minimum required sum of differences: {optimal_sum_diff:.8f}")

    # Ask user if they want to save the results
    save_results = input("\nDo you want to save the results to a file? (y/n): ")
    if save_results.lower() == 'y':
        filename = input("Enter filename (default: power_flow_results.txt): ") or "power_flow_results.txt"

        with open(filename, 'w') as f:
            f.write("Power Flow Optimization Results\n")
            f.write("==============================\n\n")

            f.write("Matrix A (Power Flow Coefficients, 7x5):\n")
            for row in A:
                f.write(str(row) + "\n")

            f.write("\nLine Power Limits:\n")
            for i, limit in enumerate(line_limits):
                f.write(f"Line {i + 1}: {limit}\n")

            f.write("\nInitial Power Injections (B, 5x1):\n")
            for i, injection in enumerate(B):
                f.write(f"Bus {i + 1}: {injection[0]}\n")

            f.write("\nOptimized Power Injections (B_optimized, 5x1):\n")
            for i, injection in enumerate(B_optimized):
                f.write(f"Bus {i + 1}: {injection[0]}\n")

            f.write("\nResultant Line Flows Before Optimization (C_unoptimized, 7x1):\n")
            for i, flow in enumerate(C_unoptimized):
                f.write(f"Line {i + 1}: {flow[0]}\n")

            f.write("\nResultant Line Flows After Optimization (C_optimized, 7x1):\n")
            for i, flow in enumerate(C_optimized):
                f.write(f"Line {i + 1}: {flow[0]}\n")

            f.write(f"\nMinimum Required Sum of Differences: {optimal_sum_diff:.8f}\n")

            # Check for violations
            violations = np.where(np.round(np.abs(C_optimized.flatten()), 1) > np.round(line_limits, 1))[0]
            if len(violations) > 0:
                f.write(f"\nWarning: {len(violations)} line constraints still violated!\n")
                for line in violations:
                    f.write(
                        f"  Line {line + 1}: Flow = {abs(C_optimized[line][0]):.4f}, Limit = {line_limits[line]:.4f}\n")
            else:
                f.write("\nAll line constraints satisfied!\n")

            f.write("\nOptimization Parameters:\n")
            f.write(f"Alpha (weight for minimal changes): {alpha}\n")

            print(f"Results saved to {filename}")

    # Ask if user wants to run another scenario
    run_again = input("\nDo you want to run another scenario? (y/n): ")
    if run_again.lower() == 'y':
        # The main() function was removed to avoid recursion issues.  Here we just re-execute the main logic.
        if __name__ == "__main__":
            # Define the given 7x5 matrix (Power Flow Coefficients)
            A = np.array([
                [0.5857, -0.2571, -0.0428, -0.0857, -0.2000],
                [0.2143, 0.0571, -0.1572, -0.1143, -0.0000],
                [0.0905, 0.1619, -0.1953, -0.1238, 0.0666],
                [0.1080, 0.1651, -0.1206, -0.1968, 0.0444],
                [0.1872, 0.2158, 0.0730, 0.0349, -0.5109],
                [0.1048, 0.0191, 0.4476, -0.4380, -0.1334],
                [0.0128, -0.0158, 0.1270, 0.1651, -0.2891]
            ])

            # User input for Line Power Limits (7 lines)
            print("Enter power limits for each line:")
            line_limits = np.array([float(input(f"Enter power limit for Line {i + 1}: ")) for i in range(7)])

            # User input for Bus Injection Matrix B (5x1)
            print("\nEnter power injections for each bus:")
            B = np.array([float(input(f"Enter power injection for Bus {i + 1}: ")) for i in range(5)]).reshape(5, 1)

            # User input for weighting factor (how much to prioritize minimal changes)
            alpha = float(input(
                "\nEnter alpha value (weight for minimal changes, 0.0-1.0, recommended: 0.9): "))

            # Run incremental optimization with increasingly small step sizes to find minimal sum_diff
            print("\nStarting incremental optimization to find minimum required sum of differences...")
            B_optimized, fitness_history, C_optimized, C_unoptimized, optimal_sum_diff = find_minimum_sum_diff(A, B,
                                                                                                              line_limits,
                                                                                                              alpha=alpha)
            print("\nVisualization of optimization results with minimum required sum of differences...")
            visualize_results(A, B, B_optimized, C_optimized, C_unoptimized, line_limits,
                              fitness_history, max_sum_diff=optimal_sum_diff)

            print("\nOptimization complete!")
            print(f"Minimum required sum of differences: {optimal_sum_diff:.8f}")

            # Ask user if they want to save the results
            save_results = input("\nDo you want to save the results to a file? (y/n): ")
            if save_results.lower() == 'y':
                filename = input("Enter filename (default: power_flow_results.txt): ") or "power_flow_results.txt"

                with open(filename, 'w') as f:
                    f.write("Power Flow Optimization Results\n")
                    f.write("==============================\n\n")

                    f.write("Matrix A (Power Flow Coefficients, 7x5):\n")
                    for row in A:
                        f.write(str(row) + "\n")

                    f.write("\nLine Power Limits:\n")
                    for i, limit in enumerate(line_limits):
                        f.write(f"Line {i + 1}: {limit}\n")

                    f.write("\nInitial Power Injections (B, 5x1):\n")
                    for i, injection in enumerate(B):
                        f.write(f"Bus {i + 1}: {injection[0]}\n")

                    f.write("\nOptimized Power Injections (B_optimized, 5x1):\n")
                    for i, injection in enumerate(B_optimized):
                        f.write(f"Bus {i + 1}: {injection[0]}\n")

                    f.write("\nResultant Line Flows Before Optimization (C_unoptimized, 7x1):\n")
                    for i, flow in enumerate(C_unoptimized):
                        f.write(f"Line {i + 1}: {flow[0]}\n")

                    f.write("\nResultant Line Flows After Optimization (C_optimized, 7x1):\n")
                    for i, flow in enumerate(C_optimized):
                        f.write(f"Line {i + 1}: {flow[0]}\n")

                    f.write(f"\nMinimum Required Sum of Differences: {optimal_sum_diff:.8f}\n")

                    # Check for violations
                    violations = np.where(np.round(np.abs(C_optimized.flatten()), 1) > np.round(line_limits, 1))[0]
                    if len(violations) > 0:
                        f.write(f"\nWarning: {len(violations)} line constraints still violated!\n")
                        for line in violations:
                            f.write(
                                f"  Line {line + 1}: Flow = {abs(C_optimized[line][0]):.4f}, Limit = {line_limits[line]:.4f}\n")
                    else:
                        f.write("\nAll line constraints satisfied!\n")

                    f.write("\nOptimization Parameters:\n")
                    f.write(f"Alpha (weight for minimal changes): {alpha}\n")

                    print(f"Results saved to {filename}")
            run_again = input("\nDo you want to run another scenario? (y/n): ")
            if run_again.lower() == 'n':
                print("\nThank you for using the Hybrid Power Flow Optimizer!")
            else:
                print("\nRestarting program...\n")
    else:
        print("\nThank you for using the Hybrid Power Flow Optimizer!")

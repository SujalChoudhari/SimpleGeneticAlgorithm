import numpy as np
import matplotlib.pyplot as plt

# Fitness function
def fitness_function(x):
    return ((x - 100) ** 2 - 10_000) / (-40)

# Selection class using Roulette Wheel Selection
class Selection:
    def __init__(self, population, fitness_function):
        self.population = population
        self.fitness_function = fitness_function

    def calculate_fitness(self):
        fitness_scores = np.array([self.fitness_function(individual) for individual in self.population])
        total_fitness = np.sum(fitness_scores)
        return fitness_scores, total_fitness

    def roulette_wheel_selection(self):
        fitness_scores, total_fitness = self.calculate_fitness()

        if total_fitness == 0:
            raise ValueError("Total fitness cannot be zero.")
        
        selection_probs = fitness_scores / total_fitness
        cumulative_probs = np.cumsum(selection_probs)

       
        parents = []
        for _ in range(4):
            rand_value = np.random.rand()
            parent_index = np.searchsorted(cumulative_probs, rand_value)
            parent_index = min(parent_index, len(self.population) - 1) 
            parents.append(self.population[parent_index])

        return parents

# Crossover class
class Crossover:
    def __init__(self, selected_parents):
        self.selected_parents = selected_parents

    def perform_crossover(self):
        children = []
        for _ in range(4): 
            parent1, parent2 = np.random.choice(self.selected_parents, 2, replace=False)
            print(f"Selected parents for crossover: {parent1}, {parent2}")
           
            child = (parent1 + parent2) / 2 
            children.append(child)
        return children

# Mutation class using bit mask
class Mutation:
    def __init__(self, children, mutation_rate=0.8):
        self.children = children
        self.mutation_rate = mutation_rate

    def perform_mutation(self):
        mutated_children = []
        for child in self.children:
           
            if np.random.rand() < self.mutation_rate:
                mutated_children.append(child + 5)
                mutated_children.append(child - 5)
                
                print(f"Mutation occurred with bits 00000101")
            else:
                mutated_children.append(child)

        return mutated_children

# Run generations
def run_generations(initial_population, max_generations=1000, convergence_threshold=100):
    population = initial_population.copy()
    best_fitness_list = [] 
    stable_generations = 0

    for generation_count in range(max_generations):
       
        selection = Selection(population, fitness_function)
        selected_parents = selection.roulette_wheel_selection()

       
        crossover = Crossover(selected_parents)
        children = crossover.perform_crossover()

       
        mutation = Mutation(children)
        mutated_children = mutation.perform_mutation()

       
        population = np.concatenate((mutated_children, population))

       
        current_fitness_scores = np.array([fitness_function(individual) for individual in population])
        best_fitness = np.max(current_fitness_scores)
        best_fitness_list.append(best_fitness)

        print(f"Generation {generation_count}: Best Fitness = {best_fitness}")

       
        if generation_count > 0 and best_fitness == best_fitness_list[-2]:
            stable_generations += 1
        else:
            stable_generations = 0

        if (stable_generations >= convergence_threshold and best_fitness > 200) or stable_generations >= convergence_threshold * 4:
            print("Convergence reached. Stopping early.")
            break

    return population, best_fitness_list

# Example usage
if __name__ == "__main__":
   
    initial_population = np.random.uniform(1, 10, 10) 
    print("Initial Population:", initial_population)

    final_population, best_fitness_list = run_generations(initial_population, max_generations=2000)

   
    plt.plot(best_fitness_list)
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid()
    
   
    plt.savefig('best_fitness_over_generations.png')
    plt.close() 

    print("Plot saved as 'best_fitness_over_generations.png'.")

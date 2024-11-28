import numpy as np
import random
from typing import List, Tuple, Optional

class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,      # Population size
        generations: int,   # Number of generations for the algorithm
        mutation_rate: float,  # Gene mutation rate
        crossover_rate: float,  # Gene crossover rate
        tournament_size: int,  # Tournament size for selection
        elitism: bool,         # Whether to apply elitism strategy
        random_seed: Optional[int],  # Random seed for reproducibility
    ):
        # Students need to set the algorithm parameters
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _init_population(self, M: int, N: int) -> List[List[int]]:
        """
        Initialize the population and generate random individuals, ensuring that every student is assigned at least one task.
        :param M: Number of students
        :param N: Number of tasks
        :return: Initialized population
        """

        # randomly generate the initial population
        population = []
        for _ in range(self.pop_size):
            individual = np.random.choice(N, size=M, replace=False).tolist()
            population.append(individual)
        return population

    def _fitness(self, individual: List[int], student_times: np.ndarray) -> float:
        """
        Fitness function: calculate the fitness value of an individual.
        :param individual: Individual
        :param student_times: Time required for each student to complete each task
        :return: Fitness value
        """

        # fitness function : calculate the fitness score(total time) of each individual
        fitness_value = 0       
        for i in range(len(individual)):
            fitness_value += student_times[i][individual[i]]
        return fitness_value

    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Use tournament selection to choose parents for crossover.
        :param population: Current population
        :param fitness_scores: Fitness scores for each individual
        :return: Selected parent
        """
        # tournament selection : select {tournament size} competitors based on the fitness scores at a time until the number of the selected individual reach the size of population
        selected = np.random.choice(range(len(population)), size=self.tournament_size, replace=False) # select {tournament size} competitors from the population
        selected_fitness = [fitness_scores[i] for i in selected] # find the corresponding fitness scores
        winner_index = selected[selected_fitness.index(min(selected_fitness))]  # find the index of individual with Minimized fitness score 
        return population[winner_index]

    def _crossover(self, parent1: List[int], parent2: List[int], M: int) -> Tuple[List[int], List[int]]:
        """
        Crossover: generate two offspring from two parents.
        :param parent1: Parent 1
        :param parent2: Parent 2
        :param M: Number of students
        :return: Generated offspring
        """
        # Single point crossover
        if random.random() < self.crossover_rate: # If crossover happen
            crossover_point = random.randint(1, M - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        else: # If crossover didn't happen
            offspring1, offspring2 = parent1, parent2
        return offspring1, offspring2

    def _mutate(self, individual: List[int], M: int, N: int) -> List[int]:
        """
        Mutation operation: randomly change some genes (task assignments) of the individual.
        :param individual: Individual
        :param M: Number of students
        :return: Mutated individual
        """
        # randomly modify genes
        for i in range(len(individual)):
            if random.random() < self.mutation_rate: # if mutation happen 
                individual[i] = random.randint(0, N-1) 
        return individual

    def __call__(self, M: int, N: int, student_times: np.ndarray) -> Tuple[List[int], int]:
        """
        Execute the genetic algorithm and return the optimal solution (allocation plan) and its total time cost.
        :param M: Number of students
        :param N: Number of tasks
        :param student_times: Time required for each student to complete each task
        :return: Optimal allocation plan and total time cost
        """

        best_individual = None
        best_fitness_score = 999999999
        solutions= [] # for M != N case, the possible task allocation
        time = [] # for M != N case, the possible total time
        tasks = list(range(N))

        # initialization
        population = self._init_population(M, N) # init population
        fitness_scores = [self._fitness(individual, student_times) for individual in population] # calculate the fitness scores of 1st generation
        for _ in range(self.generations):           
            new_population = []
            
            while len(new_population) < self.pop_size:
                # tournament selection
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                # crossover
                offspring1, offspring2 = self._crossover(parent1, parent2, M)
                # mutation
                new_population.append(self._mutate(offspring1, M, N))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(offspring2, M, N))   
                
            population = new_population # next generation
            fitness_scores = [self._fitness(individual, student_times) for individual in population] # recalculate the fitness scores of new population

            # find the best solution after the new generation appear 
            for i, fitness_score in enumerate(fitness_scores):
                curr_individual = population[i]
                if M == N:
                    if fitness_score <= best_fitness_score and len(set(curr_individual)) == len(curr_individual):                    
                        tmp = [a for a in curr_individual] # Deep copy                        
                        best_fitness_score = fitness_score
                        best_individual = tmp
                else:
                    if len(set(curr_individual)) == len(curr_individual):
                        required_time = 999999 # the minimal time required of the task that haven't been assigned to the students 
                        student = -1 # the student that need to do extra task
                        tmp = [a for a in curr_individual] # Deep copy
                        for task in tasks:
                            if task not in curr_individual:
                                for j in range(len(curr_individual)):
                                    if student_times[j][task] <= required_time:
                                        required_time = student_times[j][task]
                                        student = j
                                fitness_score += required_time
                                tmp[student] = [tmp[student]]
                                tmp[student].append(task)                             
                                time.append(fitness_score)
                                solutions.append(tmp)

        if M != N:
            best_fitness_score = min(time)
            best_individual = solutions[time.index(best_fitness_score)]

        best_fitness_score = int(best_fitness_score) # convert from numpy.int32 to int
        #print(best_individual)
        return best_individual, best_fitness_score

if __name__ == "__main__":
    def write_output_to_file(problem_num: int, total_time: int, filename: str = "answer.txt") -> None:
        """
        Write results to a file and check if the format is correct.
        """
        print(f"Problem {problem_num}: Total time = {total_time}")

        if not isinstance(total_time, int) :
            raise ValueError(f"Invalid format for problem {problem_num}. Total time should be an integer.")
        
        with open(filename, 'a') as file:
            file.write(f"Total time = {total_time}\n")

    M1, N1 = 2, 3
    cost1 = [[3, 2, 4],
             [4, 3, 2]]
    
    M2, N2 = 4, 4
    cost2 = [[5, 6, 7, 4],
             [4, 5, 6, 3],
             [6, 4, 5, 2],
             [3, 2, 4, 5]]
    
    M3, N3 = 8, 9
    cost3 = [[90, 100, 60, 5, 50, 1, 100, 80, 70],
             [100, 5, 90, 100, 50, 70, 60, 90, 100],
             [50, 1, 100, 70, 90, 60, 80, 100, 4], 
             [60, 100, 1, 80, 70, 90, 100, 50, 100],
             [70, 90, 50, 100, 100, 4, 1, 60, 80],
             [100, 60, 100, 90, 80, 5, 70, 100, 50],
             [100, 4, 80, 100, 90, 70, 50, 1, 60],
             [1, 90, 100, 50, 60, 80, 100, 70, 5]]
    
    M4, N4 = 3, 3
    cost4 = [[2, 5, 6],
             [4, 3, 5],
             [5, 6, 2]]
    
    M5, N5 = 4, 4
    cost5 = [[4, 6, 1, 5],
             [9, 6, 2, 1],
             [6, 5, 3, 9],
             [2, 2, 5, 4]]
    
    M6, N6 = 4, 4
    cost6 = [[5, 4, 6, 7],
             [8, 3, 4, 6],
             [6, 7, 3, 8],
             [7, 8, 9, 2]]
    
    M7, N7 = 4, 4
    cost7 = [[int(25*0.28), int(24*.333), int(23*.304), int(25*.12)],
             [int(25*.32), int(24*.25), int(23*.087), int(25*.24)],
             [int(25*.24), int(24*.125), int(23*.261), int(25*.28)],
             [int(25*.16), int(24*.292), int(23*.348), int(25*.36)]]
    
    nsuitable = 99999
    M8, N8 = 5, 5
    cost8 = [[8, 8, nsuitable, nsuitable, nsuitable],
             [6, nsuitable, 6, nsuitable, nsuitable],
             [nsuitable, 10, nsuitable, 10, nsuitable],
             [nsuitable, nsuitable, nsuitable, 7, 7],
             [nsuitable, nsuitable, 9, nsuitable, 9]]
    
    M9, N9 = 5, 5
    cost9 = [[10, 10, nsuitable, nsuitable, nsuitable],
             [12, nsuitable, nsuitable, 12, 12],
             [nsuitable, 15, 15, nsuitable, nsuitable],
             [11, nsuitable, 11, nsuitable, nsuitable],
             [nsuitable, 14, nsuitable, 14, 14]]
    
    M10, N10 = 9, 10
    cost10 = [[1, 90, 100, 50, 70, 20, 100, 60, 80, 90], 
            [100, 10, 1, 100, 60, 80, 70, 100, 50, 90],
            [90, 50, 70, 1, 100, 100, 60, 90, 80, 100],
            [70, 100, 90, 5, 10, 60, 100, 80, 90, 50],
            [50, 100, 100, 90, 20, 4, 80, 70, 60, 100],
            [100, 5, 80, 70, 90, 100, 4, 50, 1, 60],
            [90, 60, 50, 4, 100, 90, 100, 5, 10, 80],
            [100, 70, 90, 100, 4, 60, 1, 90, 100, 5],
            [80, 100, 5, 60, 50, 90, 70, 100, 4, 1]]

    problems = [(M1, N1, np.array(cost1)),
                (M2, N2, np.array(cost2)),
                (M3, N3, np.array(cost3)),
                (M4, N4, np.array(cost4)),
                (M5, N5, np.array(cost5)),
                (M6, N6, np.array(cost6)),
                (M7, N7, np.array(cost7)),
                (M8, N8, np.array(cost8)),
                (M9, N9, np.array(cost9)),
                (M10, N10, np.array(cost10))]
    
    ga = GeneticAlgorithm(
        pop_size=200,
        generations=2000,
        mutation_rate=0.25,
        crossover_rate=0.7,
        tournament_size=20,
        elitism=False,
        random_seed=None
    )

    # Solve each problem and immediately write the results to the file
    for _ in range(300):
        for i, (M, N, student_times) in enumerate(problems, 1):
            best_allocation, total_time = ga(M=M, N=N, student_times=student_times)
            write_output_to_file(i, total_time)

    print("Results have been written to answer.txt")

import random
import numpy as np
import requests


class Community:

    def __init__(self, totalPopulation=100, row_size=8, mutationRate=0.05):
        self.totalPopulation = totalPopulation
        self.row_size = row_size
        self.mutationRate = mutationRate
        self.population_array = self.initial_population()
        self.fitness_score()
        self.perfect_config = np.zeros((1, 1))

    def initial_population(self):
        # generate an array of configurations represented as arrays
        # expect shape as (totalPopulation, row_size)

        def fill_position(a):
            return np.apply_along_axis(lambda x: random.randint(0, self.row_size - 1), 0, population_array)

        population_array = np.zeros((self.totalPopulation, self.row_size))
        population_array = np.apply_along_axis(fill_position, 1, population_array)
        return population_array

    def fitness_score(self):
        # generate an array of shape (totalPopulation, 1) where each number corresponds
        # with a configuration in population array
        # the highest possible score is the length of the configuration
        self.score = np.apply_along_axis(lambda x: len(np.unique(x)), 1, self.population_array)

    def get_score(self):
        return self.score

    def crossover_mutation(self):
        # selection, crossover, and mutation in one function

        def parental_selection_and_crossover(a):
            # choose parents by weighted average in their scores
            # choose a random partition to crossover
            parents = np.array(random.choices(self.population_array, weights=self.score, k=2))
            crossOver = random.randint(0, self.row_size - 1)
            return np.concatenate((parents[0, :crossOver], parents[1, crossOver:]))

        def mutation(a):
            # randomly select an index to mutate if chosen to mutate
            # each configuration was extended by a random float from 0-1
            # so it uses the last indexed number to determine whether it mutates
            # will drop the last index after applying this function

            if a[-1] < self.mutationRate:
                a[random.randint(0, self.row_size - 1)] = random.randint(0, self.row_size - 1)

            return a

        queuing = np.zeros((self.totalPopulation, 1))
        developing_population = np.apply_along_axis(parental_selection_and_crossover, 1, queuing)
        mutation_chance = np.concatenate((developing_population, np.random.uniform(size=(self.totalPopulation, 1))),
                                         axis=1)
        new_population = np.apply_along_axis(mutation, 1, mutation_chance)
        self.population_array = np.delete(new_population, -1, 1)

    def find_perfect_config(self):
        # should be used only when the a configuation has all unique numbers
        # otherwise send a dummy response
        try:
            self.perfect_config = self.population_array[np.where(self.score == self.row_size)[0][0]]
        except IndexError:
            self.perfect_config = np.zeros(1)

    def get_perfect_config(self):
        # return response in string format with a space between each number
        return ' '.join(list(map(lambda x: str(int(x)), list(self.perfect_config))))


def main():
    totalPopulation = 100
    row_size = 8
    mutationRate = 0.05

    the_community = Community(totalPopulation, row_size, mutationRate)

    endless_loop = True
    counter = 0

    while endless_loop:
        counter += 1

        if max(the_community.get_score()) == row_size:
            the_community.find_perfect_config()
            print(the_community.get_perfect_config())
            break

        the_community.crossover_mutation()
        the_community.fitness_score()

if __name__ == "__main__":
    main()

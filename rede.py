import numpy as np
import random
import sys
import math

class ReSHADE:

    def __init__(self, borders, function, problem_size, population_size, max_nfes):
        self.borders = borders
        self.function = function
        self.problem_size = problem_size
        self.population_size = population_size
        self.max_nfes = max_nfes
        self.population = list()

    def create_population(self):
        for ind in range(self.population_size):
            individual = list()
            for param in range(self.problem_size):
                individual.append(random.uniform(self.borders[0], self.borders[1]))
            self.population.append(individual)

    def choose_random(self):
        r = [0, 0, 0, 0]
        counter = 0
        number = random.randint(0, self.population_size - 1)
        while counter != 4:
            if number != r[0] and number != r[1] and number != r[2] and number != r[3]:
                r[counter] = number
                counter += 1
            else:
                number = random.uniform(0, self.population_size - 1)
        return r

    def best_index(self, r):
        min = sys.maxsize
        min_ind = 0
        for ind in r:
            num = self.function(self.population[int(ind)])
            if num < min:
                min = num
                min_ind = ind
        return min_ind

    def worst_index(self, r):
        max = -sys.maxsize
        max_ind = 0
        for ind in r:
            num = self.function(self.population[int(ind)])
            if num < max:
                max = num
                max_ind = ind
        return max_ind

    def mutation(self, best, worst, t1, t2, f):
        r0 = -1
        while r0 == -1:
            num = random.randint(0, len(self.population) - 1)
            if num != best and num != worst and num != t1 and num != t2:
                r0 = num

        beta = random.uniform(0, 1)
        xc = beta * np.array(self.population[int(t1)]) + (1 - beta) * np.array(self.population[int(t2)])
        if self.function(self.population[r0]) <= self.function(self.population[worst]):
            vi = xc + f * (np.array(self.population[int(best)]) - xc) + f * (np.array(self.population[int(r0)]) -
                                                                        np.array(self.population[int(worst)]))
        else:
            vi = xc + f * (np.array(self.population[int(best)]) - xc) + f * (np.array(self.population[int(worst)]) -
                                                                        np.array(self.population[int(r0)]))
        return vi

    @staticmethod
    def crossover(x, v, cr):
        u = list()
        for i in range(len(x)):
            num = random.uniform(0, 1)
            j_rand = random.randint(0, len(x))
            if num <= cr or i == j_rand:
                u.append(v[i])
            else:
                u.append(x[i])
        return u

    def ReDE(self, cr, f):
        self.create_population()
        nfes = 0
        while nfes < self.max_nfes:
            u = list()
            for i in range(self.population_size):
                # Choose random parameters for mutation
                r = self.choose_random()
                best_ind = self.best_index(r)
                worst_ind = self.worst_index(r)
                t = []
                for ind in r:
                    if ind != best_ind and ind != worst_ind:
                        t.append(ind)
                vi = self.mutation(best_ind, worst_ind, t[0], t[1], f)  # Mutation
                u.append(self.crossover(self.population[i], vi, cr))  # Crossover
                nfes += 1

            for i in range(self.population_size):
                if self.function(u[i]) <= self.function(self.population[i]):
                    self.population[i] = u[i]
            print(nfes)
        print(self.population[0])
        return self.function(self.population[0])

    def ReSHADE(self):

        self.create_population()

        # Initialize memory lists of parameters

        m_cr = [0.5 for val in range(self.population_size)]
        m_f = [0.5 for val in range(self.population_size)]
        nfes = 0

        while nfes < self.max_nfes:
            u = list()
            # Initialize lists of parameters on current iterataion
            cr = [0 for val in range(self.population_size)]
            f = [0 for val in range(self.population_size)]
            s_cr = [0 for val in range(self.population_size)]
            s_f = [0 for val in range(self.population_size)]
            for i in range(self.population_size):

                # Choose random parameters for current iteration
                rand = random.randint(0, self.population_size - 1)
                cr[i] = np.random.normal(m_cr[rand], 0.1)
                f[i] = np.random.normal(m_f[rand], 0.1)

                r = self.choose_random()
                best_ind = self.best_index(r)
                worst_ind = self.worst_index(r)
                t = []
                for ind in r:
                    if ind != best_ind and ind != worst_ind:
                        t.append(ind)
                vi = self.mutation(best_ind, worst_ind, t[0], t[1], f[i])
                u.append(self.crossover(self.population[i], vi, cr[i]))
                nfes += 1

            population_tmp = list()
            for i in range(self.population_size):
                population_tmp.append(self.population[i])
                if self.function(u[i]) <= self.function(self.population[i]):
                    self.population[i] = u[i]
                if self.function(u[i]) < self.function(self.population[i]):
                    s_cr[i] = cr[i]
                    s_f[i] = f[i]

            if len(s_cr) and len(s_f):
                for i in range(self.population_size):
                    m_cr[i] = s_cr[i]
                    m_f[i] = s_f[i]

            print(nfes)
        print(self.population[0])
        return self.function(self.population[0])


def func(x):
    res = 0
    for x_ in x:
        res += x_ ** 2
    return res


def func2(x):
    sum = 0
    p = 0
    for i in range(len(x)):
        sum += math.fabs(x[i])
        p *= math.fabs(x[i])
    res = sum + p
    return res


test_case = ReSHADE(borders=[-100, 100], function=func, max_nfes=100000, population_size=100, problem_size=30)
# print(test_case.ReDE(0.8, 0.7))
print(test_case.ReSHADE())


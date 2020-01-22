# Basic Genetic Algortithm for Solving Inverse Kinematics Problem
# By : Eko Rudiawan
# Januari 2020

import numpy as np
import matplotlib.pyplot as plt 
import math
from scipy.spatial.distance import euclidean

class RobotArm:
    def __init__(self, links=[50, 40], target_pos=[0,0]):
        # Robot link length parameter
        self.links = links
        self.target_pos = target_pos

    def rotateZ(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return rz

    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return t

    # Forward Kinematics
    def FK(self, joints_angle):
        n_links = len(self.links)
        P = []
        P.append(np.eye(4))
        for i in range(0, n_links):
            R = self.rotateZ(joints_angle[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P

    # Here is objective function 
    # GA will minimize this function
    def calc_distance_error(self, joints_angle):
        P = self.FK(joints_angle)
        current_pos = [float(P[-1][0,3]), float(P[-1][1,3])]
        error = euclidean(current_pos, self.target_pos)
        return error

    # Plot joint configuration result
    def plot(self, joints_angle):
        fig = plt.figure() 
        ax = fig.add_subplot(1,1,1)
        P = self.FK(joints_angle)
        for i in range(len(self.links)):
            start_point = P[i]
            end_point = P[i+1]
            ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
        plt.show()

# Population Class
class Population:
    def __init__(self, l=8, limits=[(0, 1)], gen=[], use_random=True ):
        self.fitness = np.random.rand()
        self.l = l
        self.limits = limits 
        self.genotype_len = len(self.limits)*self.l
        if use_random:
            self.genotype = np.random.randint(0, 2, self.genotype_len)
        else:
            self.genotype = np.array(gen)
            self.genotype_len = self.genotype.shape[0]
        self.phenotype = self.decode()

    # Function for decoding genotype
    def decode(self):
        list_phenotype = []
        for i in range(len(self.limits)):
            lower, upper = self.limits[i]
            precission = (upper - lower) / (2**self.l - 1)
            _sum = 0
            cnt = 0
            for j in range(i*self.l, i*self.l+self.l):
                _sum += self.genotype[j] * 2**cnt 
                cnt += 1
            phenotype = _sum * precission + lower
            list_phenotype.append(phenotype)
        return tuple(list_phenotype)

class GeneticAlgorithm:
    def __init__(self, n_generations=10, n_populations=5, prob_crossover=1.0, prob_mutation=0.1, k=3):
        # Here we define simple 2 link arm robot with length l1 = 50 and l2 = 50
        self.robot = RobotArm(links=[50,50])
        # Initialize GA parameter
        self.n_generations = n_generations 
        self.n_populations = n_populations 
        self.prob_crossover = prob_crossover 
        self.prob_mutation = prob_mutation 
        self.k = k 
        # Generate population randomly
        self.populations = []
        for i in range(n_populations):
            # limits equal with joints angle limit in range -pi to pi
            pop = Population(l=16, limits=[(-np.pi, np.pi), (-np.pi, np.pi)])
            self.populations.append(pop)

    # Crossover operation between two parents, result in two children genotype
    def crossover(self, parent_1_idx, parent_2_idx, split_idx=[8,16,24]):
        genotype_len = self.populations[parent_1_idx].genotype_len
        child_1_genotype = np.hstack((self.populations[parent_1_idx].genotype[0:split_idx[0]], \
                                      self.populations[parent_2_idx].genotype[split_idx[0]:split_idx[1]], \
                                      self.populations[parent_1_idx].genotype[split_idx[1]:split_idx[2]], \
                                      self.populations[parent_2_idx].genotype[split_idx[2]:]))
        child_2_genotype = np.hstack((self.populations[parent_2_idx].genotype[0:split_idx[0]], \
                                      self.populations[parent_1_idx].genotype[split_idx[0]:split_idx[1]], \
                                      self.populations[parent_2_idx].genotype[split_idx[1]:split_idx[2]], \
                                      self.populations[parent_1_idx].genotype[split_idx[2]:]))
        return child_1_genotype, child_2_genotype

    # Mutation operation of children genotype, result in new children genotype
    def mutation(self, child_genotype):
        genotype_len = self.populations[0].genotype_len
        for i in range(genotype_len):
            mutate = np.random.choice([True, False], p=[self.prob_mutation, (1-self.prob_mutation)])
            if mutate:
                if child_genotype[i] == 0:
                    child_genotype[i] = 1
                else: 
                    child_genotype[i] = 0
        return child_genotype

    # Selection operation using tournament selection, result in two best parents from populations
    def tournament_selection(self):
        list_parents_idx = []
        for i in range(2):
            min_fitness = 999.0
            best_parent_idx = -1
            for j in range(self.k):
                accept = False
                while not accept:
                    parent_idx = np.random.choice(np.arange(0, len(self.populations)))
                    if parent_idx not in list_parents_idx:
                        accept = True
                if self.populations[parent_idx].fitness < min_fitness:
                    best_parent_idx = parent_idx
                    min_fitness = self.populations[parent_idx].fitness
            list_parents_idx.append(best_parent_idx)
        return tuple(list_parents_idx)

    # Here evolution process
    def evolution(self):
        for generation in range(self.n_generations):
            print("Generation ", generation)
            # Generate new children
            child_populations = []
            while len(child_populations) < self.n_populations:
                # Select best parent from population
                parent_1_idx, parent_2_idx = self.tournament_selection()
                # Crossover operation
                child_1_genotype, child_2_genotype = self.crossover(parent_1_idx, parent_2_idx)
                # Mutation operation
                child_1_genotype = self.mutation(child_1_genotype)
                child_2_genotype = self.mutation(child_2_genotype)

                child = Population(l=16, limits=[(-np.pi, np.pi), (-np.pi, np.pi)], gen=child_1_genotype, use_random=False)
                joint_1, joint_2 = child.phenotype
                # Get fitness value of new children
                child.fitness = self.robot.calc_distance_error([joint_1, joint_2])
                child_populations.append(child)

                child = Population(l=16, limits=[(-np.pi, np.pi), (-np.pi, np.pi)], gen=child_2_genotype, use_random=False)
                joint_1, joint_2 = child.phenotype
                child.fitness = self.robot.calc_distance_error([joint_1, joint_2])
                child_populations.append(child)

            # Update current parent with new child and track best population
            best_idx = -1
            best_fitness = 999
            for i in range(self.n_populations):
                self.populations[i] = child_populations[i]
                if self.populations[i].fitness < best_fitness:
                    best_idx = i 
                    best_fitness = self.populations[i].fitness
            print("Best Population :", self.populations[best_idx].phenotype, self.populations[best_idx].fitness)
            print("================================================================================")
        return self.populations[best_idx].phenotype

    def run(self):
        # Here we define target position of robot arm
        self.robot.target_pos = [10, 20]
        # Solving the solution with GA
        joint1, joint2 = self.evolution()
        # Plot robot configuration
        self.robot.plot([joint1, joint2])

def main():
    ga = GeneticAlgorithm(n_generations=50, n_populations=100, k=20)
    ga.run()

if __name__ == "__main__":
    main()
import numpy as np
import random
import itertools
import pandas as pd
import os
import sys  
from sklearn.model_selection import ParameterGrid
from scipy.stats import truncnorm

def get_truncated_normal(mean=0.5, sd=0.15, low=0, upp=1):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class population():
    def __init__(self, nInd, PayoffMtx, founder='PT', intruder=None, intruder_freq=0, memory=1, run_n=0): 
        self.nInd = nInd
        self.PayoffMtx = PayoffMtx
        fitness_dist = get_truncated_normal(mean=0.5, sd=0.3)
        np.random.seed(seed=run_n)
        self.fitness = fitness_dist.rvs(nInd)
        self.memory_fitness = []
        self.memory = memory
        self.strategies = np.array([founder]*self.nInd, dtype=np.dtype('U100'))
        self.founder = founder
        if intruder is not None:
            self.strategies[0 : int(intruder_freq*self.nInd)] = intruder
            np.random.shuffle(self.strategies)
        self.memory_about_me = None
        self.memory_about_them = None
        self.calculate_cooperativity()
        self.fitness_history = []
        self.coop_history = []
        self.pop_history = [self.strategies]

    def return_payoffs(self, row, col):
        if (not row.size) | (not col.size):
            return False
        result1, result2 = zip(*map(lambda x, y: self.PayoffMtx[(x,y)], row, col))
        return np.array(result1), np.array(result2)

    def calculate_fitness(self, payoffs):
        #softmax
        max_payoff = np.max(payoffs)
        return np.exp(payoffs - max_payoff)/sum(np.exp(payoffs - max_payoff))
    
    def individual_cooperativity(self, fitness, strategy, ind_id):
        if strategy == 'C':
            return 1.0
        elif strategy == 'D':
            return 0.0
        elif strategy == 'PT':
            #the simplest case: probability to cooperate equals fitness
            if (self.memory == 1) | (not self.memory_fitness):
                return fitness
            else:
                temp = np.vstack(self.memory_fitness)
                #print(temp[:, ind_id], fitness)
                return (fitness + temp[:, ind_id].sum())/(len(self.memory_fitness) + 1)
            #1/(1 + np.exp(-fitness + half_prob))
        elif strategy == 'RT':
            if (self.memory == 1) | (not self.memory_fitness):
                return 1-fitness
            else:
                temp = np.vstack(self.memory_fitness)
                return 1 - ((fitness + temp[:, ind_id].sum())/(len(self.memory_fitness) + 1))
            
        elif strategy == 'TFT':
            if self.memory_about_them is None:
                return 1.0
            else:
                #decisions others did to me
                previous_decisions = self.memory_about_them[:, ind_id]
            values, counts = np.unique(previous_decisions, return_counts=True)
            return previous_decisions[np.argmax(counts)]
        
        elif strategy == 'WSLS':
            if self.memory_about_me is None:
                return float((fitness >= 0.5))
            else:
                #decisions I did to others
                previous_decisions = self.memory_about_me[ind_id, :]
            values, counts = np.unique(previous_decisions, return_counts=True)
            #if fitness >= 0.5:
            if fitness > self.fitness_history[-1].mean():
                return previous_decisions[np.argmax(counts)]
            else:
                return 1 - previous_decisions[np.argmax(counts)]
        
        elif strategy == 'STFT':
            if self.memory_about_them is None:
                return 1.0
            else:
                #decisions others did to me
                previous_decisions = self.memory_about_them[:, ind_id]
            values, counts = np.unique(previous_decisions, return_counts=True)
            if previous_decisions[np.argmax(counts)] < 0.5:
                return 0.01
            else:
                return 0.99
        
        elif strategy == 'SWSLS':
            if self.memory_about_me is None:
                return float((fitness >= 0.5))
            else:
                #decisions I did to others
                previous_decisions = self.memory_about_me[ind_id, :]
            values, counts = np.unique(previous_decisions, return_counts=True)
            if fitness > self.fitness_history[-1].mean():
                if previous_decisions[np.argmax(counts)] < 0.5:
                    return 0.01
                else:
                    return 0.99
            else:
                if previous_decisions[np.argmax(counts)] < 0.5:
                    return 0.99
                else:
                    return 0.01
        
        elif strategy == 'random':
            return 0.5
    
    def calculate_cooperativity(self):
        self.cooperativity = np.array(list(map(self.individual_cooperativity, 
                                               self.fitness, self.strategies, 
                                               range(self.nInd))))
        #print(self.cooperativity)
    
    def generation(self):
        #game
        self.memory_fitness.append(self.fitness.copy())
        if len(self.memory_fitness) == self.memory:
            self.memory_fitness.pop(0)
        decisions_mtx = []
        for i in range(self.nInd):
            i_cooperativity = self.cooperativity[i]
            #print(i_cooperativity)
            i_decisions = np.random.choice([0, 1], self.nInd, p=[1 - i_cooperativity, i_cooperativity])
            decisions_mtx.append(i_decisions)
        decisions_mtx = np.vstack(decisions_mtx)
        payoffs_mtx = np.zeros_like(decisions_mtx, dtype=float)
        for i in range(self.nInd):
            payoffs = self.return_payoffs(decisions_mtx[i, i:], decisions_mtx[i:, i])
            if payoffs:
                payoffs_mtx[i, i:], payoffs_mtx[i:, i] = payoffs
            payoffs_mtx[i, i] = 0
        ind_payoffs = payoffs_mtx.sum(axis=1)
        #print(ind_payoffs)
        self.fitness = self.calculate_fitness(ind_payoffs)
        self.fitness_history.append(self.fitness)
        new_indices = np.random.choice(np.arange(self.nInd), self.nInd, 
                                       replace=True, p=self.fitness)
        self.strategies = np.take(self.strategies, new_indices)
        self.fitness = np.take(self.fitness, new_indices)
        #updating memories
        for i in range(len(self.memory_fitness)):
            self.memory_fitness[i] = np.take(self.memory_fitness[i], new_indices)
        self.memory_about_me = np.take(decisions_mtx, new_indices, axis=0)
        self.memory_about_them = np.take(decisions_mtx, new_indices, axis=1)
        self.calculate_cooperativity()

    def evolve(self,nGen):
        for i in range(nGen):
            self.coop_history.append(self.cooperativity)
            self.generation()
            #self.fitness_history.append(self.fitness)
            self.pop_history.append(self.strategies)
        self.getTraj()
        
    def save_history(self, file_to_save):
        gen = 1
        with open(file_to_save, 'w') as outf:
            outf.write(f'gen\tcooperativity\tfitness\tfounder_freq\n')
            for coop, fitness, freq in zip(self.coop_traj, self.fitness_traj, self.pop_traj):
                outf.write(f'{gen}\t{coop}\t{fitness}\t{freq}\n')
                gen += 1
        
    def getTraj(self):
        self.fitness_traj = np.array(self.fitness_history).mean(axis=1)
        self.coop_traj = np.array(self.coop_history).mean(axis=1)
        history_array = np.array(self.pop_history)
        self.pop_traj = (history_array == self.founder).sum(axis=1)/self.nInd

    def plotTraj(self,ax="auto", save=False, track='cooperativity'):
        if ax=="auto":
            #plt.plot(self.fitness_traj)
            #plt.axis([0, len(self.fitness_traj), 0, 1])
            if track == 'cooperativity':
                plt.plot(self.coop_traj)
            if track == 'fitness':
                plt.plot(self.fitness_traj)
            if track == 'frequency':
                plt.plot(self.pop_traj)
            #plt.axis([0, len(self.coop_traj), 0, 1])

def run_invasion_experiment(founder, intruder, nInd, payoff_mtx, intruder_freq, nGen, run_n, memory=1):
    pop = population(nInd, payoff_mtx, founder=founder, intruder=intruder, 
                     intruder_freq=intruder_freq, memory=memory, run_n=int(run_n))
    pop.evolve(nGen)
    pop.save_history(f'invasion_results_repeat/f={founder},i={intruder},n={nInd},ifreq={intruder_freq},mem={memory},run={run_n}.tsv')
    #del pop

if __name__ == '__main__':
    #0 - cheat, 1 - cooperate
    payoff_mtx = {(0, 0): (0, 0), (0, 1): (0.8, 0), (1, 0): (0, 0.8), (1, 1): (0.4, 0.4)}
    #parameters
    #nRuns = 30
    nGen = 100
    nInd = 1000
    intruder_freq = 0.001
    memory = 1
    founder = sys.argv[1]
    intruder = sys.argv[2]
    run_n = sys.argv[3]
    #for i in range(nRuns):
    run_invasion_experiment(founder, intruder, nInd, payoff_mtx, intruder_freq, nGen, run_n, memory=memory)
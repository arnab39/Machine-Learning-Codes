

from random import (random, randint)
import matplotlib.pyplot as plt

__all__ = ['Chromosome', 'Population']

class Chromosome:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = Chromosome._update_fitness(gene)
    
    def mate(self, mate):
        pivot = randint(0, len(self.gene) - 1)
        gene1 = self.gene[:pivot] + mate.gene[pivot:]
        gene2 = mate.gene[:pivot] + self.gene[pivot:]        
        return Chromosome(gene1), Chromosome(gene2)
        
    def mate_2point(self, mate):
        pivot1 = randint(0, int(len(self.gene)/2) - 1)
        pivot2 = randint(int(len(self.gene)/2),len(self.gene)-1)
        gene1 = self.gene[:pivot1] + mate.gene[pivot1:pivot2] + self.gene[pivot2:]
        gene2 = mate.gene[:pivot1] + self.gene[pivot1:pivot2] + mate.gene[pivot2:]         
        return Chromosome(gene1), Chromosome(gene2)
    
    def mutate(self):
        gene = self.gene
        mut_range=10
        idx = randint(0, len(gene) - (1+mut_range))
        for i in range(mut_range):
            gene[idx+i] = 100-gene[idx+i];        
        return Chromosome(gene)

    @staticmethod            
    def _update_fitness(gene):
        fitness = len([1 for item in gene if item==5 ])
        return fitness
        
    @staticmethod
    def gen_random():
        gene = []
        for x in range(100):
            gene.append(randint(0, 101))               
        return Chromosome(gene)
        
class Population:
    
    def __init__(self, size=100, crossover=0.8, elitism=0.0, mutation=0.05):
        self.elitism = elitism
        self.mutation = mutation
        self.crossover = crossover
        
        buf = []
        for i in range(size): buf.append(Chromosome.gen_random())
        #self.population = sorted(buf, key=lambda x: x.fitness)
        self.population = buf
                        
    def roulette_wheel(self):
        size = len(self.population)
        tot_fitness=0;
        for i in range(size):
            tot_fitness+=self.population[i].fitness
        rand_fitness=randint(0, tot_fitness)
        tot_fitness=0
        for i in range(size):
            tot_fitness+=self.population[i].fitness
            if rand_fitness<=tot_fitness:
                return self.population[i]

    def select_parents(self):
        return (self.roulette_wheel(), self.roulette_wheel())
        
    def evolve(self):
        size = len(self.population)
        idx = int(round(size * self.elitism))
        buf = self.population[:idx]
        
        while (idx < size):
            if random() <= self.crossover:
                (p1, p2) = self.select_parents()
                #children = p1.mate_2point(p2)
                children = p1.mate(p2)
                for c in children:
                    if random() <= self.mutation:
                        buf.append(c.mutate())
                    else:
                        buf.append(c)
                idx += 2
            else:
                if random() <= self.mutation:
                    buf.append(self.population[idx].mutate())
                else:
                    buf.append(self.population[idx])
        
        self.population = sorted(buf[:size], key=lambda x: x.fitness)
        self.population = buf[:size]
        
    def find_fitness(self):
        max_fitness=0
        min_fitness=100
        avg_fitness=0
        for i in range(len(self.population)):
            if(max_fitness<self.population[i].fitness):
                max_fitness=self.population[i].fitness
            if(min_fitness>self.population[i].fitness):
                min_fitness=self.population[i].fitness
            avg_fitness+=self.population[i].fitness
        avg_fitness/=len(self.population)    
        print("Max Fitness: %d" % (max_fitness))
        print("Avg Fitness: %d" % (avg_fitness))
        print("Min Fitness: %d" % (min_fitness))
        return max_fitness,min_fitness,avg_fitness

if __name__ == "__main__":
    maxGenerations = 1000
    pop = Population(size=100, crossover=0.8, elitism=0.0, mutation=0.05)
    max_fitness=[]
    min_fitness=[]
    avg_fitness=[]
    gen=[]
    conv_thres=100
    max_last_iter=0
    count=0
    for i in range(1, maxGenerations+1):
        print("Generation %d:" % (i))
        maf,mif,avf=pop.find_fitness()
        pop.evolve()
        if(max_last_iter==maf):
            count+=1
        else:
            count=0        
        max_fitness.append(maf)
        min_fitness.append(mif)
        avg_fitness.append(avf)
        gen.append(i)
        if(count==conv_thres):
            break
        max_last_iter=maf
    print("Convergence success.")
    plt.plot(gen,max_fitness)
    plt.plot(gen,avg_fitness)
    plt.plot(gen,min_fitness)
    plt.ylim([0, 30])
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Max, Min & Avg Fitness vs Generations')
    plt.legend(['Max_fitness', 'Avg_fitness', 'Min_fitness'], loc='upper left')
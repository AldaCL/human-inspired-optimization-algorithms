import random
import math
import numpy as np

# Parámetros globales
POPSIZE = 30               # Tamaño de la población
FES = 300000               # Máximo de evaluaciones de función
TIMES = 30                 # Número de ejecuciones
DIMS = 30                  # Número de variables del problema
p_i = 0.7                  # Probabilidad de imitación
p_r = 0.2                  # Probabilidad de randomización
SN1 = POPSIZE // 2         # Número de miembros modelo
SN2 = POPSIZE - SN1        # Número de no-modelos
lbound, ubound = -100, 100 # Límites de variables

# Estructura de un individuo
class Individual:
    def __init__(self):
        self.x = [random.uniform(lbound, ubound) for _ in range(DIMS)]
        self.fit = float('inf')

# Poblaciones
pop = [Individual() for _ in range(POPSIZE)]
newpop = [Individual() for _ in range(POPSIZE)]

# Variables de control
fes = 0
gbestval = float('inf')
gbestind = -1
t_Val = [0.0] * DIMS
AT = mAT = 0

# Función objetivo, reemplaza esta con tu función
def function_name(pos, dim):
    return sum(x**2 for x in pos)  # Ejemplo: función esfera

# Funciones auxiliares
def randval(low, high):
    return random.uniform(low, high)

def cmp_fitness(ind):
    return ind.fit

def t_test(sample1, sample2):
    mean_1, mean_2 = np.mean(sample1), np.mean(sample2)
    var_1, var_2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    df = len(sample1) + len(sample2) - 2
    pooled_var = ((len(sample1) - 1) * var_1 + (len(sample2) - 1) * var_2) / df
    if pooled_var == 0:
        return 0
    return (mean_1 - mean_2) / math.sqrt(pooled_var * (1/len(sample1) + 1/len(sample2)))

# Inicialización
def Initialize():
    global fes, gbestval
    fes = 0
    for ind in pop:
        ind.x = [randval(lbound, ubound) for _ in range(DIMS)]
    gbestval = float('inf')

# Evaluación
def Evaluate():
    global fes, gbestval, gbestind
    for i, ind in enumerate(pop):
        fes += 1
        ind.fit = function_name(ind.x, DIMS)
        if ind.fit < gbestval:
            gbestval = ind.fit
            gbestind = i

# Atención
def Attention():
    global AT, mAT
    pop.sort(key=cmp_fitness)
    for j in range(DIMS):
        s1 = [pop[i].x[j] for i in range(SN1)]
        s2 = [pop[i].x[j] for i in range(SN1, POPSIZE)]
        t_Val[j] = t_test(s1, s2)
    AT = abs(t_Val[random.randint(0, DIMS - 1)])
    mAT = -AT

# Reproducción y Refuerzo
def Reproduction_and_Reinforcement():
    for i in range(POPSIZE):
        for j in range(DIMS):
            r = random.randint(0, SN1 - 1)
            r1 = random.randint(0, POPSIZE - 1)
            nd = random.uniform(0, 1)
            delta = abs(pop[r].x[j] - pop[i].x[j])

            if t_Val[j] >= AT:
                newpop[i].x[j] = pop[r].x[j] + nd * delta
            elif t_Val[j] <= mAT:
                newpop[i].x[j] = pop[r].x[j] - nd * delta
            else:
                if random.uniform(0, 1) < p_i:
                    newpop[i].x[j] = pop[r1].x[j]
                elif random.uniform(0, 1) < p_r:
                    newpop[i].x[j] = randval(lbound, ubound)
                else:
                    newpop[i].x[j] = pop[i].x[j]

            # Control de límites
            if newpop[i].x[j] > ubound:
                newpop[i].x[j] = ubound - 0.5 * (newpop[i].x[j] - ubound)
            elif newpop[i].x[j] < lbound:
                newpop[i].x[j] = lbound + 0.5 * (lbound - newpop[i].x[j])

# Motivación
def Motivation():
    global fes, gbestval, gbestind
    for i in range(POPSIZE):
        fes += 1
        newpop[i].fit = function_name(newpop[i].x, DIMS)
        if newpop[i].fit <= pop[i].fit:
            pop[i] = newpop[i]
            if pop[i].fit < gbestval:
                gbestval = pop[i].fit
                gbestind = i

# Proceso principal
def Process():
    Initialize()
    Evaluate()
    while fes < FES:
        Attention()
        Reproduction_and_Reinforcement()
        Motivation()

# Main
if __name__ == "__main__":
    for run in range(TIMES):
        Process()
        print(f"Mejor valor en ejecución {run + 1}: {gbestval}")
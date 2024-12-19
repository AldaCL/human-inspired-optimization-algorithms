import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time 

# Parámetros globales
POPSIZE = 30               # Tamaño de la población
FES = 300                  # Máximo de evaluaciones de función
TIMES = 30                 # Número de ejecuciones
DIMS = 20                  # Número de variables del problema
p_i = 0.2                 # Probabilidad de imitación
p_r = 0.9                  # Probabilidad de randomización
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

# Almacenamiento de datos de todas las ejecuciones
all_runs_data = {
    "best_fitness_per_run": [],
    "average_fitness_per_run": [],
    "diversity_per_run": []
}

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

# Almacenar datos de iteración
def StoreIterationData(run_data):
    best = min(ind.fit for ind in pop)
    avg = np.mean([ind.fit for ind in pop])
    diversity = np.std([ind.fit for ind in pop])
    run_data["best_fitness"].append(best)
    run_data["average_fitness"].append(avg)
    run_data["diversity"].append(diversity)

# Proceso principal
def Process():
    run_data = {
        "best_fitness": [],
        "average_fitness": [],
        "diversity": []
    }
    Initialize()
    Evaluate()
    while fes < FES:
        Attention()
        Reproduction_and_Reinforcement()
        Motivation()
        StoreIterationData(run_data)
    return run_data

# Gráficas
def PlotResults(all_runs_data):
    
    #Plot benchmark function
    x = np.linspace(lbound, ubound, 100)
    y = np.linspace(lbound, ubound, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function_name([X[i, j], Y[i, j]], 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    # Graficar convergencia promedio
    avg_best_fitness = np.mean(all_runs_data["best_fitness_per_run"], axis=0)
    avg_average_fitness = np.mean(all_runs_data["average_fitness_per_run"], axis=0)
    avg_diversity = np.mean(all_runs_data["diversity_per_run"], axis=0)

    plt.figure()
    plt.plot(avg_best_fitness, label="Average Best Fitness")
    plt.plot(avg_average_fitness, label="Average Population Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Over All Runs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # save figure to file with a timestamp in the name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"convergence_{timestamp}.png")
    
    plt.figure()
    plt.plot(avg_diversity, label="Average Diversity")
    plt.xlabel("Iteration")
    plt.ylabel("Diversity (Std. Dev.)")
    plt.title("Diversity Over All Runs")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # save figure to file with a timestamp in the name
    plt.savefig(f"diversity_{timestamp}.png")

# Main
if __name__ == "__main__":
    for run in range(TIMES):
        print(f"Running iteration {run + 1}/{TIMES}...")
        run_data = Process()
        all_runs_data["best_fitness_per_run"].append(run_data["best_fitness"])
        all_runs_data["average_fitness_per_run"].append(run_data["average_fitness"])
        all_runs_data["diversity_per_run"].append(run_data["diversity"])

    # Generar gráficos al final
    PlotResults(all_runs_data)
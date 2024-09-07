import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def vietoris_rips(points, epsilon):
    simplices = []
    n = len(points)
    
    # Adiciona 0-simplices (pontos)
    for i in range(n):
        simplices.append([i])
    
    # Adiciona 1-simplices (arestas)
    for i, j in combinations(range(n), 2):
        if np.linalg.norm(points[i] - points[j]) < 2* epsilon:
            simplices.append([i, j])
    
    # Adiciona 2-simplices (triângulos)
    for simplex1, simplex2, simplex3 in combinations(simplices, 3):
        if len(set(simplex1).intersection(set(simplex2))) > 0:
            if len(set(simplex1).intersection(set(simplex3))) > 0:
                if len(set(simplex2).intersection(set(simplex3))) > 0:
                    simplices.append(list(set(simplex1).union(set(simplex2)).union(set(simplex3))))
    
    return simplices

def plot_vietoris_rips(points, epsilon_values):
    num_plots = len(epsilon_values)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    for ax, epsilon in zip(axes, epsilon_values):
        simplices = vietoris_rips(points, epsilon)
        
        # Plota os pontos com as circunferências de raio epsilon
        for point in points:
            circle = Circle(point, epsilon, edgecolor='b', facecolor='none')
            ax.add_patch(circle)
            ax.plot(point[0], point[1], 'bo')
        
        # Plota os triângulos
        for simplex in simplices:
            if len(simplex) == 3:  # Triângulos
                triangle = points[simplex]
                ax.fill(triangle[:, 0], triangle[:, 1], edgecolor='none', facecolor='lightblue')  
        
        # Plota as arestas
        for simplex in simplices:
            if len(simplex) == 2:  # Arestas
                point1, point2 = points[simplex[0]], points[simplex[1]]
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-')
        
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_title(f'ε = {epsilon}')
        
    
    plt.suptitle('Complexo de Vietoris-Rips para diferentes valores de ε')
    plt.show()

# Exemplo de conjunto de dados
points = np.array([[0, 0], [1, 0.5], [0.5, 1], [2, 2]])

# Valores consecutivos de epsilon
epsilon_values = [0.2, 0.4, 0.7, 1, 1.5]

# Plota cada complexo de Vietoris-Rips em subplots lado a lado
plot_vietoris_rips(points, epsilon_values)

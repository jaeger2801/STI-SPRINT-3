import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import ipywidgets as widgets

# Función que servirá para calcular la fuerza de vínculo entre usuarios
def calcular_fuerza_vinculo(usuario1, usuario2):
    similitud_coseno = 1 - cosine(usuario1, usuario2)
    return similitud_coseno

# Función para encontrar los K vecinos más cercanos a un usuario
def encontrar_vecinos_cercanos(usuario, usuarios, nombres, k):
    # Crear el modelo KNN y ajustarlo a los datos
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(usuarios)

    # Encontrar los índices y distancias de los K vecinos más cercanos
    distancias, indices = knn.kneighbors([usuario])

    vecinos = []
    for i in range(k):
        vecino = {
            'nombre': nombres[indices[0][i]],
            'distancia': distancias[0][i],
            'fuerza_vinculo': calcular_fuerza_vinculo(usuario, usuarios.loc[nombres[indices[0][i]]])
        }
        vecinos.append(vecino)
    
    return vecinos

# Cargar datos desde el archivo CSV
datos = pd.read_csv('Base De Datos Pizza.csv', index_col='Nombre')

# Obtener matriz de características (sabores de pizza) y usuario seleccionado
sabores_pizza = datos.iloc[:, 1:].copy()
nombres = datos.index
usuario_seleccionado = sabores_pizza.iloc[0]

k = int(np.sqrt(len(sabores_pizza)))  # Valor arbitrario para K (raíz cuadrada del número total de usuarios)

vecinos_cercanos = encontrar_vecinos_cercanos(usuario_seleccionado, sabores_pizza, nombres, k)

# Ordenar los vecinos por distancia y fuerza de vínculo
vecinos_cercanos = sorted(vecinos_cercanos, key=lambda x: (x['distancia'], x['fuerza_vinculo']), reverse=True)

# Crear el Dropdown con los nombres de los usuarios
dropdown_nombres = widgets.Dropdown(
    options=nombres,
    description='Seleccionar usuario:',
    value=usuario_seleccionado.name,
    layout=widgets.Layout(width='400px')
)

# Función para actualizar los vecinos cercanos al cambiar de nombre en el Dropdown
def actualizar_vecinos_cercanos(change):
    usuario_seleccionado = sabores_pizza.loc[change.new]
    vecinos_cercanos = encontrar_vecinos_cercanos(usuario_seleccionado, sabores_pizza, nombres, k)
    vecinos_cercanos = sorted(vecinos_cercanos, key=lambda x: (x['distancia'], x['fuerza_vinculo']), reverse=True)

    print("Usuario seleccionado:", usuario_seleccionado.name)
    print("Vecinos más cercanos:")
    for vecino in vecinos_cercanos:
        print("Nombre:", vecino['nombre'])
        print("Distancia:", vecino['distancia'])
        print("Fuerza de vínculo:", vecino['fuerza_vinculo'])

#Asociar la función de actualización al evento 'value_changed' del Dropdown
dropdown_nombres.observe(actualizar_vecinos_cercanos, names='value')

#Mostrar el Dropdown y los resultados iniciales
print("Usuario seleccionado:", usuario_seleccionado.name)
print("Vecinos más cercanos:")
for vecino in vecinos_cercanos:
    print("Nombre:", vecino['nombre'])
    print("Distancia:", vecino['distancia'])
    print("Fuerza de vínculo:", vecino['fuerza_vinculo'])

#Mostrar el dropdown inicialmente
dropdown_nombres

import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

# Generar datos aleatorios para 5 personas
personas = ['Persona 1', 'Persona 2', 'Persona 3', 'Persona 4', 'Persona 5']
peso = [random.uniform(60, 100) for _ in range(5)]  # Peso en kg (entre 60 y 100)
estatura = [random.uniform(1.50, 2.00) for _ in range(5)]  # Estatura en metros (entre 1.50 y 2.00)
edad = [random.randint(18, 70) for _ in range(5)]  # Edad en años (entre 18 y 70)

# Crear un DataFrame con los datos
data = pd.DataFrame({
    'Persona': personas,
    'Peso (kg)': peso,
    'Estatura (m)': estatura,
    'Edad (años)': edad
})

# Crear un gráfico de radar
categories = list(data.columns[1:])  # Variables a representar en el gráfico (excluyendo 'Persona')
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Cierre del gráfico

fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 8))

for i, persona in enumerate(personas):
    values = data.iloc[i, 1:].tolist()
    values += values[:1]  # Cierre del gráfico
    ax.fill(angles, values, alpha=0.25, label=persona)
    
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Comparación de Peso, Estatura y Edad')
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
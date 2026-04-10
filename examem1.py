"""
SIMULACIÓN DE TRÁFICO DE RED CON IA
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""
GENERAR DATOS SIMULADOS
"""

def generar_datos(n=300):
    np.random.seed(42)

    paquetes = np.random.randint(50, 1000, n)
    tamaño = np.random.randint(100, 5000, n)
    latencia = np.random.randint(1, 200, n)
    ancho_banda = np.random.randint(1, 100, n)

    tipo = []
    for i in range(n):
        if paquetes[i] > 800 or latencia[i] > 150:
            tipo.append(1)  # Anómalo
        else:
            tipo.append(0)  # Normal

    data = pd.DataFrame({
        "paquetes": paquetes,
        "tamaño": tamaño,
        "latencia": latencia,
        "ancho_banda": ancho_banda,
        "tipo": tipo
    })

    return data


"""
MODELO PARA LA IA
"""

data = generar_datos()

X = data[["paquetes", "tamaño", "latencia", "ancho_banda"]]
y = data["tipo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
print("Accuracy del modelo:", accuracy_score(y_test, pred))


"""
SIMULACIÓN EN TIEMPO REAL
"""

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data = []
y_data = []
z_data = []
colores = []

print("\nSimulación en tiempo real...\n")

for i in range(200):

    paquetes = np.random.randint(50, 1000)
    tamaño = np.random.randint(100, 5000)
    latencia = np.random.randint(1, 200)
    ancho_banda = np.random.randint(1, 100)

    nuevo = pd.DataFrame([[paquetes, tamaño, latencia, ancho_banda]],
                         columns=["paquetes", "tamaño", "latencia", "ancho_banda"])

    prediccion = modelo.predict(nuevo)[0]

    x_data.append(paquetes)
    y_data.append(tamaño)
    z_data.append(latencia)

    if prediccion == 0:
        colores.append("blue")
        print(f"Dato {i}: NORMAL")
    else:
        colores.append("red")
        print(f"Dato {i}: ANÓMALO")

    ax.clear()

    ax.scatter(x_data, y_data, z_data, c=colores)

    ax.set_xlabel("Paquetes")
    ax.set_ylabel("Tamaño")
    ax.set_zlabel("Latencia")

    plt.draw()
    plt.pause(0.1)

    time.sleep(0.1)

plt.ioff()
plt.show()
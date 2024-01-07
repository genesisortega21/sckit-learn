import joblib as jb
import numpy as np

model = jb.load('./SCKIT-LEARN/models/best_model_0.923.pkl')

X_test = np.array([8,65,0.001984127,23.41535119,84.20419048,20.1311627,0.196924603,0.94796627,173.0823413])  # Datos de entrada ajustados con 9 características

# Obtener las probabilidades de las clases
probabilities = model.predict(X_test.reshape(1, -1))

# Definir un umbral para decidir si es 0 o 1
umbral = 0.5

# Redondear la probabilidad de la clase 1 a 0 o 1 según el umbral
prediccion = (probabilities[0] > umbral).astype(int)

if prediccion == 1:
    print('Su plantación tiene la enfermedad ESCOBA DE BRUJA:', prediccion)
elif prediccion == 0:
    print('Su plantación no tiene enfermedad:', prediccion)

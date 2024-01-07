import joblib #importa las bibliotecas joblib para cargar el modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask
from flask import jsonify
app = Flask(__name__)
#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test =np.array([6.636723069,6.507276984,1.69227767,1.353814363,0.949492395,0.549840569,0.345965981,0.464307785,1.216362])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion' : list(prediction)})
if __name__ == "__main__":
    model = joblib.load('./SCKIT-LEARN/models/best_model.pkl')
    app.run(port=8080)
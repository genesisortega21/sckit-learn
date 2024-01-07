# Importamos las bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

if __name__ == "__main__":
    # Importamos el dataset de 2017
    dataset = pd.read_csv('./SCKIT-LEARN/in/DataFinal.csv')
    
    # Mostramos el reporte estadístico
    # print(dataset.describe())
    
    # Elegimos los features que vamos a usar
    X = dataset[['Rain', 'Temperature', 'RH', 'WindSpeed', 'WindDirection', 'FRUTO', 'SEVERIDAD']]
    y = dataset['INCIDENCIA']  # No es necesario usar doble corchete
    
    # Imprimimos los conjuntos que creamos
    # print(X.shape)
    # print(y.shape)
    
    # Partimos nuestro entrenamiento en training y test, no hay que olvidar el orden
    # Con el test size elegimos nuestro porcentaje de datos para training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Definimos nuestros regresores uno por uno y llamamos al fit o ajuste
    modelLinear = LinearRegression().fit(X_train, y_train)
    
    # Calculamos la predicción que nos da con la función predict usando la regresión lineal
    y_predict_linear = modelLinear.predict(X_test)
    
    # Configuramos alpha, que es el valor lambda; entre más valor tenga alpha en Lasso, más penalización tendremos
    # Y lo entrenamos con la función fit
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    
    # Hacemos una predicción para ver si es mejor o peor que lo que teníamos en el modelo lineal
    # sobre exactamente los mismos datos que teníamos anteriormente
    y_predict_lasso = modelLasso.predict(X_test)
    
    # Hacemos la misma predicción, pero para nuestra regresión Ridge
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    
    # Calculamos el valor predicho para nuestra regresión Ridge
    y_predict_ridge = modelRidge.predict(X_test)
    
    # Hacemos la misma predicción, pero para nuestra regresión ElasticNet
    modelElasticNet = ElasticNet(random_state=0).fit(X_train, y_train)
    
    # Calculamos el valor predicho para nuestra regresión ElasticNet
    y_pred_elastic = modelElasticNet.predict(X_test)
    
    # Calculamos la pérdida para cada uno de los modelos que entrenamos, empezando con nuestro modelo lineal,
    # usando el error medio cuadrático y aplicándolo con los datos de prueba y la predicción que hicimos
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    
    # Mostramos la pérdida lineal con la variable que acabamos de calcular
    print("Linear Loss. " + "%.10f" % float(linear_loss))
    
    # Mostramos nuestra pérdida Lasso, con la variable lasso_loss
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss. " + "%.10f" % float(lasso_loss))
    
    # Mostramos nuestra pérdida de Ridge con la variable ridge_loss
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss: " + "%.10f" % float(ridge_loss))
    
    # Mostramos nuestra pérdida de ElasticNet con la variable elastic_loss
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)
    print("ElasticNet Loss: " + "%.10f" % float(elastic_loss))
    
    # Imprimimos los coeficientes para ver cómo afecta a cada una de las regresiones
    # La línea "="*32 lo único que hará es repetir el símbolo de igual 32 veces
    print("="*32)
    print("Coeficientes linear: ")
    # Esta información la podemos encontrar en la variable coef_
    print(modelLinear.coef_)
    print("="*32)
    print("Coeficientes lasso: ")
    # Esta información la podemos encontrar en la variable coef_
    print(modelLasso.coef_)
    # Hacemos lo mismo con Ridge
    print("="*32)
    print("Coeficientes ridge:")
    print(modelRidge.coef_)
    # Hacemos lo mismo con ElasticNet
    print("="*32)
    print("Coeficientes elastic net:")
    print(modelElasticNet.coef_)
    
    # Calculamos nuestra exactitud de nuestra predicción lineal
    print("="*32)
    print("Score Lineal", modelLinear.score(X_test, y_test))
    
    # Calculamos nuestra exactitud de nuestra predicción Lasso
    print("="*32)
    print("Score Lasso", modelLasso.score(X_test, y_test))
    
    # Calculamos nuestra exactitud de nuestra predicción Ridge
    print("="*32)
    print("Score Ridge", modelRidge.score(X_test, y_test))
    
    # Calculamos nuestra exactitud de nuestra predicción Elastic Net
    print("="*32)
    print("Score ElasticNet", modelElasticNet.score(X_test, y_test))

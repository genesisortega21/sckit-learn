import pandas as pd  
import sklearn  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.decomposition import IncrementalPCA  
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA  
from sklearn.decomposition import IncrementalPCA  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
if __name__ == '__main__':
    dt_heart = pd.read_csv('./SCKIT-LEARN/in/DataFinal.csv')  
    print(dt_heart.head(5)) #imprimimos los 5 primeros datos
    # las featurus sin el target
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    dt_target = dt_heart['INCIDENCIA']  

    dt_features = StandardScaler().fit_transform(
        dt_features)  

    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, dt_target, test_size=0.30, random_state=42)  
    print(X_train.shape)  
    print(y_train.shape)
    '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará de esta forma:
    a: n_components = min(n_muestras, n_features)'''
    mayor_pca = 0
    mayor_ipca = 0
    variables_pca = 0
    variable_ipca = 0
    for variables_artificiales in range(1, 8):
        pca = PCA(n_components=variables_artificiales)
        pca.fit(X_train)
        '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
        poco a poco y combinarlos en el resultado final'''
        ipca = IncrementalPCA(n_components=variables_artificiales,
                              batch_size=10) 
        ipca.fit(X_train)
        ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
        me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
        en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
        para nuestro modelo '''
        plt.plot(range(len(pca.explained_variance_)),
                 pca.explained_variance_ratio_)  
        # plt.show()
        # Ahora vamos a configurar nuestra regresión logística
        logistic = LogisticRegression(solver='lbfgs')
        # Configuramos los datos de entrenamiento
        dt_train = pca.transform(X_train)  # conjunto de entrenamiento
        dt_test = pca.transform(X_test)  # conjunto de prueba
        # Mandamos los data frames la la regresión logística
        # mandasmos a regresion logistica los dos datasets
        logistic.fit(dt_train, y_train)
        # Calculamos nuestra exactitud de nuestra predicción

        # print("SCORE PCA: ", logistic.score(dt_test, y_test))
        if logistic.score(dt_test, y_test) > mayor_pca:
            mayor_pca = logistic.score(dt_test, y_test)
            variables_pca = variables_artificiales


        # Configuramos los datos de entrenamiento
        dt_train = ipca.transform(X_train)
        dt_test = ipca.transform(X_test)
        # Mandamos los data frames la la regresión logística
        logistic.fit(dt_train, y_train)
        # Calculamos nuestra exactitud de nuestra predicción
        # print("SCORE IPCA: ", logistic.score(dt_test, y_test))


        if logistic.score(dt_test, y_test) > mayor_ipca:
            mayor_ipca = logistic.score(dt_test, y_test)
            variables_ipca = variables_artificiales

   
    print(
        "pca: ", [mayor_pca, variables_pca],
        "ipca: ", [mayor_ipca, variables_ipca]
    )

    
    def kpca():
        zdt_heart = pd.read_csv('./SCKIT-LEARN/in/DataFinal.csv')
        # Imprimimos un encabezado con los primeros 5 registros
        # print(dt_heart.head(5))
        zdt_features = zdt_heart.drop(['INCIDENCIA'], axis=1)
        # Este será nuestro dataset, pero sin la columna
        zdt_target = zdt_heart['INCIDENCIA']
        # Normalizamos los datos
        zdt_features = StandardScaler().fit_transform(zdt_features)
        zX_train, zX_test, zy_train, zy_test = train_test_split(zdt_features, zdt_target,
                                                            test_size=0.3, random_state=42)
        zkernel = ['linear', 'poly', 'rbf']
        # Aplicamos la función de kernel de tipo polinomial
        mayor_linear = 0
        mayor_poly = 0
        mayor_rbf = 0
        mayor_linear_componentes = 0
        mayor_poly_componentes = 0
        mayor_rbf_componentes = 0

        for k in zkernel:
            for i in range(1, 8):
                zkpca = KernelPCA(n_components=i, kernel=k)
                # kpca = KernelPCA(n_components=4, kernel='poly' )
                # Vamos a ajustar los datos
                zkpca.fit(zX_train)
                # Aplicamos el algoritmo a nuestros datos de prueba y de entrenamiento
                zdt_train = zkpca.transform(zX_train)
                zdt_test = zkpca.transform(zX_test)
                # Aplicamos la regresión logística un vez que reducimos su
                # dimensionalidad
                zlogistic = LogisticRegression(solver='lbfgs')
                # Entrenamos los datos
                zlogistic.fit(zdt_train, zy_train)
                # Imprimimos los resultados
                if k == "linear":
                    if zlogistic.score(zdt_test, zy_test) >= mayor_linear:
                        mayor_linear = zlogistic.score(zdt_test, zy_test)
                        mayor_linear_componentes = i
                if k == "poly":
                    if zlogistic.score(zdt_test, zy_test) >= mayor_poly:
                        mayor_poly = zlogistic.score(zdt_test, zy_test)
                        mayor_poly_componentes = i
                if k == "rbf":
                    if zlogistic.score(zdt_test, zy_test) >= mayor_rbf:
                        mayor_rbf = zlogistic.score(zdt_test, zy_test)
                        mayor_rbf_componentes = i

                # print("SCORE KPCA " + k + " : ", logistic.score(dt_test, y_test))
        print(
            "Kpca->linear: ", [mayor_linear, mayor_linear_componentes],
             "poly: ", [mayor_poly, mayor_poly_componentes],
             "rbf: ", [mayor_rbf, mayor_rbf_componentes]
        )
    kpca()

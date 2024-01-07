import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
if __name__ == "__main__":
    dataset = pd.read_csv('./SCKIT-LEARN/in/DataFinal.csv')
    #print(dataset.head(10))
    X = dataset.fillna(0)
    X = dataset.drop('INCIDENCIA', axis=1)

    #Selecionamos 3 grupos
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=8).fit(X)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))
    #K- means
    dataset['group'] = kmeans.predict(X)
    dataset.to_csv("ready.csv", index=False)
    print(dataset)
    sns.pairplot(dataset[['SEVERIDAD','Rain','group']],
    hue = 'group')

    plt.show()
   
import pandas as pd
import numpy as np
import sklearn
from utils import Utils
from models import Models
if __name__ == "__main__":
    utils = Utils()
    models = Models()
    data = utils.load_from_csv('./SCKIT-LEARN/in/DataFinal.csv')
    X, y = utils.features_target(data, ['INCIDENCIA'],['INCIDENCIA'])
    models.grid_training(X,y)
    print(data)

import numpy as np

from data_prep import features, targets, features_test, targets_tests

def sigmoide(x):
    return 1/(1 + np.exp(-x))

#Hyperparameters
n_hiden = 2 #número de unidades en la capa escondida
epochs = 1000 #número de iteraciones sobre el conjunto de entrenamiento
alpha = 0.05 #Taza de aprendizaje

m, k = features.shape #número de ejemplos de entrenamiento, número de dimensiones en los datos
#inicialización de los pesos
entrada_escondida = np.random.normal(scale = 1/k**0.5
                                     size = (k, n_hidden)
                                      )
escondida_salida = np.random. normal(scale = 1/k**0.5,
                                     size = n_hidden
                                     )

#Entrenamiento
for e in range (epochs): 
    #variables para el gradiente
    gradiente_entrada_escondida = np.zeros(entrada_escondida.shape)
    gradiente_escondida_salida = np.zeros(escondida_salida.shape)
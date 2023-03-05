import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

a_renomear = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    'unfinished': 'nao_finalizados'
}

dados = dados.rename(columns=a_renomear)
#print(dados.head())
troca = {
    0: 1,
    1: 0
}
dados['finalizado'] = dados.nao_finalizados.map(troca)
#print(dados.tail())

#sns.scatterplot(x="horas_esperadas", y="preco", data=dados)
#sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)
#sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data=dados)
#plt.show()

x = dados[['horas_esperadas','preco']]
y = dados['finalizado']

SEED = 5

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

#modelo = LinearSVC()
#modelo.fit(treino_x, treino_y)
#previsoes = modelo.predict(teste_x)
#
#acuracia = accuracy_score(teste_y, previsoes) * 100
#print("A acurácia foi %.2f%%" % acuracia)
#
#previsoes_de_base = np.ones(540)
#
#acuracia = accuracy_score(teste_y, previsoes_de_base) * 100
#print("A acurácia do algoritmo de previsoes de baseline foi %.2f%%" % acuracia)

#sns.relplot(x="horas_esperadas", y="preco", hue=teste_y, data=teste_x)
#plt.show()


modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
print(x_min, x_max,y_min,y_max)

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min)/pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min)/pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
xx.ravel()
pontos = np.c_[xx.ravel(), yy.ravel()]
print(pontos)

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)
print(Z)


#plt.contourf(xx, yy, Z, alpha=0.3)
#plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)


raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC(gamma='auto')
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

data_x = teste_x[:,0]
data_y = teste_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data_x, data_y, c=teste_y, s=3)
plt.show()
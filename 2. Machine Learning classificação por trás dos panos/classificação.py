import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

dados = pd.read_csv(r"C:\Users\willi\Downloads\Customer-Churn.csv")
dados.head()

traducao_dic = {'Sim': 1, 
                'Nao': 0}

dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)
dadosmodificados.head()

#transformação pelo get_dummies
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'],
                axis=1))

dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)

dados_final.head()


pd.set_option('display.max_columns', 39)

dados_final.head()

Xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

x = dados_final.drop('Churn', axis = 1)
y = dados_final['Churn']

norm = StandardScaler()

x_normalizado = norm.fit_transform(x)
#print(x_normalizado)

#print(x_normalizado[0])

Xmaria_normalizado = norm.transform(pd.DataFrame(Xmaria, columns = x.columns))
#print(Xmaria_normalizado)

"Distancia euclidiana"

a = Xmaria_normalizado

b = x_normalizado[0]

#print(a - b)
#print(np.square(a-b))
#print(np.sum(np.square(a-b)))
#print(np.sqrt(84.07574038273466))

x_treino, x_teste, y_treino, y_teste = train_test_split(x_normalizado, y, test_size=0.3, random_state=123)

knn = KNeighborsClassifier(metric='euclidean')

knn.fit(x_treino, y_treino)

predict_knn = knn.predict(x_teste)

print(predict_knn)

#print(x_treino)

#print(y_treino)

print(np.median(x_treino))

bnb = BernoulliNB(binarize=0.52)

bnb.fit(x_treino, y_treino)

predict_bnb = bnb.predict(x_teste)

print(predict_bnb)


acuracia = accuracy_score(y_teste, predict_bnb) * 100
print("A acurácia foi %.2f%%" % acuracia)

dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)

dtc.fit(x_treino, y_treino)

print(dtc.feature_importances_)

predict_arvoredecisao = dtc.predict(x_teste)

print(predict_arvoredecisao)

#print(confusion_matrix(y_teste, predict_knn))
#print(confusion_matrix(y_teste, predict_bnb))
#print(confusion_matrix(y_teste, predict_arvoredecisao))

#print(accuracy_score(y_teste, predict_knn))
#print(accuracy_score(y_teste, predict_bnb))
#print(accuracy_score(y_teste, predict_arvoredecisao))

#print(precision_score(y_teste , predict_knn))
#print(precision_score(y_teste , predict_bnb))
#print(precision_score(y_teste , predict_arvoredecisao))



#print(recall_score(y_teste, predict_knn))

#print(recall_score(y_teste, predict_bnb))

#print(recall_score(y_teste, predict_arvoredecisao))

predict_alura = [0,0,0,0,1,1,1,1,0,1,0,1]
alura_real = [1,1,0,0,1,1,1,0,1,0,1,0]

print(recall_score(alura_real, predict_alura))

print(f1_score(alura_real, predict_alura))
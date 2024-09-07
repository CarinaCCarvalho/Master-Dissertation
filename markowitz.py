import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from pulp import LpMinimize

# Parametros
d1=126
d2=63
tau= 1
#dimensao
dimensao=3
#v
nu = 21
r_star = (1 +0.02)**(1/252)-1 
#datas
#inicio="2005-01-01" #setembro ate marco 2016 e prevê ate fim de junho
#fim="2018-11-01"

#indice="^DJI"

nome_arquivo_csv = "C:/Users/acari/Desktop/Tese/dowjonesartigo.csv"
data = pd.read_csv(nome_arquivo_csv)


lista_acoes=data.columns[2:]


precos_fechamento_indice=np.array(data[data.columns[1]].values.tolist())

#____________________________________________________________________________________________________________________________________________


#MODELO MARKOWITZ

def modelo_portefoliomarkowitz_completo(inicio,fim):
    n=len(lista_acoes)

    def calcular_retornos(prices):
            prices = np.array(prices)
            # Calcula os retornos logarítmicos
            return np.log(prices[1:] / prices[:-1])

    retornos = []
    precos_fechamentos=[]
    for acao in lista_acoes:
            print(acao)
            precos_fechamentos.append(np.array(data[acao].values.tolist()))
            retornos.append(calcular_retornos(precos_fechamentos[-1]))


    # matriz de covariância
    retornos=np.array(retornos)[:,inicio:inicio+d1]
    Q = np.cov(retornos, rowvar=True)
    #print (np.shape(Q))
    x = cp.Variable(n)
    r= np.log(1.02)/252 #0.02 0.2 e 2 n da
    mu=np.mean(retornos, axis=1)


    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Q)),
                    [mu @ x >= r,
                    x >= 0,
                    cp.sum(x) == 1])
    
    prob.solve()

        # Print result.
    #print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    

    z=np.array([x.value[i]/precos_fechamentos[i][inicio+d1] for i in range(n)])
   
    indicefuturomarkowitz=[]
    indicereal=[]
    for dia in range(inicio+d1,min(inicio+d1+d2,fim)):
        precos=np.array([precos_fechamentos[i][dia] for i in range(n)])
        indicefuturomarkowitz.append(np.matmul(z,precos))
        indicereal.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return(np.array(indicefuturomarkowitz),np.array(indicereal))

def calcula_portefoliomarkowitz_para_todas_as_janelas(inicio, fim, d1, d2):
        indicefuturomarkowitz_total = [1]
        indicereal_total = [1]
            
        while inicio + d1<= fim:  
            indicefuturomarkowitz, indicereal = modelo_portefoliomarkowitz_completo(inicio,fim)
            indicefuturomarkowitz_total.extend(indicefuturomarkowitz*indicefuturomarkowitz_total[-1])
            indicereal_total.extend(indicereal*indicereal_total[-1])
            inicio += d2
        return (indicefuturomarkowitz_total, indicereal_total)

indicefuturomarkowitz_total, indicereal_total = calcula_portefoliomarkowitz_para_todas_as_janelas(0, len(precos_fechamento_indice), d1, d2)


plt.figure(figsize=(30, 10))
plt.plot(np.log(indicefuturomarkowitz_total)/np.log(10), label='Índice Modelo Markowitz')
plt.plot(np.log(indicereal_total)/np.log(10), label='Índice real')
plt.title('Comportamento dos índices')
plt.xlabel('Dias')
plt.ylabel('Retornos Cumulativos')
plt.legend()
plt.show()

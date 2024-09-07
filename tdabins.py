import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ripser import Rips
import pandas as pd
import time

from pulp import LpVariable, LpProblem, lpSum, LpMinimize
from datetime import datetime, timedelta
from openpyxl import load_workbook
from scipy.stats import skew
from scipy.stats import kurtosis


inicio = time.time()
# Parametros

d1= 63# 63 #126
d2= 21#21 #63
tau= 1
#dimensao
dimensao=4
#v
nu = 21
r_star = np.log(1.02)/252 

nome_arquivo_csv = "C:/Users/acari/Desktop/Tese/sp500.csv"

data = pd.read_csv(nome_arquivo_csv)

lista_acoes=data.columns[2:]



#data= yf.download(indice, start=inicio, end=fim)
precos_fechamento_indice=np.array(data[data.columns[1]].values.tolist()) 

def calcular_retornos_logaritmicos(prices):
    prices = np.array(prices)
    # Calcula os retornos logarítmicos
    return np.log(prices[1:] / prices[:-1])


# Converte a lista de preços para um vetor
precos_array = np.array(precos_fechamento_indice)
# Calcula os retornos logarítmicos
retornos_log = calcular_retornos_logaritmicos(precos_array)
 
def incorporacao_takens(serie_temporal, dimensao, tau):
    #d1=126
    num_pontos = d1 - (dimensao - 1) * tau
    dados_multidimensionais = np.zeros((num_pontos, dimensao))

    for i in range(num_pontos):
        dados_multidimensionais[i, :] = serie_temporal [i: i+tau * dimensao : tau] #vai de tau em tau, comeca em i ate i*tau+dim
    return dados_multidimensionais



serie_temporal = np.array(retornos_log)
dados_multidimensionais = incorporacao_takens(serie_temporal, dimensao, tau)
#print("Dados Multidimensionais:\n", dados_multidimensionais)

def criar_nuvem_pontos(serie_temporal, dimensao, tau, nu):
    num_pontos = nu - (dimensao - 1) * tau
    nuvem_pontos = np.zeros((num_pontos, dimensao))

    for i in range(num_pontos):
        nuvem_pontos[i, :] = serie_temporal[i:i+ tau * dimensao : tau]

    return nuvem_pontos


nuvem_pontos = criar_nuvem_pontos(serie_temporal, dimensao, tau,nu)
#print("Nuvem de Pontos:\n", nuvem_pontos)

# Calcular a norma L1
def norma_L1(lista):
    soma=0
    for ponto in lista: 
        soma=soma+((ponto[1]-ponto[0])/2)**2
    return soma


rips = Rips(maxdim=2)


def serie_normas_TDA(precos):
    normas=[]
    retornos_log= calcular_retornos_logaritmicos(precos)
    serie_temporal = np.array(retornos_log)
    for i in range(len(serie_temporal)-nu+1):  
        nuvem_pontos = criar_nuvem_pontos(serie_temporal[i:], dimensao, tau, nu)
        dim= rips.fit_transform(nuvem_pontos)
        norma = norma_L1(dim[1])
        normas.append(norma)
    return (normas)

#Norma indice de referencia
Nchapeu=serie_normas_TDA(precos_fechamento_indice)

Nacoes=len(lista_acoes) #aqui
df=[]
for i in range(Nacoes):
    df.append([0,0])
N=[]
precos_fechamentos=[]


#cria as normas de todas as açoes
for acao in lista_acoes:
    print(acao)
    #data= yf.download(acao, start=inicio, end=fim)
    precos_fechamentos.append(np.array(data[acao].values.tolist()))
    N.append(serie_normas_TDA(precos_fechamentos[-1]))
#print(len(precos_fechamentos[-1])) #p ver o tamanho
    
 # Dar nomes às categorias
nome_bin1= "Bin1"
nome_bin2 = "Bin2"
nome_bin3 = "Bin3"

def calcula_bins(inicio):
    #Calculo dfi
    def calcula_df(inicio):
        for i in range(Nacoes):
            df[i]=[N[i][inicio+d1-nu]-np.mean(N[i][inicio:inicio+d1-nu+1]),i]
    calcula_df(inicio)
    df_ordenada = sorted(df, key=lambda x: x[0])
    print(df_ordenada)

    tamanho_total = len(df_ordenada)
    parte = tamanho_total // 3

    # Dividir em três categorias
    bin1=df_ordenada[tamanho_total-parte:] #ESTE BIN1 É O BIN2 NO ARTIGO!!!!!
    bin2=df_ordenada[:parte] #os mais pequenos
    bin3=df_ordenada[parte:tamanho_total-parte]
 


    # Exemplo de impressão dos tamanhos e nomes das categorias
    print(f"Tamanho da Categoria {nome_bin1}: {len(bin1)}")
    print(f"Tamanho da Categoria {nome_bin2}: {len(bin2)}")
    print(f"Tamanho da Categoria {nome_bin3}: {len(bin3)}")

    print("bin1:", bin1)
    print("bin2:", bin2)
    print("bin3:", bin3)
    return(bin1,bin2,bin3)

bin1,bin2,bin3=calcula_bins(0)
#no artigo
#bin1 mais baixo -bin3- e bin2 em crise q nao queremos
#trocaram bin1 com bin2


#CONTAR OS RETORNOS MAIORES SEM SBTDA

#bin2 sao os debaixo da media-diferenças menores
#bin3 a volta da media
#bin1 acima da media- maior norma tda tem mais volatilidade- açoes em crise-risco crash



def media_retornos(precos):
    retornos_log= calcular_retornos_logaritmicos(precos)
    return np.mean(retornos_log)

def calcula_portefolio_TDA1(inicio,fim):

    bin1,bin2,bin3=calcula_bins(inicio)
    mu=[]
    Bin=bin1
    n=len(Bin)
    print("ESTE É O N", n)

    componente=[Bin[i][1] for i in range(n)] #componentes do bin

    for i in range(n):
        mu.append(media_retornos(precos_fechamentos[componente[i]][inicio:inicio+d1]))

    mu_I = np.mean(retornos_log[inicio:inicio+d1]) # retornos esperados do índice de referência

    prob = LpProblem("Otimizacao", LpMinimize)

    # Criar variáveis de decisão
    w = [LpVariable(f"w_{i}", lowBound=0) for i in range(n)]

    # Criar variável de ajuda
    intermediate_expr = [[LpVariable(f"intermediate_{i},{t}") for t in range(d1-(nu-1))] for  i in range (n)]


    # Função objetivo
    prob += lpSum([intermediate_expr[i][t] for i in range(n) for t in range(d1-(nu-1))]), "Funcao_Objetivo"
    #print(prob)
    # Função objetivo
    for i in range(n):
        for t in range(d1-(nu-1)):
            prob += intermediate_expr[i][t] >= (N[componente[i]][inicio+t] * w[i] - Nchapeu[inicio+t])
            prob += intermediate_expr[i][t] >= -(N[componente[i]][inicio+t] * w[i] - Nchapeu[inicio+t])


    # Restrições
    prob += lpSum([mu[i] * w[i] for i in range(n)]) - mu_I >= r_star, "Restricao_1"
    prob += lpSum(w) == 1, "Restricao_2"

    #print(prob.objective)

    # Resolver o problema
    prob.solve()
    s=0
    # Mostrar os resultados
    print("Status:", prob.status)
    print("Valor ótimo da função objetivo:", round(prob.objective.value(),8))
    for i, var in enumerate(w):
        print(f"Valor ótimo de {var.name}:", round(var.value(),8))
        s+=var.value()
    print(s)

    #quantas acoes de cada tipo estou a investir p 1 dolar
    # wi proporcao do investimento na açcao i
    y=np.array([w[i].value()/precos_fechamentos[componente[i]][inicio+d1] for i in range(n)])
    print(y)

    #soma_portefolioinicial = 0
    #for i in range(n):
    #    soma_portefolioinicial += y[i]*precos_fechamentos[componente[i]][d1] 
    #print(soma_portefolioinicial)    

    #comportamento do indice que nos criamos ate d1 nos dias a seguir d1+d2
    indicefuturo1=[]
    indicereal=[]
    for dia in range(inicio+d1,min(inicio+d1+d2, fim)):
        precos=np.array([precos_fechamentos[componente[i]][dia] for i in range(n)])
        indicefuturo1.append(np.matmul(y,precos))
        indicereal.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return(np.array(indicefuturo1),np.array(indicereal))




def calcula_portefolio1_para_todas_as_janelas(inicio, fim, d1, d2):
    indicefuturo1_total = [1]
    indicereal_total = [1]
    
    while inicio + d1<= fim:  
        indicefuturo1, indicereal = calcula_portefolio_TDA1(inicio,fim)
        indicefuturo1_total.extend(indicefuturo1*indicefuturo1_total[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo1_total,indicereal_total)

indicefuturo1_total, indicereal_total = calcula_portefolio1_para_todas_as_janelas(0,len(precos_fechamento_indice), d1, d2)





def calcula_portefolio_TDA2(inicio,fim):

    bin1,bin2,bin3=calcula_bins(inicio)
    mu=[]
    Bin=bin2
    n=len(Bin)


    componente=[Bin[i][1] for i in range(n)] #componentes do bin

    for i in range(n):
        mu.append(media_retornos(precos_fechamentos[componente[i]][inicio:inicio+d1]))

    mu_I = np.mean(retornos_log[inicio:inicio+d1]) # retornos esperados do índice de referência

    prob = LpProblem("Otimizacao", LpMinimize)

    # Criar variáveis de decisão
    w = [LpVariable(f"w_{i}", lowBound=0) for i in range(n)]

    # Criar variável de ajuda
    intermediate_expr = [[LpVariable(f"intermediate_{i},{t}") for t in range(d1-(nu-1))] for  i in range (n)]


    # Função objetivo
    prob += lpSum([intermediate_expr[i][t] for i in range(n) for t in range(d1-(nu-1))]), "Funcao_Objetivo"
    #print(prob)
    # Função objetivo
    for i in range(n):
        for t in range(d1-(nu-1)):
            prob += intermediate_expr[i][t] >= (N[componente[i]][inicio+t] * w[i] - Nchapeu[inicio+t])
            prob += intermediate_expr[i][t] >= -(N[componente[i]][inicio+t] * w[i] - Nchapeu[inicio+t])


    # Restrições
    prob += lpSum([mu[i] * w[i] for i in range(n)]) - mu_I >= r_star, "Restricao_1"
    prob += lpSum(w) == 1, "Restricao_2"

    #print(prob.objective)

    # Resolver o problema
    prob.solve()
    s=0
    # Mostrar os resultados
    print("Status:", prob.status)
    print("Valor ótimo da função objetivo:", round(prob.objective.value(),8))
    for i, var in enumerate(w):
        print(f"Valor ótimo de {var.name}:", round(var.value(),8))
        s+=var.value()
    print(s)

    #quantas acoes de cada tipo estou a investir p 1 dolar
    # wi proporcao do investimento na açcao i
    y=np.array([w[i].value()/precos_fechamentos[componente[i]][inicio+d1] for i in range(n)])
    print(y)

    #soma_portefolioinicial = 0
    #for i in range(n):
    #    soma_portefolioinicial += y[i]*precos_fechamentos[componente[i]][d1] 
    #print(soma_portefolioinicial)    

    #comportamento do indice que nos criamos ate d1 nos dias a seguir d1+d2
    indicefuturo2=[]
    indicereal=[]
    for dia in range(inicio+d1,min(inicio+d1+d2, fim)):
        precos=np.array([precos_fechamentos[componente[i]][dia] for i in range(n)])
        indicefuturo2.append(np.matmul(y,precos))
        indicereal.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return(np.array(indicefuturo2),np.array(indicereal))




def calcula_portefolio2_para_todas_as_janelas(inicio, fim, d1, d2):
    indicefuturo2_total = [1]
    indicereal_total = [1]
    
    while inicio + d1 <= fim:  
        indicefuturo2, indicereal = calcula_portefolio_TDA2(inicio,fim)
        indicefuturo2_total.extend(indicefuturo2*indicefuturo2_total[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo2_total,indicereal_total)

indicefuturo2_total, indicereal_total = calcula_portefolio2_para_todas_as_janelas(0,len(precos_fechamento_indice), d1, d2)






def calcula_portefolio_TDA3(inicio,fim):

    bin1,bin2,bin3=calcula_bins(inicio)
    mu=[]
    Bin=bin3
    n=len(Bin)


    componente=[Bin[i][1] for i in range(n)] #componentes do bin

    for i in range(n):
        mu.append(media_retornos(precos_fechamentos[componente[i]][inicio:inicio+d1]))

    mu_I = np.mean(retornos_log[inicio:inicio+d1]) # retornos esperados do índice de referência

    prob = LpProblem("Otimizacao", LpMinimize)

    # Criar variáveis de decisão
    w = [LpVariable(f"w_{i}", lowBound=0) for i in range(n)]

    # Criar variável de ajuda
    intermediate_expr = [[LpVariable(f"intermediate_{i},{t}") for t in range(d1-(nu-1))] for  i in range (n)]


    # Função objetivo
    prob += lpSum([intermediate_expr[i][t] for i in range(n) for t in range(d1-(nu-1))]), "Funcao_Objetivo"
    #print(prob)
    # Função objetivo
    for i in range(n):
        for t in range(d1-(nu-1)):
            prob += intermediate_expr[i][t] >= (N[componente[i]][inicio+t] * w[i] - Nchapeu[inicio+t])
            prob += intermediate_expr[i][t] >= -(N[componente[i]][inicio+t] * w[i] - Nchapeu[inicio+t])


    # Restrições
    prob += lpSum([mu[i] * w[i] for i in range(n)]) - mu_I >= r_star, "Restricao_1"
    prob += lpSum(w) == 1, "Restricao_2"

    #print(prob.objective)

    # Resolver o problema
    prob.solve()
    s=0
    # Mostrar os resultados
    print("Status:", prob.status)
    print("Valor ótimo da função objetivo:", round(prob.objective.value(),8))
    for i, var in enumerate(w):
        print(f"Valor ótimo de {var.name}:", round(var.value(),8))
        s+=var.value()
    print(s)

    #quantas acoes de cada tipo estou a investir p 1 dolar
    # wi proporcao do investimento na açcao i
    y=np.array([w[i].value()/precos_fechamentos[componente[i]][inicio+d1] for i in range(n)])
    print(y)

    #soma_portefolioinicial = 0
    #for i in range(n):
    #    soma_portefolioinicial += y[i]*precos_fechamentos[componente[i]][d1] 
    #print(soma_portefolioinicial)    

    #comportamento do indice que nos criamos ate d1 nos dias a seguir d1+d2
    indicefuturo3=[]
    indicereal=[]
    for dia in range(inicio+d1,min(inicio+d1+d2, fim)):
        precos=np.array([precos_fechamentos[componente[i]][dia] for i in range(n)])
        indicefuturo3.append(np.matmul(y,precos))
        indicereal.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return(np.array(indicefuturo3),np.array(indicereal))




def calcula_portefolio3_para_todas_as_janelas(inicio, fim, d1, d2):
    indicefuturo3_total = [1]
    indicereal_total = [1]
    
    while inicio + d1 <= fim:  
        indicefuturo3, indicereal = calcula_portefolio_TDA3(inicio,fim)
        indicefuturo3_total.extend(indicefuturo3*indicefuturo3_total[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo3_total,indicereal_total)

indicefuturo3_total, indicereal_total = calcula_portefolio3_para_todas_as_janelas(0,len(precos_fechamento_indice), d1, d2)


#SEM BINS e para tudo

def calcula_portefolio_TDA_sembins(inicio,fim):
    mu=[]
    n=len(lista_acoes)

    for i in range(n):
       mu.append(media_retornos(precos_fechamentos[i][inicio:inicio+d1]))
    mu_I = np.mean(retornos_log[inicio:inicio+d1]) # retornos esperados do índice de referência

    prob = LpProblem("Otimizacao", LpMinimize)

    # Criar variáveis de decisão
    w = [LpVariable(f"w_{i}",lowBound=0) for i in range(n)]

    # Criar variável de ajuda
    intermediate_expr = [[LpVariable(f"intermediate_{i},{t}") for t in range(d1-(nu-1))] for  i in range (n)]


    # Função objetivo
    prob += lpSum([intermediate_expr[i][t] for i in range(n) for t in range(d1-(nu-1))]), "Funcao_Objetivo"
    # Função objetivo
    for i in range(n):
        for t in range(d1-(nu-1)):
            prob += intermediate_expr[i][t] >= (N[i][inicio+t] * w[i] - Nchapeu[inicio+t])
            prob += intermediate_expr[i][t] >= -(N[i][inicio+t] * w[i] - Nchapeu[inicio+t])

    # Restrições
    prob += lpSum([mu[i] * w[i] for i in range(n)]) - mu_I >= r_star, "Restricao_1"
    prob += lpSum(w) == 1, "Restricao_2"

    # Resolver o problema
    prob.solve()
    s=0
  
    # Mostrar os resultados
    print("Status:", prob.status)
    print("Valor ótimo da função objetivo:", round(prob.objective.value(),8))
    for i, var in enumerate(w):
        print(f"Valor ótimo de {var.name}:", round(var.value(),8))
        s+=var.value()
    print(s)
   
    #quantas acoes de cada tipo estou a investir p 1 dolar
    # wi proporcao do investimento na açcao i
    y=np.array([w[i].value()/precos_fechamentos[i][inicio+d1] for i in range(n)])

    #comportamento do indice que nos criamos ate d1 nos dias a seguir d1+d2
    indicefuturo_sembins=[]
    for dia in range(inicio+d1,min(inicio+d1+d2, fim)):
        precos=np.array([precos_fechamentos[i][dia] for i in range(n)])
        indicefuturo_sembins.append(np.matmul(y,precos))
    return(np.array(indicefuturo_sembins))



def calcula_portefolio_para_todas_as_janelas_TDA_sembins(inicio, fim, d1, d2):
    indicefuturo_sembins_total = [1]
    
    while inicio + d1 <= fim:  
        indicefuturo_sembins= calcula_portefolio_TDA_sembins(inicio,fim)
        indicefuturo_sembins_total.extend(indicefuturo_sembins*indicefuturo_sembins_total[-1])
        inicio += d2
    return (indicefuturo_sembins_total)

indicefuturo_sembins_total= calcula_portefolio_para_todas_as_janelas_TDA_sembins(0,len(precos_fechamento_indice), d1, d2)


fim = time.time()

# Calcular e mostrar o tempo de execução
tempo_execucao = fim - inicio
print(f"O programa demorou {tempo_execucao:.4f} segundos para executar.")


#SEM BINS e para tudo

plt.figure(figsize=(30, 10))
plt.plot(np.log(indicefuturo1_total), label='TDA1')
plt.plot(np.log(indicefuturo2_total), label='TDA2')
plt.plot(np.log(indicefuturo3_total), label='TDA3')
plt.plot(np.log(indicefuturo_sembins_total), label='SBTDA')
plt.plot(np.log(indicereal_total), label='S&P 500')
plt.title('Comportamento dos índices')
plt.xlabel('Dias')
plt.ylabel('Retornos Cumulativos')
plt.legend()
plt.show()



retornos_logaritmicos_indicefuturobin1_total = calcular_retornos_logaritmicos(indicefuturo1_total[1:])
retornos_logaritmicos_indicefuturobin2_total = calcular_retornos_logaritmicos(indicefuturo2_total[1:])
retornos_logaritmicos_indicefuturobin3_total = calcular_retornos_logaritmicos(indicefuturo3_total[1:])
retornos_logaritmicos_indicefuturosembins_total= calcular_retornos_logaritmicos(indicefuturo_sembins_total[1:])

# Analise estatistica
def media(w):
    return np.mean(w)
media1_tda=media(retornos_logaritmicos_indicefuturobin1_total)
media2_tda=media(retornos_logaritmicos_indicefuturobin2_total)
media3_tda=media(retornos_logaritmicos_indicefuturobin3_total)
media_tda_sembins=media(retornos_logaritmicos_indicefuturosembins_total)
print("Média: ",round(media1_tda,8), "&",round(media2_tda,8), "&",round(media3_tda,8) ,"&",round(media_tda_sembins,8),"&",)



def desvio_padrao(w):
    return np.std(w)
desvio_padrao1 = desvio_padrao(retornos_logaritmicos_indicefuturobin1_total)
desvio_padrao2= desvio_padrao(retornos_logaritmicos_indicefuturobin2_total)
desvio_padrao3=desvio_padrao(retornos_logaritmicos_indicefuturobin3_total)
desvio_padrao_sb=desvio_padrao(retornos_logaritmicos_indicefuturosembins_total)
print("Desvio Padrão:", round(desvio_padrao1,6), "&",round(desvio_padrao2,5),"&", round(desvio_padrao3,6), "&",round(desvio_padrao_sb,5),"&")


def EMR(w,I):
    return  np.mean(w) - np.mean(I)
emr1=EMR(retornos_logaritmicos_indicefuturobin1_total, retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin1_total)])
emr2=EMR(retornos_logaritmicos_indicefuturobin2_total, retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin2_total)])
emr3=EMR(retornos_logaritmicos_indicefuturobin3_total, retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin3_total)])
emr4=EMR(retornos_logaritmicos_indicefuturosembins_total, retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturosembins_total)])
print("EMR:", round(emr1,5),"&",round(emr2,5),"&",round(emr3,5),"&", round(emr4,5),"&")




##DD-Desvio Negativo (DN)
def downside_deviation(returns, r_star):
    r_star=0
    abaixorreturns = [r for r in returns if r < r_star]
    deviations = [(r - r_star) ** 2 for r in abaixorreturns]
    dd = np.sqrt(sum(deviations) / len(returns))
    return dd


dd1 = downside_deviation(retornos_logaritmicos_indicefuturobin1_total, 0)
dd2 = downside_deviation(retornos_logaritmicos_indicefuturobin2_total, 0)
dd3 = downside_deviation(retornos_logaritmicos_indicefuturobin3_total, 0)
dd4 = downside_deviation(retornos_logaritmicos_indicefuturosembins_total, 0)
print("Downside Deviation:", round(dd1,6), "&",round(dd2,6),"&", round(dd3,6),"&", round(dd4,6),"&")


def sortino(returns, downside_deviation):
    mean_return = np.mean(returns)
    rf=0
    if mean_return > rf:
        sortino= (mean_return - rf)/ downside_deviation
        return sortino
    else:
        return 0
sortino1 = sortino(retornos_logaritmicos_indicefuturobin1_total,dd1)
sortino2= sortino(retornos_logaritmicos_indicefuturobin2_total,dd2)
sortino3=sortino(retornos_logaritmicos_indicefuturobin3_total,dd3)
sortino4=sortino(retornos_logaritmicos_indicefuturosembins_total,dd4)
print("Sortino:", round(sortino1,5) ,"&", round(sortino2,5), "&",round(sortino3,5), "&",round(sortino4,5),"&")



def VaR(w,alfa):
    return(np.percentile(w, alfa * 100))
var1=VaR(-retornos_logaritmicos_indicefuturobin1_total,0.95)
var2=VaR(-retornos_logaritmicos_indicefuturobin2_total,0.95)
var3=VaR(-retornos_logaritmicos_indicefuturobin3_total,0.95)
var4=VaR(-retornos_logaritmicos_indicefuturosembins_total,0.95)
print("var funcao (0.95):", round(var1,5),"&", round(var2,5), "&",round(var3,5), "&",round(var4,5),"&")
var5=VaR(-retornos_logaritmicos_indicefuturobin1_total,0.97)
var6=VaR(-retornos_logaritmicos_indicefuturobin2_total,0.97)
var7=VaR(-retornos_logaritmicos_indicefuturobin3_total,0.97)
var8=VaR(-retornos_logaritmicos_indicefuturosembins_total,0.97)
print("var funcao (0.97):", round(var5,5), "&",round(var6,5),"&", round(var7,5), "&",round(var8,5),"&")



def CVaR(w,alfa):
    return(np.mean(w[w > VaR(w,alfa)]))
cvar1=CVaR(-retornos_logaritmicos_indicefuturobin1_total,0.95)
cvar2=CVaR(-retornos_logaritmicos_indicefuturobin2_total,0.95)
cvar3=CVaR(-retornos_logaritmicos_indicefuturobin3_total,0.95)
cvar4=CVaR(-retornos_logaritmicos_indicefuturosembins_total,0.95)
print("cvar funcao (0.95):", round(cvar1,5), "&",round(cvar2,5), "&",round(cvar3,5), "&",round(cvar4,5),"&")
bbb=CVaR(-retornos_logaritmicos_indicefuturobin1_total,0.97)
bbb1=CVaR(-retornos_logaritmicos_indicefuturobin2_total,0.97)
bbb2=CVaR(-retornos_logaritmicos_indicefuturobin3_total,0.97)
bbb3=CVaR(-retornos_logaritmicos_indicefuturosembins_total,0.97)
print("cvar funcao (0.97):", round(bbb,5), "&",round(bbb1,5),"&",round(bbb2,5),"&",round(bbb3,5),"&")




def RS(w,I,alfa1,alfa2):
   return(CVaR(w-I,alfa1)/CVaR(-(w-I),alfa2))

# ganhos a dividiir pelas perdas, ganhos 1 
r1= RS(retornos_logaritmicos_indicefuturobin1_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin1_total)],0.95,0.95)
r2 = RS(retornos_logaritmicos_indicefuturobin2_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin2_total)],0.95,0.95)
r3 = RS(retornos_logaritmicos_indicefuturobin3_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin3_total)],0.95,0.95)
r4 = RS(retornos_logaritmicos_indicefuturosembins_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturosembins_total)],0.95,0.95)
print("Razão de Rachev 0.95,0.95:", round(r1,5),"&",round(r2,5),"&",round(r3,5),"&",round(r4,5),"&")

# ganhos a dividiir pelas perdas, ganhos 1 
r11= RS(retornos_logaritmicos_indicefuturobin1_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin1_total)],0.97,0.97)
r22 = RS(retornos_logaritmicos_indicefuturobin2_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin2_total)],0.97,0.97)
r33 = RS(retornos_logaritmicos_indicefuturobin3_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin3_total)],0.97,0.97)
r44 = RS(retornos_logaritmicos_indicefuturosembins_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturosembins_total)],0.97,0.97)
print("Razão de Rachev 0.97,0.97:", round(r11,5),"&",round(r22,5),"&",round(r33,5),"&",round(r44,5),"&")




# razao do var
def RRV(w,I,alfa1,alfa2):
   return(VaR(w-I,alfa1)/VaR(-w+I,alfa2))


rrv= RRV(retornos_logaritmicos_indicefuturobin1_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin1_total)],0.95,0.95)
rrv1 = RRV(retornos_logaritmicos_indicefuturobin2_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin2_total)],0.95,0.95)
rrv2 = RRV(retornos_logaritmicos_indicefuturobin3_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin3_total)],0.95,0.95)
rrv3 = RRV(retornos_logaritmicos_indicefuturosembins_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturosembins_total)],0.95,0.95)
print("Razão de Rachev var 0.95,0.95:",round(rrv,5),"&",round(rrv1,5),"&",round(rrv2,5),"&",round(rrv3,5),"&")


rrvv= RRV(retornos_logaritmicos_indicefuturobin1_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin1_total)],0.97,0.97)
rrv11 = RRV(retornos_logaritmicos_indicefuturobin2_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin2_total)],0.97,0.97)
rrv22= RRV(retornos_logaritmicos_indicefuturobin3_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin3_total)],0.97,0.97)
rrv33 = RRV(retornos_logaritmicos_indicefuturosembins_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturosembins_total)],0.97,0.97)
print("Razão de Rachev var 0.97,0.97:", round(rrvv,5),"&",round(rrv11,5),"&",round(rrv22,5),"&",round(rrv33,5),"&")

# rachev do mercado
def RRVm(I,alfa1,alfa2):
   return(VaR(-I,alfa1)/VaR(I,alfa2))

mercado=RRVm(retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturobin1_total)],0.95,0.95)

print("Razão de mercado:", round(mercado,5))



#SRstd
def sharpe_ratio_with_std(returns,std_dev):
    mean_return = np.mean(returns)
  
    rf=0
    sharpe = (mean_return - rf) / std_dev
    
    return sharpe

sharpe1 = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturobin1_total,desvio_padrao1)
sharpe2 = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturobin2_total, desvio_padrao2)
sharpe3 = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturobin3_total, desvio_padrao3)
sharpe4 = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturosembins_total, desvio_padrao_sb)
print("Índice de Sharpe com std:", round(sharpe1,5),"&",round(sharpe2,5),"&",round(sharpe3,5),"&",round(sharpe4,5),"&")


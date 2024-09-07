import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Import TDA utilities
from ripser import Rips
import persim
import statistics

# Import Scikit-Learn tools
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pulp import LpVariable, LpProblem, lpSum, LpMinimize
from datetime import datetime, timedelta

# Parametros

d1=63
d2=42
#tau
tau= 1
#dimensao
dimensao=3
#v
nu = 21
r_star = (1 +0.02)**(1/252)-1 
#datas
#inicio="2005-01-01"
#fim= "2018-11-01"

#indice="^DJI"

nome_arquivo_csv = "C:/Users/acari/Desktop/Tese/sp500.csv"

data = pd.read_csv(nome_arquivo_csv)


#do ^OEX-S&P 100 (wikipedia)
lista_acoes=data.columns[2:]

#data = yf.download(indice, start=inicio, end=fim)
precos_fechamento_indice=np.array(data[data.columns[1]].values.tolist())


def calcular_retornos_logaritmicos(prices):
    prices = np.array(prices)
    # Calcula os retornos logarítmicos
    return np.log10(prices[1:] / prices[:-1])


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
#data= yf.download(indice, start=inicio, end=fim)
#precos_fechamento_indice=np.array(data["Close"].values.tolist())
Nchapeu=serie_normas_TDA(precos_fechamento_indice)

Nacoes=len(lista_acoes)
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
    

# NAIVE PARA TODOS 
def port_naive(inicio,fim):
    indicefuturo_naive= []
    indicereal_naive = []
    n=len(lista_acoes)

    y=np.array([(1/n)/precos_fechamentos[i][inicio+d1] for i in range(n)])
    print(y)


    for dia in range(inicio+d1, min(inicio+d1+d2, fim)):
        precos = np.array([precos_fechamentos[i][dia] for i in range(n)])
        indicefuturo_naive.append(np.matmul(y,precos))
        indicereal_naive.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return (np.array(indicefuturo_naive), np.array(indicereal_naive))
    
def port_naive_para_todas_as_janelas(inicio, fim, d1, d2):
    indicefuturo_naive_total = [1]
    indicereal_total = [1]
    
    while inicio + d1 <= fim:  
        indicefuturo_naive, indicereal = port_naive(inicio,fim)
        indicefuturo_naive_total.extend(indicefuturo_naive*indicefuturo_naive_total[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo_naive_total,indicereal_total)

indicefuturo_naive_total, indicereal_total = port_naive_para_todas_as_janelas(0,len(precos_fechamento_indice), d1, d2)



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
    bin1=df_ordenada[tamanho_total-parte:] #os maiores
    bin2=df_ordenada[:parte] #os mais pequenos
    bin3=df_ordenada[parte:tamanho_total-parte]


    # Exemplo de impressão dos tamanhos e nomes das categorias
    print(f"Tamanho da Categoria {nome_bin1}: {len(bin1)}")
    print(f"Tamanho da Categoria {nome_bin2}: {len(bin2)}")
    print(f"Tamanho da Categoria {nome_bin3}: {len(bin3)}")

    return(bin1,bin2,bin3)

# PARA BIN1
def calcula_portefolio_naive_bin1(inicio,fim):
    bin1, bin2, bin3 = calcula_bins(inicio)
    Bin = bin1
    n = len(Bin)

    componente = [Bin[i][1] for i in range(n)]  # componentes do bin

    indicefuturo_naive_bin1 = []
    indicereal_naive = []


    y=np.array([(1/n)/precos_fechamentos[componente[i]][inicio+d1] for i in range(n)])
    #print(y)


    for dia in range(inicio+d1, min(inicio+d1+d2, fim)):
        precos = np.array([precos_fechamentos[componente[i]][dia] for i in range(n)])
        indicefuturo_naive_bin1.append(np.matmul(y,precos))
        indicereal_naive.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return (np.array(indicefuturo_naive_bin1), np.array(indicereal_naive))




def calcula_portefolio_para_todas_as_janelas_bin1(inicio, fim, d1, d2):
    indicefuturo_naive_total_bin1 = [1]
    indicereal_total = [1]
    
    while inicio + d1  <= fim:  
        indicefuturo_naive, indicereal = calcula_portefolio_naive_bin1(inicio,fim)
        indicefuturo_naive_total_bin1.extend(indicefuturo_naive*indicefuturo_naive_total_bin1[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo_naive_total_bin1,indicereal_total)

indicefuturo_naive_total_bin1, indicereal_total = calcula_portefolio_para_todas_as_janelas_bin1(0,len(precos_fechamento_indice), d1, d2)


# PARA BIN2
def calcula_portefolio_naive_bin2(inicio,fim):
    bin1, bin2, bin3 = calcula_bins(inicio)
    Bin = bin2
    n = len(Bin)

    componente = [Bin[i][1] for i in range(n)]  # componentes do bin

    indicefuturo_naive_bin2 = []
    indicereal_naive = []


    y=np.array([(1/n)/precos_fechamentos[componente[i]][inicio+d1] for i in range(n)])
    #print(y)


    for dia in range(inicio+d1, min(inicio+d1+d2, fim)):
        precos = np.array([precos_fechamentos[componente[i]][dia] for i in range(n)])
        indicefuturo_naive_bin2.append(np.matmul(y,precos))
        indicereal_naive.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return (np.array(indicefuturo_naive_bin2), np.array(indicereal_naive))




def calcula_portefolio_para_todas_as_janelas_bin2(inicio, fim, d1, d2):
    indicefuturo_naive_total_bin2 = [1]
    indicereal_total = [1]
    
    while inicio + d1 <= fim:  
        indicefuturo_naive, indicereal = calcula_portefolio_naive_bin2(inicio,fim)
        indicefuturo_naive_total_bin2.extend(indicefuturo_naive*indicefuturo_naive_total_bin2[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo_naive_total_bin2,indicereal_total)

indicefuturo_naive_total_bin2, indicereal_total = calcula_portefolio_para_todas_as_janelas_bin2(0,len(precos_fechamento_indice), d1, d2)


# PARA BIN3
def calcula_portefolio_naive_bin3(inicio,fim):
    bin1, bin2, bin3 = calcula_bins(inicio)
    Bin = bin3
    n = len(Bin)

    componente = [Bin[i][1] for i in range(n)]  # componentes do bin

    indicefuturo_naive_bin3 = []
    indicereal_naive = []


    y=np.array([(1/n)/precos_fechamentos[componente[i]][inicio+d1] for i in range(n)])
    #print(y)


    for dia in range(inicio+d1,min(inicio+d1+d2, fim)):
        precos = np.array([precos_fechamentos[componente[i]][dia] for i in range(n)])
        indicefuturo_naive_bin3.append(np.matmul(y,precos))
        indicereal_naive.append(precos_fechamento_indice[dia]/precos_fechamento_indice[inicio+d1])
    return (np.array(indicefuturo_naive_bin3), np.array(indicereal_naive))




def calcula_portefolio_para_todas_as_janelas_bin3(inicio, fim, d1, d2):
    indicefuturo_naive_total_bin3 = [1]
    indicereal_total = [1]
    
    while inicio + d1 <= fim:  
        indicefuturo_naive, indicereal = calcula_portefolio_naive_bin3(inicio,fim)
        indicefuturo_naive_total_bin3.extend(indicefuturo_naive*indicefuturo_naive_total_bin3[-1])
        indicereal_total.extend(indicereal*indicereal_total[-1])
        inicio += d2

    return (indicefuturo_naive_total_bin3,indicereal_total)

indicefuturo_naive_total_bin3, indicereal_total = calcula_portefolio_para_todas_as_janelas_bin3(0,len(precos_fechamento_indice), d1, d2)



retornos_logaritmicos_indicefuturo_naive_total_bin1 = calcular_retornos_logaritmicos(indicefuturo_naive_total_bin1[1:])
retornos_logaritmicos_indicefuturo_naive_total_bin2 = calcular_retornos_logaritmicos(indicefuturo_naive_total_bin2[1:])
retornos_logaritmicos_indicefuturo_naive_total_bin3 = calcular_retornos_logaritmicos(indicefuturo_naive_total_bin3[1:])
retornos_logaritmicos_indicefuturo_naive_total = calcular_retornos_logaritmicos(indicefuturo_naive_total[1:])


plt.figure(figsize=(30, 10))
plt.plot(np.log(indicefuturo_naive_total_bin1)/np.log(10), label='Nbin1')
plt.plot(np.log(indicefuturo_naive_total_bin2)/np.log(10), label='Nbin2')
plt.plot(np.log(indicefuturo_naive_total_bin3)/np.log(10), label='Nbin3')
plt.plot(np.log(indicefuturo_naive_total)/np.log(10), label='Ntodos')
plt.plot(np.log(indicereal_total)/np.log(10), label='Índice Sensex 30')
plt.title('Comportamento dos índices')
plt.xlabel('Dias')
plt.ylabel('Retornos Cumulativos')
plt.legend()
plt.show()


def media(w):
    return np.mean(w)
# Analise estatistica
media1 =media(retornos_logaritmicos_indicefuturo_naive_total_bin1)
media2 =media(retornos_logaritmicos_indicefuturo_naive_total_bin2)
media3 = media(retornos_logaritmicos_indicefuturo_naive_total_bin3)
mediat = media(retornos_logaritmicos_indicefuturo_naive_total)
print("Média:", round(media1, 6), "&", round(media2, 6), "&", round(media3, 6),"&",  round(mediat, 6),"&")


def mediana(w):
    return np.median(w)
mediana1 = mediana(retornos_logaritmicos_indicefuturo_naive_total_bin1)
mediana2 = mediana(retornos_logaritmicos_indicefuturo_naive_total_bin2)
mediana3 = mediana (retornos_logaritmicos_indicefuturo_naive_total_bin3)
medianat = mediana(retornos_logaritmicos_indicefuturo_naive_total)
print("Mediana:", round(mediana1, 5),"&",  round(mediana2, 5),"&",  round(mediana3, 5), "&", round(medianat, 5),"&" )


def desvio_padrao(w):
    return np.std(w)
desvio_padrao1 = desvio_padrao(retornos_logaritmicos_indicefuturo_naive_total_bin1)
desvio_padrao2 = desvio_padrao(retornos_logaritmicos_indicefuturo_naive_total_bin2)
desvio_padrao3 = desvio_padrao(retornos_logaritmicos_indicefuturo_naive_total_bin3)
desvio_padraot = desvio_padrao(retornos_logaritmicos_indicefuturo_naive_total)
print("Desvio Padrão:", round(desvio_padrao1, 5),"&",  round(desvio_padrao2, 5),"&",  round(desvio_padrao3, 5),"&",  round(desvio_padraot, 5),"&")


def minimo(w):
    return np.min(w)
minimo1 = minimo(retornos_logaritmicos_indicefuturo_naive_total_bin1)
minimo2 = minimo(retornos_logaritmicos_indicefuturo_naive_total_bin2)
minimo3 = minimo(retornos_logaritmicos_indicefuturo_naive_total_bin3)
minimot = minimo(retornos_logaritmicos_indicefuturo_naive_total)
print("Mínimo:", round(minimo1, 5), "&", round(minimo2, 5), "&", round(minimo3, 5),"&",  round(minimot, 5),"&" )


def maximo(w):
    return np.max(w)
maximo1 = maximo(retornos_logaritmicos_indicefuturo_naive_total_bin1)
maximo2 = maximo(retornos_logaritmicos_indicefuturo_naive_total_bin2)
maximo3 = maximo(retornos_logaritmicos_indicefuturo_naive_total_bin3)
maximot = maximo(retornos_logaritmicos_indicefuturo_naive_total)
print("Máximo:", round(maximo1, 5), "&", round(maximo2, 5),"&",  round(maximo3, 5),"&",  round(maximot, 5),"&" )


def IQR(w):
    return np.percentile(w,75)-np.percentile(w,25)

iqr1 = IQR(retornos_logaritmicos_indicefuturo_naive_total_bin1)
iqr2 = IQR(retornos_logaritmicos_indicefuturo_naive_total_bin2)
iqr3 = IQR(retornos_logaritmicos_indicefuturo_naive_total_bin3)
iqrt = IQR(retornos_logaritmicos_indicefuturo_naive_total)
print("IQR:", round(iqr1, 5), "&", round(iqr2, 5),"&",  round(iqr3, 5), "&", round(iqrt, 5),"&")




def mean_absolute_deviation(data):
    mean = sum(data) / len(data)
    deviations = [abs(x - mean) for x in data]
    return sum(deviations) / len(deviations)

mad1 = mean_absolute_deviation(retornos_logaritmicos_indicefuturo_naive_total_bin1)
mad2 = mean_absolute_deviation(retornos_logaritmicos_indicefuturo_naive_total_bin2)
mad3 = mean_absolute_deviation(retornos_logaritmicos_indicefuturo_naive_total_bin3)
madt = mean_absolute_deviation(retornos_logaritmicos_indicefuturo_naive_total)
print("Desvio Médio Absoluto:", round(mad1, 6), "&", round(mad2, 6),"&",  round(mad3, 6), "&", round(madt, 6),"&")



#SRstd
def sharpe_ratio_with_std(returns,std_dev):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
  
    rf=0
    sharpe = (mean_return - rf) / std_dev
    
    return sharpe

sharpe1s = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturo_naive_total_bin1, desvio_padrao1)
sharpe2s = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturo_naive_total_bin2, desvio_padrao2)
sharpe3s = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturo_naive_total_bin3, desvio_padrao3)
sharpets = sharpe_ratio_with_std(retornos_logaritmicos_indicefuturo_naive_total, desvio_padraot)
print("Índice de Sharpe com std:", round(sharpe1s, 5), "&", round(sharpe2s, 5),"&",  round(sharpe3s, 5),"&", round(sharpets, 5),"&")





def VaR(w,alfa):
    return(np.percentile(w, alfa * 100))
aa1=VaR(-retornos_logaritmicos_indicefuturo_naive_total_bin1,0.95)
aa2=VaR(-retornos_logaritmicos_indicefuturo_naive_total_bin2,0.95)
aa3=VaR(-retornos_logaritmicos_indicefuturo_naive_total_bin3,0.95)
aat=VaR(-retornos_logaritmicos_indicefuturo_naive_total,0.95)
print("var funcao (0.95):", round(aa1, 5), "&", round(aa2, 5), "&", round(aa3, 5),"&",  round(aat, 5),"&")

aaa1=VaR(-retornos_logaritmicos_indicefuturo_naive_total_bin1,0.97)
aaa2=VaR(-retornos_logaritmicos_indicefuturo_naive_total_bin2,0.97)
aaa3=VaR(-retornos_logaritmicos_indicefuturo_naive_total_bin3,0.97)
aaat=VaR(-retornos_logaritmicos_indicefuturo_naive_total,0.97)
print("var funcao (0.97):", round(aaa1, 5), "&",round(aaa2, 5),"&", round(aaa3, 5), "&",round(aaat, 5),"&")



def CVaR(w,alfa):
    return(np.mean(w[w > VaR(w,alfa)]))
bb1=CVaR(-retornos_logaritmicos_indicefuturo_naive_total_bin1,0.95)
bb2=CVaR(-retornos_logaritmicos_indicefuturo_naive_total_bin2,0.95)
bb3=CVaR(-retornos_logaritmicos_indicefuturo_naive_total_bin3,0.95)
bbt=CVaR(-retornos_logaritmicos_indicefuturo_naive_total,0.95)
print("cvar funcao (0.95):", round(bb1, 5),"&", round(bb2, 5), "&",round(bb3, 5), "&",round(bbt, 5),"&")

bbb1=CVaR(-retornos_logaritmicos_indicefuturo_naive_total_bin1,0.97)
bbb2=CVaR(-retornos_logaritmicos_indicefuturo_naive_total_bin2,0.97)
bbb3=CVaR(-retornos_logaritmicos_indicefuturo_naive_total_bin3,0.97)
bbbt=CVaR(-retornos_logaritmicos_indicefuturo_naive_total,0.97)
print("cvar funcao (0.97):", round(bbb1, 6), "&",round(bbb2, 6),"&", round(bbb3, 5),"&", round(bbbt, 5),"&")





nivel_de_confianca=0.95
#SRvar
def sharpe_ratio_with_var(returns, VaR):
    mean_return = np.mean(returns)
    
    rf=0
    # Calcula o Índice de Sharpe usando o VaR como a taxa livre de risco
    sharpe = (mean_return - rf) / VaR
    
    return sharpe

sharpe1v = sharpe_ratio_with_var(retornos_logaritmicos_indicefuturo_naive_total_bin1, aa1)
sharpe2v = sharpe_ratio_with_var(retornos_logaritmicos_indicefuturo_naive_total_bin2,aa2)
sharpe3v = sharpe_ratio_with_var(retornos_logaritmicos_indicefuturo_naive_total_bin3, aa3)
sharpetv = sharpe_ratio_with_var(retornos_logaritmicos_indicefuturo_naive_total, aat)
print("Índice de Sharpe com VaR:", round(sharpe1v, 5),"&", round(sharpe2v, 5), "&",round(sharpe3v, 5),"&", round(sharpetv, 5),"&")



#SRcvar
def sharpe_ratio_with_cvar(returns, cvar):
    mean_return = np.mean(returns)
    
    # Calcula o VaR com o nível de confiança especificado
    var = np.percentile(-returns,  nivel_de_confianca* 100)
    cvar= np.mean(-returns[-returns >= var])
    rf=0
    # Calcula o Índice de Sharpe usando o VaR como a taxa livre de risco
    sharpe = (mean_return - rf) / cvar
    
    return sharpe

sharpe1cv = sharpe_ratio_with_cvar(retornos_logaritmicos_indicefuturo_naive_total_bin1, bb1)
sharpe2cv = sharpe_ratio_with_cvar(retornos_logaritmicos_indicefuturo_naive_total_bin2,bb2)
sharpe3cv = sharpe_ratio_with_cvar(retornos_logaritmicos_indicefuturo_naive_total_bin3, bb3)
sharpetcv = sharpe_ratio_with_cvar(retornos_logaritmicos_indicefuturo_naive_total, bbt)
print("Índice de Sharpe com CVaR:", round(sharpe1cv, 5),"&", round(sharpe2cv, 5),"&", round(sharpe3cv, 5), "&",round(sharpetcv, 5),"&")



##DD-Desvio Negativo (DN)
def downside_deviation(returns, r_star):
    r_star=0
    abaixorreturns = [r for r in returns if r < r_star]
    deviations = [(r - r_star) ** 2 for r in abaixorreturns]
    dd = np.sqrt(sum(deviations) / len(returns))
    return dd


dd1 = downside_deviation(retornos_logaritmicos_indicefuturo_naive_total_bin1, r_star)
dd2 = downside_deviation(retornos_logaritmicos_indicefuturo_naive_total_bin2, r_star)
dd3 = downside_deviation(retornos_logaritmicos_indicefuturo_naive_total_bin3, r_star)
ddt = downside_deviation(retornos_logaritmicos_indicefuturo_naive_total, r_star)
print("Downside Deviation:", round(dd1, 5), "&",round(dd2, 5), "&",round(dd3, 6), "&",round(ddt, 6),"&")




def sortino(returns, downside_deviation):
    mean_return = np.mean(returns)
    rf=0
    if mean_return > rf:
        sortino= (mean_return - rf)/ downside_deviation
        return sortino
    else:
        return 0


sortino1 = sortino(retornos_logaritmicos_indicefuturo_naive_total_bin1, dd1)
sortino2 = sortino(retornos_logaritmicos_indicefuturo_naive_total_bin2, dd2)
sortino3 = sortino(retornos_logaritmicos_indicefuturo_naive_total_bin3, dd3)
sortinot =sortino(retornos_logaritmicos_indicefuturo_naive_total, ddt)
print("Razão de Sortino:", round(sortino1, 5), "&",round(sortino2, 5),"&", round(sortino3, 5),"&",round(sortinot, 5),"&")




def RS(w,I,alfa1,alfa2):
   return(CVaR(w-I,alfa1)/CVaR(-(w-I),alfa2))

# ganhos a dividiir pelas perdas, ganhos primeiro
rs1= RS(retornos_logaritmicos_indicefuturo_naive_total_bin1,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin1)],0.95,0.95)
rs2 = RS(retornos_logaritmicos_indicefuturo_naive_total_bin2,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin2)],0.95,0.95)
rs3 = RS(retornos_logaritmicos_indicefuturo_naive_total_bin3,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin3)],0.95,0.95)
rst = RS(retornos_logaritmicos_indicefuturo_naive_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total)],0.95,0.95)
print("Razão de Rachev 0.95,0.95:", round(rs1, 5), "&",round(rs2, 5),"&", round(rs3, 5), "&",round(rst, 5),"&")



rs11= RS(retornos_logaritmicos_indicefuturo_naive_total_bin1,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin1)],0.97,0.97)
rs22 = RS(retornos_logaritmicos_indicefuturo_naive_total_bin2,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin2)],0.97,0.97)
rs33 = RS(retornos_logaritmicos_indicefuturo_naive_total_bin3,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin3)],0.97,0.97)
rstt = RS(retornos_logaritmicos_indicefuturo_naive_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total)],0.97,0.97)
print("Razão de Rachev 0.97,0.97:", round(rs11, 5),"&", round(rs22, 5), "&",round(rs33, 5),"&",round(rstt, 5),"&")




# razao do var
def RRV(w,I,alfa1,alfa2):
   return(VaR(w-I,alfa1)/VaR(-w+I,alfa2))


rrv1= RRV(retornos_logaritmicos_indicefuturo_naive_total_bin1,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin1)],0.95,0.95)
rrv2 = RRV(retornos_logaritmicos_indicefuturo_naive_total_bin2,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin2)],0.95,0.95)
rrv3 = RRV(retornos_logaritmicos_indicefuturo_naive_total_bin3,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin3)],0.95,0.95)
rrvt = RRV(retornos_logaritmicos_indicefuturo_naive_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total)],0.95,0.95)
print("Razão de Rachev var 0.95,0.95:", round(rrv1, 5),"&", round(rrv2, 5), "&",round(rrv3, 5), "&",round(rrvt,5),"&")



rrv11= RRV(retornos_logaritmicos_indicefuturo_naive_total_bin1,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin1)],0.97,0.97)
rrv22 = RRV(retornos_logaritmicos_indicefuturo_naive_total_bin2,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin2)],0.97,0.97)
rrv33 = RRV(retornos_logaritmicos_indicefuturo_naive_total_bin3,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin3)],0.97,0.97)
rrvtt = RRV(retornos_logaritmicos_indicefuturo_naive_total,retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total)],0.97,0.97)
print("Razão de Rachev var 0.97,0.97:", round(rrv11, 5),"&", round(rrv22, 5), "&",round(rrv33, 5),"&", round(rrvtt, 5)),"&",



# rachev do mercado
def RRVm(I,alfa1,alfa2):
   return(VaR(-I,alfa1)/VaR(I,alfa2))

mercado1=RRVm(retornos_log[d1:d1+len(retornos_logaritmicos_indicefuturo_naive_total_bin1)],0.95,0.95)
print("Razão de mercado:", round(mercado1, 5))




# def normaL1(lista):
#     soma=0
#     for ponto in lista: 
#         soma=soma+((ponto[1]-ponto[0])/2)**2
#     return soma

# normas=[]
# rips = Rips(maxdim=2)
# N = len(serie_temporal) 
# # Loop para criar a nuvem de pontos várias vezes, somando um a todos os índices a cada iteração
# for i in range(N-nu+1):  
#     nuvem_pontos = criar_nuvem_pontos(serie_temporal[i:], dimensao, tau, nu)
#     pers= rips.fit_transform(nuvem_pontos)
#     #print(dim[1])
#     norma = norma_L1(pers[1]) #0 dá nada em ATD e 2 dá nd interessente em atd
#     normas.append(norma)


# norma1=normaL1(retornos_logaritmicos_indicefuturo_naive_total_bin1)
# norma2=normas(retornos_logaritmicos_indicefuturo_naive_total_bin2)
# norma3=normas(retornos_logaritmicos_indicefuturo_naive_total_bin3)
# normat=normas(retornos_logaritmicos_indicefuturo_naive_total)
# print("Norma L1:", round(norma1, 5), round(norma2, 5), round(norma3, 5), round(normat, 5))



# def stdL1(returns, normaL1):
#     mean_return = np.mean(returns)
    
#     rf=0
#     # Calcula o Índice de Sharpe usando o VaR como a taxa livre de risco
#     sharpe = (mean_return - rf) / (np.mean(normaL1) / np.log(10))
    
#     return sharpe

# stdnorma1=stdL1(retornos_logaritmicos_indicefuturo_naive_total_bin1,norma1)
# stdnorma2=stdL1(retornos_logaritmicos_indicefuturo_naive_total_bin2,norma2)
# stdnorma3=stdL1(retornos_logaritmicos_indicefuturo_naive_total_bin3,norma3)
# stdnormat=stdL1(retornos_logaritmicos_indicefuturo_naive_total,normat)

# print("STD norma L1:", round(stdnorma1, 5), round(stdnorma2, 5), round(stdnorma3, 5), round(stdnormat, 5))



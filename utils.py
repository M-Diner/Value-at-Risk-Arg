
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def plot_series(df, title, y_label):   
    plt.figure(figsize=(12, 6))
    plt.margins(x=0)
    plt.plot(df.index, df['BMA'], label='BMA')
    plt.plot(df.index, df['BBAR'], label='BBAR')
    plt.plot(df.index, df['SUPV'], label='SUPV')
    plt.plot(df.index, df['GGAL'], label='GGAL')
    plt.legend()
    plt.xlabel('Fecha')
    plt.ylabel(f'{y_label}')
    plt.title(f'{title}')


def plot_volatility(df, name):
    plt.figure(figsize=(8, 6))
    plt.margins(x=0)
    plt.plot(df.index, df['desv_EWMA'], label='EWMA')
    plt.plot(df.index, df['desv_MM'], label='Box-shapped')
    plt.legend()
    plt.xlabel('Fecha')
    plt.ylabel('Desvio')
    plt.title(f'Desvios para {name}')

def generate_stock_volatility(df_Returns,ticker, Lambda = 0.94, Days=50):

    Returns = df_Returns[ticker]

    # Se genera una list de ponderadores que se iterará sobre los días previos a cada fecha
    n = len(Returns)
    pond=  list()

    for j in list(range(Days)):
        pond.append( (1-Lambda) * (Lambda**(j) )  )


    # Para obtener la varianza se multiplican los ponderadores por el retorno cuadrado y se suman todos los productos.
    i=1
    variance_EWMA = list()

    while(i+Days<n):
    
        ret2_temp= Returns[i:(i+Days)]**2

        variance_EWMA.append( np.dot( pond, ret2_temp).sum() * 252 )
        i=i+1

    # Calculo del desvio EWMA
    desv_EWMA = np.sqrt(variance_EWMA)
    
    # Calculo del desvio por media movil (box-shapped)
    desv_MM = (pd.Series (Returns) 
            .rolling (window = Days) 
            .std (). iloc [Days-1:]
            .values [0:(n-Days-1)] * np.sqrt(252))

    variance_MM = desv_MM**2

    # Dataframe final
    df_volatility = pd.DataFrame({
                    'return': Returns[0:(n-Days-1)],
                    'variance_EWMA': variance_EWMA,
                    'desv_EWMA': desv_EWMA,
                    'variance_MM': variance_MM,
                    'desv_MM': desv_MM})
    return  df_volatility



def generate_multi_volatility_EWMA(Returns, Lambda = 0.94, Days=50):
    
    # Se genera una list de ponderadores que se iterará sobre los días previos a cada fecha
    n = len(Returns)
    pond=  list()
    EWMA_variance = pd.DataFrame()
    EWMA_desv = pd.DataFrame()
   

    # Generacion de ponderadores
    for j in list(range(Days)):
        pond.append( (1-Lambda) * (Lambda**(j) )  )

    # itero sobre cada columna
    for ticker in Returns.columns:
        variance_vec_temp = list()
    # Para obtener la varianza se multiplican los ponderadores por el retorno cuadrado y se suman todos los productos.
        i=1     
        while(i+Days<n):
            
            ret2_temp= Returns[ticker][i:(i+Days) ]**2
            variance_vec_temp.append(np.dot( pond, ret2_temp).sum() )
            
            i=i+1
       
    # Se genera la columna de varianza dentro del df EWMA_variance
        EWMA_variance[ticker] = variance_vec_temp

    # Se genera la columna de desvio dentro del df EWMA_desv
    EWMA_desv = np.sqrt(EWMA_variance)
    EWMA_desv['Date'] = Returns.index[0:(n-Days-1)]
    EWMA_desv = EWMA_desv.set_index('Date')

    return  EWMA_desv

def generate_multi_volatility_MM(Returns , Days=50):
    
    MM_desv = pd.DataFrame()
    
    # itero sobre cada columna
    for ticker in Returns.columns:

        MM_desv[ticker] = (pd.Series (Returns[ticker]) 
                    .sort_index(ascending= True)
                    .rolling (window = Days) 
                    .apply(np.std)
                    .values[Days+1:] 
                    )
        
    MM_desv['Date'] = Returns.sort_index(ascending= True).index[Days+1:]
    MM_desv = MM_desv.set_index('Date').sort_index(ascending= False)
    return  MM_desv


def check_distribution (Returns,ticker):
    mu = np.mean(Returns[ticker])
    sigma =  np.std(Returns[ticker])
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

    Returns[ticker].hist(bins=40,alpha=0.8 , density=True)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    
    plt.title(f'{ticker} returns (binned) vs. normal distribution')
    plt.show()

    kstest_result = stats.kstest(Returns[ticker], 'norm', args=(mu, sigma))
    print(f'{ticker} - Kolmogorov-Smirnov test p-value: {kstest_result.pvalue}')


def generate_multi_COV(Returns, Days):
       
    COV = []
    
    # itero sobre cada columna
    for i in range(len(Returns)-Days):

        COV.append(Returns.iloc[i:i+50, ].cov())
    
    COV= np.array(COV)
    return  COV


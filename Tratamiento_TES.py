
# TES:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

pd.set_option('display.max_columns',None)    


# Para analizar y graficar los datos de tus títulos de deuda, 
# se seguirá una serie de pasos detallados para asegurar que los datos 
# están bien preparados y que las visualizaciones y análisis son significativos. 



    ### Paso 1: Cargar los datos

TES = pd.read_csv("C:/Users/angie.llanos/Documents/Proyecto_SURA/TES_900.csv")
# TES = pd.read_csv("TES_900.csv")

# Mostrar las primeras filas para verificar la carga
print(TES.head())

# Las columnas TITULO y TITULO.1 parecen ser las misma, verifiquemos:
equals = (TES['TITULO'] == TES['TITULO.1']).all()
print(equals)
# Como el resultado es True, se elimina la columna TITULO.1
TES = TES.drop('TITULO.1', axis=1)





    ### Paso 2: Exploración inicial de los datos

# Mostrar información general del DataFrame
print(TES.info())

# Descripción estadística de las columnas numéricas
print(TES.describe())





    ### Paso 3: Limpieza de los datos

# 1. **Manejo de valores nulos**: Identificar y manejar valores nulos.
# 2. **Conversión de tipos de datos**: Asegurarnos de que las fechas y 
    #  otros tipos de datos estén en el formato correcto.

# Como las columnas "TASA", "VR_TASA" y "MARGEN" son todas nulas, se eliminan.
TES = TES.drop(columns = ["TASA", "VR_TASA", "MARGEN"])
print(TES.describe())

# Identificar valores nulos
print(TES.isnull().sum())

# No se evidencian valores nulos en las variables: 'TITULO', 'FECHA_DE_EMISION', 
# 'FECHA_DE_VENCIMIENTO', 'VALOR_NOMINAL'
# Sin embargo sí se evidencian en 'TASA_FACIAL'

print(TES.TASA_FACIAL.unique())

# Mostrar las filas donde TASA_FACIAL es nulo
null_tasa_facial_rows = TES[TES['TASA_FACIAL'].isnull()]
print(null_tasa_facial_rows)


print(null_tasa_facial_rows.TIPO_DE_TASA.unique())
# Como estos que tienen tasa facial nula son al descuento, no hay que eliminar 
# estos títulos de deuda.
del(null_tasa_facial_rows)
# Así el dataframe ya tiene los datos limpios.


# Convertir columnas de fechas a tipo datetime
columnas = [col for col in TES.columns if "FECHA" in col]
TES.loc[:, columnas] = TES.loc[:, columnas].apply(pd.to_datetime, errors = 'coerce')

print(TES.info())





    ### Paso 4: Análisis exploratorio de datos (EDA)

# Se realiza un análisis descriptivo y algunas visualizaciones básicas 
# para entender mejor los datos.

print(f"\n Hay {len(TES.TITULO.unique())} títulos de deuda en la muestra consultada.")

print(f"\n Los tipos de tasa de los títulos son: {TES.TIPO_DE_TASA.unique()}")

#######    Para encontrar los títulos que siguen abiertos a la fecha de valoración
# Fecha de hoy
date_today = pd.to_datetime('today')

# DataFrame con fechas de vencimiento menores o iguales a hoy
Titulos_unique = TES.iloc[:, :32].drop_duplicates()

til_vcto = Titulos_unique.filter(like="VENCIMIEN").columns
TES_not_current = Titulos_unique[Titulos_unique[til_vcto[0]] <= date_today]
print(f"\n La cantidad de títulos vencidos a hoy son: {len(TES_not_current)}")

# DataFrame con fechas mayores a hoy
TES_current = Titulos_unique[Titulos_unique[til_vcto[0]] > date_today]
len(TES_current)

# DataFrame con títulos con fecha de venta (no vigentes a hoy)
TES__not_current = TES_current[~TES_current.FECHA_VENTA.isna()]
len(TES__not_current)

# DataFrame con títulos sin fecha de venta (vigentes a hoy)
TES__current = TES_current[TES_current.FECHA_VENTA.isna()]
print(f"\n La cantidad de títulos vigentes a hoy son: {len(TES__current)}")
TES__current.FECHA_VENTA.unique()


portfolios = TES__current.CODIGO_PORTAFOLIO.unique()
print(f"Hay {len(portfolios)} portafolios de TES")


# Llamaremos al df del portafolio X, como: TES_current_X, p.e. el port. 1V su df se llama: TES_current_1V
mc = 0
for port in portfolios:
    globals()['TES_current_'+str(port)] = TES__current[TES__current.CODIGO_PORTAFOLIO==port]
    
    # descubramos el tipo de títulos en cada portafolio
    if len(globals()['TES_current_'+str(port)]) > 100:
        print("\n\n______________________________________________________________________________")
        print("\nPara el portafolio: ", port, " de longitud: ", len(globals()['TES_current_'+str(port)]))
        print("\nTIPO_DE_TASA: ",list(globals()['TES_current_'+str(port)].TIPO_DE_TASA.unique()))
        print("\nPLAZOS: ",list(np.sort(globals()['TES_current_'+str(port)].PLAZO.unique())))
        print("\nAÑOS DE VENCIMIENTO: ",list(np.sort(globals()['TES_current_'+str(port)][til_vcto[0]].dt.year.unique())))
        print("\nTASA_FACIAL: ",list(globals()['TES_current_'+str(port)].TASA_FACIAL.unique()))
        print("\nSPREAD: ",list(np.sort(globals()['TES_current_'+str(port)].SPREAD.unique())))
        print("\nPERIODICIDAD: ",list(globals()['TES_current_'+str(port)].PERIODICIDAD.unique()))
        print("\nMONEDA: ",list(globals()['TES_current_'+str(port)].MONEDA.unique()))
        print("\nVALOR_NOMINAL: ",list(np.sort(globals()['TES_current_'+str(port)].VALOR_NOMINAL.unique())))
        mc += 1

print("\n\n----------------------------------------")
print("\n Total de portafolios con más de 100 títulos: ", mc)
    
# PORTAFOLIO QUE CONTIENE: TASAS: Fija y Variable (IPC, IBR) y moneda ($, UVR y USD) es 3V    


### Gráficos comunes para visualizar los tipos de títulos en la muestra de títulos vigentes

#### Distribución de los valores nominales
def format_func(value, tick_number):
    return f'{int(value/1e9)}'


plt.figure(figsize=(10, 6))
sns.histplot(TES__current['VALOR_NOMINAL'], bins=30, kde=True)
plt.title('Distribución de los Valores Nominales')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.xlabel('Valor Nominal en MM')
plt.ylabel('Frecuencia')
plt.show()


#### Evolución de los valores nominales a lo largo del tiempo

plt.figure(figsize=(10, 6))
sns.lineplot(data=TES__current, x='FECHA_DE_EMISION', y='VALOR_NOMINAL')
plt.title('Evolución de los Valores Nominales a lo Largo del Tiempo')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.xlabel('Fecha de Emisión')
plt.ylabel('Valor Nominal en MM')
plt.show()


#### Evolución del spred o tasa a lo largo del tiempo

plt.figure(figsize=(10, 6))
sns.lineplot(data=TES__current, x='FECHA_DE_EMISION', y='SPREAD', hue='TIPO_DE_TASA')
plt.title('Evolución del spread a lo Largo del Tiempo')
plt.xlabel('Fecha de Emisión')
plt.ylabel('Spread')
plt.show()

#### Distribución de las tasas de interés por año de emisión

TES__current['AÑO_DE_EMISION'] = TES__current['FECHA_DE_EMISION'].dt.year

plt.figure(figsize=(12, 6))
sns.boxplot(x='AÑO_DE_EMISION', y='TASA_FACIAL', data=TES__current)
plt.title('Distribución de las Tasas de Interés por Año de Emisión')
plt.xlabel('Año de Emisión')
plt.ylabel('Tasa Facial')
plt.xticks(rotation=45)
plt.show()


# with pd.ExcelWriter('TES_total.xlsx') as writer:
#     TES.to_excel(writer, index=True)




    
    ### Paso 5: Análisis avanzado

#### Análisis de series temporales

# Podemos analizar cómo han cambiado las tasas de interés 
# y los valores nominales a lo largo del tiempo.


# Crear una columna 'DURACION' para calcular el tiempo entre la emisión y el vencimiento#
TES__current['DURACION'] = (TES__current[til_vcto[0]] - TES__current['FECHA_DE_EMISION']).dt.days

# Evolución de la tasa facial a lo largo del tiempo
plt.figure(figsize=(10, 6))
sns.lineplot(data=TES__current, x='FECHA_DE_EMISION', y='TASA_FACIAL')
plt.title('Evolución de la Tasa Facial a lo Largo del Tiempo')
plt.xlabel('Fecha de Emisión')
plt.ylabel('Tasa Facial')
plt.show()

# Distribución de la duración de los títulos de deuda
plt.figure(figsize=(10, 6))
sns.histplot(TES__current['DURACION'], bins=30, kde=True)
plt.title('Distribución de la Duración de los Títulos de Deuda')
plt.xlabel('Duración (días)')
plt.ylabel('Frecuencia')
plt.show()

sorted(list(TES__current.DURACION), reverse=True)[:10]

TES__current['DURACION_A'] = TES__current['DURACION']/365
TES__current_sorted = TES__current.sort_values(by='DURACION_A')
TES__current_sorted = TES__current_sorted.reset_index(drop=True)

# Crear la gráfica de puntos
plt.figure(figsize=(10, 6))
plt.scatter(TES__current_sorted.index, TES__current_sorted['DURACION_A'], color='blue', s=50)  # s es el tamaño de los puntos
# Título y etiquetas de los ejes
plt.title('Diagrama de Puntos de PLAZO')
plt.xlabel('Observación')
plt.ylabel('Valor de PLAZO')
# Mostrar la gráfica
plt.show()




     #### Paso 6: Enfoque en el portafolio 3V

# Evolución de la tasa facial a lo largo del tiempo
plt.figure(figsize=(10, 6))
sns.lineplot(data=TES_current_3V, x='FECHA_DE_EMISION', y='TASA_FACIAL')
plt.title('Evolución de la Tasa Facial a lo Largo del Tiempo')
plt.xlabel('Fecha de Emisión')
plt.ylabel('Tasa Facial')
plt.show()

# Distribución de la duración de los títulos de deuda
TES_current_3V['DURACION'] = (TES_current_3V[til_vcto[0]] - TES_current_3V['FECHA_DE_EMISION']).dt.days
TES_current_3V['DURACION_A'] = TES_current_3V['DURACION']/365

TES_current_3V_sorted = TES_current_3V.sort_values(by='DURACION_A')
TES_current_3V_sorted = TES_current_3V_sorted.reset_index(drop=True)
# Crear la gráfica de puntos
plt.figure(figsize=(10, 6))
plt.scatter(TES_current_3V_sorted.index, TES_current_3V_sorted['DURACION_A'], color='blue', s=50)  # s es el tamaño de los puntos
# Título y etiquetas de los ejes
plt.title('Diagrama de PLAZO portafolio 3V')
plt.xlabel('Observación')
plt.ylabel('Valor de PLAZO')
# Mostrar la gráfica
plt.show()



plt.figure(figsize=(10, 6))
sns.histplot(TES_current_3V['DURACION'], bins=30, kde=True)
plt.title('Distribución de la Duración de los Títulos de Deuda')
plt.xlabel('Duración (días)')
plt.ylabel('Frecuencia')
plt.show()



TES_current_3V = TES_current_3V.sort_values(by='VALOR_NOMINAL')
TES_current_3V = TES_current_3V.reset_index(drop=True)
plt.figure(figsize=(10, 6))
sns.histplot(TES_current_3V['VALOR_NOMINAL'], bins=30, kde=True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
plt.title('Distribución de nominales del portafolio 3v')
plt.xlabel('Nominal en MM')
plt.ylabel('Frecuencia')
plt.show()


#### Evolución del spred o tasa a lo largo del tiempo

plt.figure(figsize=(10, 6))
sns.lineplot(data=TES_current_3V, x='FECHA_DE_EMISION', y='SPREAD', hue='TIPO_DE_TASA')
plt.title('Evolución del spread a lo Largo del Tiempo')
plt.xlabel('Fecha de Emisión')
plt.ylabel('Spread')
plt.show()


# Intervalo de tiempo en los que están comprendidos los títulos
# Lim inf, menor fecha de inicio
lim_inf_t = min(TES_current_3V.FECHA_DE_EMISION)
print(f"Menor fecha de inicio: {lim_inf_t}")
# Lim sup, mayor fecha de fin
lim_sup_t = max(TES_current_3V[til_vcto[0]])
print(f"Mayor fecha de vencimiento: {lim_sup_t}")

        
        ### Paso 7: Valoración por un horizonte de inversión de 6 meses
        
    # Paso 1: Valorar los títulos a partir de una fecha donde todos estén  
    # vigentes hasta los 6 meses con periodicidad diaria
ini_vigen = max(TES_current_3V.FECHA_DE_EMISION)
fin_vigen = max(TES_current_3V[til_vcto[0]])
print(f"El mayor inicio de vigencia es: {ini_vigen} y menor vencimiento es: {fin_vigen}")
        
        
        # Para usar valores reales de los índices se va a valorar desde el mayor inicio de vigencia con 
        # horizonte de inversión 6 meses, pero cuando se quiera correr a la fecha actual, 
        # ya no se usan las inputs conocidas sino las proyecciones, las cuales se deben diseñar 
        # En el mismo formato que las reales.
    
    # Cargar los valores reales del IBR e IPC
IPC = pd.read_excel("C:/Users/angie.llanos/Documents/Proyecto_SURA/IPC_Serie.xlsx")
IBR = pd.read_excel("C:/Users/angie.llanos/Documents/Proyecto_SURA/IBR_overnight.xlsx")

    # Cargar los valores reales de UVR y USD
UVR = pd.read_excel("C:/Users/angie.llanos/Documents/Proyecto_SURA/UVR_Serie.xlsx")
USD = pd.read_excel("C:/Users/angie.llanos/Documents/Proyecto_SURA/TRM.xlsx")


    #### Como el ini_vigen = 2022-05-16 se realiza la Valoración a partir del 01/07/2022 al 31/12/2022

## Valoración de todos estos títulos:

# class Valoration:
    
#     def __init__(self, portfolio):
#         self.portfolio = portfolio


# Cargar betas del BanRep
BETAS_COP = pd.read_excel("C:/Users/angie.llanos/Documents/Proyecto_SURA/Betas_TES_COP.xlsx")
BETAS_UVR = pd.read_excel("C:/Users/angie.llanos/Documents/Proyecto_SURA/Betas_TES_UVR.xlsx")

def curva_nelson_siegel(t, beta0, beta1, beta2, tau):
    """Calcula la tasa usando la curva de Nelson-Siegel."""
    term1 = beta0
    term2 = beta1 * ((1 - np.exp(-t / tau)) / (t / tau))
    term3 = beta2 * ((1 - np.exp(-t / tau)) / (t / tau) - np.exp(-t / tau))
    return term1 + term2 + term3


def calcular_tasa_descuento(fecha_actual, vencimiento, moneda):
    """Calcula la tasa de descuento utilizando la curva de Nelson-Siegel."""
    plazo = (vencimiento - fecha_actual).days / 365
    betas = BETAS_UVR if moneda == 'UVR' else BETAS_COP
    beta0 = betas.loc[betas.Fecha == fecha_actual, 'B0'].item()
    beta1 = betas.loc[betas.Fecha == fecha_actual, 'B1'].item()
    beta2 = betas.loc[betas.Fecha == fecha_actual, 'B2'].item()
    tau = betas.loc[betas.Fecha == fecha_actual, 'Tau'].item()
    return curva_nelson_siegel(plazo, beta0, beta1, beta2, tau)


def calcular_valor_presente(tasa_descuento, valor_futuro, plazo, periodos_anuales):
    """Calcula el valor presente de un flujo de caja futuro."""
    factor_descuento = (1 + tasa_descuento / periodos_anuales) ** (plazo * periodos_anuales)
    return valor_futuro / factor_descuento


def calcular_valor_presente(tasa_dsct, nominal, plazo, f_v, periodos_anuales, spread):
    if periodos_anuales == 0 and f_v == 0:   # No periódico y tasa fija
        tasa_dsct = tasa_dsct / (1 if tasa_dsct < 1 else 100)
        PV = nominal * (1 + spread) / (1 + tasa_dsct)**plazo
        return PV


def obtener_periodos_anuales(periodo):
    """Determina el número de periodos en un año según la periodicidad."""
    periodos_dict = {
        'Mensual': 12,
        'Trimestral': 4,
        'Semestral': 2,
        'Anual': 1,
        'No Periodico': 0
    }
    if periodo not in periodos_dict:
        raise ValueError("Periodicidad no reconocida.")
    return periodos_dict[periodo]


# def calcular_valor_bono_variable_ns(bono, fecha_actual):
#     # Calcular la tasa de descuento utilizando la función que ya tienes
#     tasa_descuento_nominal = calcular_tasa_descuento_nelson_siegel(fecha_actual, bono['FECHA_DE_VENCIMIENTO'] , bono['MONEDA'] )
#     # Determinar el número de periodos en un año
#     periodo = bono['PERIODICIDAD']
#     if periodo == 'Mensual':
#         periodos_anuales = 12
#     elif periodo == 'Trimestral':
#         periodos_anuales = 4
#     elif periodo == 'Semestral':
#         periodos_anuales = 2
#     elif periodo == 'Anual':
#         periodos_anuales = 1
#     elif periodo == 'No Periodico':
#         periodos_anuales = 0
#     else:
#         raise ValueError("Periodicidad no reconocida.")
    
#     fija_o_var = 1 if bono.TIPO_DE_TASA == 'Tasa Variable' else 0       # 0: Fija 1: Variable 
    
#     if fija_o_var == 1:
#         if bono.TASA_FACIAL == 'IPC5' or bono.TASA_FACIAL == 'IPC':
#             fija_o_var = 11         # 11: IPC
#         if bono.TASA_FACIAL == 'IB1':
#             fija_o_var = 22         # 22: IBR
    
#     # Si no es periódico, calculamos un solo flujo al vencimiento
#     if periodos_anuales == 0:
#         plazo = (bono['FECHA_DE_VENCIMIENTO'] - fecha_actual).days / 365
#         valor_presente = calcular_valor_presente(tasa_descuento_nominal, bono['VALOR_NOMINAL'] , plazo, fija_o_var, periodos_anuales, bono['SPREAD'] )
#         if bono['MONEDA'] != '$':
#             mon = UVR if bono['MONEDA'] == 'UVR' else USD
#             col = 'COP_x_UVR' if bono['MONEDA'] == 'UVR' else 'TRM'
#             valor_presente = valor_presente * mon.loc[mon.Fecha==fecha_actual, col].item()
            
#         return valor_presente
    
    
    
#     ############################################################################
    
#     # Calcular el número total de cupones restantes
#     plazo_total = (bono['FECHA_DE_VENCIMIENTO'] - fecha_actual).days / 365
#     cantidad_cupones = int(np.ceil(plazo_total * periodos_anuales))
#     # Calcular el valor del cupón y el spread
#     tasa_variabl○e = tasa_descuento_nominal + bono['SPREAD']   # Tasa 
#     valor_cupon = (bono['TASA_FACIAL'] / periodos_anuales) * bono['VALOR_NOMINAL']
#     valor_presente_total = 0.0
#     for n in range(1, cantidad_cupones + 1):
#         plazo_cupon = n / periodos_anuales
#         valor_presente_cupon = valor_cupon / (1 + tasa_variable / periodos_anuales) ** (plazo_cupon * periodos_anuales)
#         valor_presente_total += valor_presente_cupon
#     # Agregar el valor presente del valor nominal al vencimiento
#     valor_presente_nominal = bono['VALOR_NOMINAL'] / (1 + tasa_variable / periodos_anuales) ** (plazo_total * periodos_anuales)
#     valor_presente_total += valor_presente_nominal
#     return valor_presente_total

def obtener_tasa_cambio(fecha_actual, moneda):
    """Obtiene la tasa de cambio de UVR o USD a COP."""
    mon = UVR if moneda == 'UVR' else USD
    col = 'COP_x_UVR' if moneda == 'UVR' else 'TRM'
    return mon.loc[mon.Fecha == fecha_actual, col].item()

 
def calcular_valor_bono_variable_ns(bono, fecha_actual):
    """Calcula el valor presente de un bono con cupones variables."""
    T = bono['PERIODICIDAD']
    periodos_anuales = obtener_periodos_anuales(T)
    plazo_total = (bono['FECHA_DE_VENCIMIENTO'] - fecha_actual).days / 365
 
    # Si no es periódico, calculamos un solo flujo al vencimiento
    if periodos_anuales == 0:
        valor_presente = calcular_valor_presente(
            calcular_tasa_descuento(fecha_actual, bono['FECHA_DE_VENCIMIENTO'], bono['MONEDA']),
            bono['VALOR_NOMINAL'], plazo_total, 1
        )
        if bono['MONEDA'] != 'COP':
            tasa_cambio = obtener_tasa_cambio(fecha_actual, bono['MONEDA'])
            valor_presente *= tasa_cambio
        return valor_presente
 
 
    # Inicializar el valor presente total
    valor_presente_total = 0.0
    
    valoration_date = fecha_actual.month
    
    list_periodo = [ 1 + num for num in list(range(periodos_anuales))]
    list_periodo = [ 12/periodos_anuales * num for num in list_periodo]
    valor_cupon = np.ndarray(10)
    
    valor_presente_total = 0
    for n in range(1, int(np.ceil(plazo_total * periodos_anuales)) + 1):
        
        for i in range(len(list_periodo)):
            if valoration_date > list_periodo[i-1] and valoration_date <= list_periodo[i]:
                next_coupon = list_periodo[i]
                
        fecha_cupon = fecha_actual.replace(month=((int(next_coupon) + 1) if int(next_coupon)==13 else 1), day=1) - timedelta(1)
        
        
        
        # Calcular el valor de cada cupón
        if bono['TASA_FACIAL']== "IBR":
            tasa_facial = IBR.IBR_efectiva_porc[IBR.Fecha==str(fecha_cupon)[:10]].item()
        if bono['TASA_FACIAL']== "IPC" or bono['TASA_FACIAL']== "IPC5":
            date = int(str(fecha_cupon)[:4]+str(fecha_cupon)[5:7])
            tasa_facial = IPC.Infl_anual_porc[IPC.A_M==date].item()
        if bono['TASA_FACIAL']== 'Fija':
            tasa_facial = 0
        
        tasa_facial = tasa_facial/(1 if tasa_facial<1 else 100)
        
        tasa_facial_ajustada = tasa_facial + bono['SPREAD']
        valor_cupon = (tasa_facial_ajustada / periodos_anuales) * bono['VALOR_NOMINAL']
        
        
        tasa_descuento = calcular_tasa_descuento(fecha_actual, fecha_cupon, bono['MONEDA'])
        valor_presente_total += calcular_valor_presente(tasa_descuento, valor_cupon, n / periodos_anuales, periodos_anuales)
 
    # Añadir el valor presente del valor nominal al vencimiento
    tasa_descuento_final = calcular_tasa_descuento(fecha_actual, bono['FECHA_DE_VENCIMIENTO'], bono['MONEDA'])
    valor_presente_total += calcular_valor_presente(tasa_descuento_final, bono['VALOR_NOMINAL'], plazo_total, periodos_anuales)
    return valor_presente_tota
 
df_portfolio_3v = TES[TES.CODIGO_PORTAFOLIO == '3V']
bono = df_portfolio_3v[df_portfolio_3v.TITULO == 109396].reset_index(drop=True)
bono = bono.iloc[0]
fecha_actual = datetime(2021,12,26)

# Ejemplo de uso:
# bono = {
#     'FECHA_DE_VENCIMIENTO': datetime(2030, 5, 16),
#     'MONEDA': 'COP',
#     'PERIODICIDAD': 'Semestral',
#     'SPREAD': 0.02,
#     'VALOR_NOMINAL': 1000000,
#     'TASA_FACIAL': 0.05,
#     'TIPO_DE_TASA': 'Tasa Variable'
# }
# fecha_actual = datetime(2024, 5, 16)
# valor_bono = calcular_valor_bono_variable_ns(bono, fecha_actual)
# print(f"Valor presente del bono: {valor_bono}")


##########################################################################################


def valorar_portafolio_ns(df_bonos, fecha_inicio, horizonte_meses):
    fecha_actual = fecha_inicio
    fecha_fin = fecha_inicio + timedelta(days=horizonte_meses * 30)
    resultados = []

    while fecha_actual <= fecha_fin:
        valor_portafolio_dia = 0
        for index, bono in df_bonos.iterrows():
            if bono['TIPO_DE_TASA'] == 'Tasa Variable':
                valor_bono = calcular_valor_bono_variable_ns(bono, fecha_actual)
            elif bono['TIPO_DE_TASA'] == 'Tasa Fija':
                tasa_descuento = calcular_tasa_descuento_nelson_siegel(fecha_actual, bono['FECHA_DE_VENCIMIENTO'], curvas_tasas_nelson_siegel[bono['Moneda']])
                valor_bono = calcular_valor_presente(
                    tasa_descuento=tasa_descuento,
                    principal=bono['VALOR_NOMINAL'],
                    plazo=(bono['FECHA_DE_VENCIMIENTO'] - fecha_actual).days / 365,
                    tasa_cupon=bono['TASA_FACIAL']
                )
            valor_portafolio_dia += valor_bono

        resultados.append({'Fecha': fecha_actual, 'Valor Portafolio': valor_portafolio_dia})
        fecha_actual += timedelta(days=1)

    return pd.DataFrame(resultados)





# Gráfica de evolución en la valoración de algunos de los títulos
for titulo in TES.TITULO.unique():
    
    print()

#### Clustering (agrupamiento)

# Seleccionar características para el clustering
features = ['VALOR_NOMINAL', 'SPREAD', 'TASA_COMPRA', 'DURACION']
X = TES__current[features].dropna()

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
TES__current['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=TES__current, x='VALOR_NOMINAL', y='TASA_FACIAL', hue='Cluster', palette='viridis')
plt.title('Clustering de los Títulos de Deuda')
plt.xlabel('Valor Nominal')
plt.ylabel('Tasa Facial')
plt.show()



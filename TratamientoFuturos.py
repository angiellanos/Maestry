
# FUTUROS:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',None)    



    ### Paso 1: Cargar los datos

FUTUROS = pd.read_csv("futuros.csv")

# Mostrar las primeras filas para verificar la carga
print(FUTUROS.head())


portfolios_futuros = FUTUROS.CODIGO_PORTAFOLIO
print(portfolios_futuros)



#######    Para encontrar los títulos que siguen abiertos a la fecha de valoración
# Fecha de hoy
date_today = pd.to_datetime('today')

FUTUROS['FECHA_CONSTITUCION'] = pd.to_datetime(FUTUROS['FECHA_CONSTITUCION'])
FUTUROS['FECHA_VENCIMIENTO'] = pd.to_datetime(FUTUROS['FECHA_VENCIMIENTO'])
FUTUROS['FECHA_CIERRE']      = pd.to_datetime(FUTUROS['FECHA_CIERRE'])

# DataFrame con fechas de vencimiento menores o iguales a hoy
FUTUROS_unique = FUTUROS.iloc[:, :28].drop_duplicates()

FTR_not_current = FUTUROS_unique[FUTUROS_unique['FECHA_VENCIMIENTO'] <= date_today]
print(f"\n La cantidad de títulos vencidos a hoy son: {len(FTR_not_current.TITULO.unique())}")











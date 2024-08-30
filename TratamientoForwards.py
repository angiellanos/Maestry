
# FORWARDS SOBRE DIVISAS:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',None)    


# Para analizar y graficar los datos de tus títulos de deuda, 
# se seguirá una serie de pasos detallados para asegurar que los datos 
# están bien preparados y que las visualizaciones y análisis son significativos. 



    ### Paso 1: Cargar los datos

FORWARDS = pd.read_csv("Forward_divisas_900.csv")

if ( (FORWARDS.iloc[:, 0] == FORWARDS.iloc[:, 2] ).all() 
    and (FORWARDS.iloc[:, 0] == FORWARDS.iloc[:, 19] ).all()
    and (FORWARDS.iloc[:, 0] == FORWARDS.iloc[:, 21]).all() 
    and ( FORWARDS.iloc[:, 1]  == FORWARDS.iloc[:, 4]).all() 
    # and (FORWARDS.iloc[:, 4]  == FORWARDS.iloc[:, 22]).all() 
    and (FORWARDS.iloc[:, 20]  == FORWARDS.iloc[:, 23]).all()):
        FORWARDS = FORWARDS.loc[:, ~FORWARDS.columns.duplicated()]
        FORWARDS = FORWARDS.drop(FORWARDS.columns[[2,19,21,4,23]], axis=1)

# Mostrar las primeras filas para verificar la carga
print(FORWARDS.head())

portfolios_forward = FORWARDS.CODIGO_PORTAFOLIO
print(portfolios_forward)


#######    Para encontrar los títulos que siguen abiertos a la fecha de valoración
# Fecha de hoy
date_today = pd.to_datetime('today')

FORWARDS['FECHA_CONSTITUCION'] = pd.to_datetime(FORWARDS['FECHA_CONSTITUCION'])
FORWARDS['FECHA_VENCIMIENTO'] = pd.to_datetime(FORWARDS['FECHA_VENCIMIENTO'])
FORWARDS['FECHA_CIERRE']      = pd.to_datetime(FORWARDS['FECHA_CIERRE'])

# DataFrame con fechas de vencimiento menores o iguales a hoy
Forwards_unique = FORWARDS.iloc[:, :28].drop_duplicates()

FWRD_not_current = Forwards_unique[Forwards_unique['FECHA_VENCIMIENTO'] <= date_today]
print(f"\n La cantidad de títulos vencidos a hoy son: {len(FWRD_not_current.TITULO.unique())}")

# DataFrame con fechas mayores a hoy
FWRD_current = Forwards_unique[Forwards_unique['FECHA_DE_VENCIMIENTO'] > date_today]
len(FWRD_current)


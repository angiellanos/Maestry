
# SWAPS:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns',None)    



    ### Paso 1: Cargar los datos

SWAPS = pd.read_csv("Swaps.csv")

# Mostrar las primeras filas para verificar la carga
print(SWAPS.head())


portfolios_SWAPS = SWAPS.CODIGO_PORTAFOLIO
print(portfolios_SWAPS)



#######    Para encontrar los títulos que siguen abiertos a la fecha de valoración
# Fecha de hoy
date_today = pd.to_datetime('today')

SWAPS['FECHA_CONSTITUCION'] = pd.to_datetime(SWAPS['FECHA_CONSTITUCION'])
SWAPS['FECHA_VENCIMIENTO'] = pd.to_datetime(SWAPS['FECHA_VENCIMIENTO'])
SWAPS['FECHA_CIERRE']      = pd.to_datetime(SWAPS['FECHA_CIERRE'])

# DataFrame con fechas de vencimiento menores o iguales a hoy
SWAPS_unique = SWAPS.iloc[:, :28].drop_duplicates()

SWAP_not_current = SWAPS_unique[SWAPS_unique['FECHA_VENCIMIENTO'] <= date_today]
print(f"\n La cantidad de títulos vencidos a hoy son: {len(SWAP_not_current.TITULO.unique())}")


# DataFrame con fechas mayores a hoy
FWRD_current = Forwards_unique[Forwards_unique['FECHA_DE_VENCIMIENTO'] > date_today]
len(FWRD_current)



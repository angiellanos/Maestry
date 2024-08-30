import pandas as pd
import numpy as np
import cx_Oracle , os
import warnings
from datetime import datetime
from pydantic import BaseModel

warnings.filterwarnings('ignore')

nvl = lambda x, y=0: x if ((x is not None) & (str(x)!='nan')) else y

already = False
Route = "C:/Oracle/instantclient_21_3/"

def init_oracle():
    global already
    if (already == False):
        cx_Oracle.init_oracle_client(lib_dir=fr'{Route}')
        already = True


def connect(hostname,port,service_name,user_name,password):
    connect_string=cx_Oracle.makedsn(hostname, port, service_name=service_name)
    conn= cx_Oracle.connect(user=user_name, password=password, dsn=connect_string)
    return conn

def conection():
    init_oracle()     
    user = "ops$porf"
    password = 'YRfRds$2023' 
    service_name = 'DLLFINAN' 
    hostname = '10.203.10.190' 
    port = '1537'
    conn = connect(hostname, port, service_name, user, password)
    return conn

conn = conection()


query1 = f""" select * from PFVIEW_SP_TDE_NEG_FIL """
query3 = f""" select * from PFVIEW_SP_TDE_VAL_FIL """
query4 = f""" select * from PFVIEW_SP_TDE_VAL_ATR """
                             

cursor = conn.cursor()
cursor.execute(query1)

consult1 = cursor.fetchall()

columns = [desc[0] for desc in cursor.description]
consulta1 = pd.DataFrame(consult1, columns = columns) 
consulta1 = consulta1.loc[:, ~consulta1.columns.duplicated()]

consulta1.to_csv("consulta_1.csv", index=False)




# cursor.execute(query2)
# consult2 = cursor.fetchall()

# columns = [desc[0] for desc in cursor.description]
# consulta2 = pd.DataFrame(consult2, columns = columns) 
# consulta2 = consulta2.loc[:, ~consulta2.columns.duplicated()]

# consulta2.to_csv("consulta_2.csv", index=False)




cursor.execute(query3)
consult3 = cursor.fetchall()

columns = [desc[0] for desc in cursor.description]
consulta3 = pd.DataFrame(consult3, columns = columns) 
consulta3 = consulta3.loc[:, ~consulta3.columns.duplicated()]

consulta3.to_csv("consulta_3.csv", index=False)






cursor.execute(query4)
consult4 = cursor.fetchall()


columns = [desc[0] for desc in cursor.description]
consulta4 = pd.DataFrame(consult4, columns = columns) 
consulta4 = consulta4.loc[:, ~consulta4.columns.duplicated()]

consulta4.to_csv("consulta_4.csv", index=False)






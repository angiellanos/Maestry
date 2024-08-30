import cx_Oracle
import warnings
import os
from dotenv import load_dotenv

# Ignorar advertencias si es necesario
warnings.filterwarnings('ignore')

# Cargar variables de entorno
load_dotenv()


# Función para inicializar el pool de conexiones
def init_oracle_pool():
    # Ruta de la biblioteca cliente de Oracle, 
    lib_dir = os.getenv("ORACLE_PATH","")
    cx_Oracle.init_oracle_client(lib_dir=lib_dir)

    # Credenciales y datos para la conexión
    user = os.getenv("DB_USER","ops$porf")
    password = os.getenv("DB_PASSWORD",'pruebas_2023')
    service_name = os.getenv("DB_SERVICE_NAME",'prsuprgq')
    hostname = os.getenv("DB_HOSTNAME",'berlin.alfagl.com')
    port =  os.getenv("DB_PORT",'1621')
    
    # Crear DSN
    dsn = cx_Oracle.makedsn(hostname, port, service_name=service_name)
    
    #Crear y retornar el pool
    pool = cx_Oracle.SessionPool(user=user,
                                 password=password,
                                 dsn=dsn,
                                 min=1,
                                 max=20,
                                 increment=1,
                                 encoding="UTF-8",
                                 wait_timeout=300)
    return pool

# Crear el pool al cargar el módulo
pool = init_oracle_pool()

# Función para obtener una conexión desde el pool
def connection_d():
    return pool.acquire()


# Función para liberar la conexión al pool
def release_conn(conn):
    conn.close()
###

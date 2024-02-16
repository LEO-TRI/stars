import pyodbc
import pandas as pd

# For local execution

SERVER = "APEDC1BO1BDU18P.ape.intranet"
DATABASE = "DWH"
USERNAME = "svc-p-infradata"
PASSWORD = ""

def get_cursor(server=SERVER, database=DATABASE, username=USERNAME, password=PASSWORD):
    connection_string = f"""DRIVER={{SQL Server}};SERVER={server};DATABASE={database};uid={username};pwd={password}"""
    connector = pyodbc.connect(connection_string)
    cursor = connector.cursor()
    return cursor


def extract_sql_to_dataframe(
    cursor, query: str = "SELECT DISTINCT * FROM DWH.dbo.InfrastructureFunds"
):
    data = cursor.execute(query).fetchall()
    columns = [column[0] for column in cursor.description]
    dataframe = pd.DataFrame.from_records(data)
    dataframe.columns = columns
    return dataframe


# For lake execution

DRIVER = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
JDBCHOSTNAME = "apeazsqldpt01p.public.59a28a428f4b.database.windows.net"
JDBCPORT = '3342'
JDBCDATABASE = 'ADS' 
JDBCUSERNAME = 'svc-p-infradata'

def get_jdbc_url(password, jdbcHostname: str=JDBCHOSTNAME, jdbcPort: str=JDBCPORT, jdbcDatabase: str=JDBCDATABASE, username: str=JDBCUSERNAME, driver: str=DRIVER):
    jdbcUrl = f"jdbc:sqlserver://{jdbcHostname}:{jdbcPort};database={jdbcDatabase};user={username};password={password};driver={driver}"
    return jdbcUrl

def extract_sql_to_pandas_from_lake(
  spark,
  jdbc_url: str,
  password: str,
  query: str="""SELECT DISTINCT * FROM ADS.outputinfra.PFCashflows"""
):
  dataframe = (
    spark.read.format("jdbc")
    .option("driver", DRIVER)
    .option("url", jdbc_url)
    .option("query", query)
    .option("user", JDBCUSERNAME)
    .option("password", password)
    .load()).toPandas()
  return dataframe



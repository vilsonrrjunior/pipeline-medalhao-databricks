# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# pyspark.sql.functions — modulo principal de funcoes do PySpark.
from pyspark.sql import functions as F

# pyspark.sql.types — modulo de tipos de dados do PySpark.
from pyspark.sql.types import (
	StructType,
	StructField,
	StringType,
	IntegerType,
	FloatType,
	DateType,
	ShortType,
)
CATALOG = "northwind"
SCHEMA = "bronze"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
# Criar Volume para armazenar arquivos
spark.sql("CREATE VOLUME IF NOT EXISTS demo_files")
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/demo_files"
print(f"Volume disponível em: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregando tabelas com `spark.table()`
# MAGIC
# MAGIC `spark.table()` retorna um **DataFrame** -- a estrutura central do PySpark para manipulacao de dados.

# COMMAND ----------

orders = spark.table("orders")
customers = spark.table("customers")

# Verificando o tipo
print(type(orders))

# COMMAND ----------

# Visualizacao rapida com display()
display(orders)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Lendo arquivos CSV com `spark.read.csv()`

# COMMAND ----------

orders.write.csv(
	f"{VOLUME_PATH}/orders_csv",
	header=True,
	mode="overwrite",
)

# COMMAND ----------

# Lendo o CSV com inferSchema=True
# O Spark tenta "adivinhar" os tipos das colunas
orders_csv = spark.read.csv(
	f"{VOLUME_PATH}/orders_csv",
	header=True,
	inferSchema=True,
)

orders_csv.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Schema explicito com `StructType`
# MAGIC
# MAGIC Cada `StructField` recebe:
# MAGIC - Nome da coluna (`string`)
# MAGIC - Tipo de dado (`StringType()`, `IntegerType()`, `FloatType()`, `DateType()`, etc.)
# MAGIC - Se aceita nulos (`True` / `False`)

# COMMAND ----------

order_details = spark.table("order_details")

order_details.write.csv(
	f"{VOLUME_PATH}/order_details_csv",
	header=True,
	mode="overwrite",
)

# COMMAND ----------

order_details_schema = StructType([
	StructField("order_id", ShortType(), nullable=False),
	StructField("product_id", ShortType(), nullable=False),
	StructField("unit_price", FloatType(), nullable=False),
	StructField("quantity", ShortType(), nullable=False),
	StructField("discount", FloatType(), nullable=False),
])

# COMMAND ----------

order_details_csv = spark.read.csv(
	f"{VOLUME_PATH}/order_details_csv",
	header=True,
	schema=order_details_schema,
)

# COMMAND ----------

# Diferença schema inferido vs schema explicito
order_details_inferido = spark.read.csv(
    f"{VOLUME_PATH}/order_details_csv",
    header=True,
    inferSchema=True,
)
print("INFERIDO")
order_details_inferido.printSchema()
print("EXPLICITO")
order_details_csv.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `spark.table()` vs `spark.read` -- quando usar cada um
# MAGIC
# MAGIC **Regra:** Se a tabela esta no catalogo, usar `spark.table()`. Se esta lendo arquivos brutos (CSV, JSON, Parquet), usar `spark.read`.

# COMMAND ----------

# Abordagem 1: spark.table() -- tabela do catalogo
products = spark.table("products")
display(products.limit(5))

# COMMAND ----------

# Abordagem 2: spark.read -- arquivo CSV
orders_from_file = spark.read.csv(
    f"{VOLUME_PATH}/orders_csv",
    header=True,
    inferSchema=True,
)
display(orders_from_file.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### CSV
# MAGIC
# MAGIC Precisa de `inferSchema` ou schema explicito.

# COMMAND ----------

# Ja demonstrado acima. Opcoes comuns:
df_csv = spark.read.csv(
	f"{VOLUME_PATH}/orders_csv",
	header=True,
	inferSchema=True,
	sep=",",
	nullValue="NA",
)
print(f"CSV: {df_csv.count()} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### JSON

# COMMAND ----------

# Escrevendo orders como JSON
orders.write.json(
	f"{VOLUME_PATH}/orders_json",
	mode="overwrite",
)
# Lendo de volta
df_json = spark.read.json(f"{VOLUME_PATH}/orders_json")
print(f"JSON: {df_json.count()} linhas")
df_json.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parquet
# MAGIC
# MAGIC Muito mais eficiente que CSV/JSON porque:
# MAGIC - Armazena tipos nativamente (sem necessidade de inferSchema)
# MAGIC - Compressao por coluna

# COMMAND ----------

# Escrevendo orders como Parquet
orders.write.parquet(
	f"{VOLUME_PATH}/orders_parquet",
	mode="overwrite",
)
# Lendo de volta -- note que nao precisa de inferSchema
df_parquet = spark.read.parquet(f"{VOLUME_PATH}/orders_parquet")
print(f"Parquet: {df_parquet.count()} linhas")
df_parquet.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Delta
# MAGIC
# MAGIC Formato nativo do Databricks. Parquet + log de transacoes. Suporte a ACID, time travel.

# COMMAND ----------

# Escrevendo orders como Delta
orders.write.format("delta").mode("overwrite").save(
	f"{VOLUME_PATH}/orders_delta"
)
# Lendo de volta
df_delta = spark.read.format("delta").load(f"{VOLUME_PATH}/orders_delta")
print(f"Delta: {df_delta.count()} linhas")
df_delta.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. DataFrame API vs `spark.sql()` -- duas formas de fazer a mesma coisa
# MAGIC
# MAGIC O PySpark oferece duas formas equivalentes de consultar dados:

# COMMAND ----------

# Para usar spark.sql(), primeiro criamos uma view
products = spark.table("northwind.bronze.products")
products.createOrReplaceTempView("products_view")

# COMMAND ----------

# DataFrame API: produtos com preco > 50
resultado_df = (
    products
    .filter(F.col("unit_price") > 50)
    .select("product_name", "unit_price")
    .orderBy(F.col("unit_price").desc())
)

display(resultado_df)

# COMMAND ----------

# spark.sql(): mesma consulta em SQL puro
resultado_sql = spark.sql("""
	SELECT product_name, unit_price
	FROM products_view
	WHERE unit_price > 50
	ORDER BY unit_price DESC
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Resumo do que fiz com ajuda da IA
# MAGIC
# MAGIC Neste bloco fiz:
# MAGIC
# MAGIC | Conceito | Funcao/Metodo |
# MAGIC |----------|---------------|
# MAGIC | Carregar tabela do catalogo | `spark.table("tabela")` |
# MAGIC | Ler arquivo CSV | `spark.read.csv(path, header=True, inferSchema=True)` |
# MAGIC | Ler arquivo JSON | `spark.read.json(path)` |
# MAGIC | Ler arquivo Parquet | `spark.read.parquet(path)` |
# MAGIC | Ler arquivo Delta | `spark.read.format("delta").load(path)` |
# MAGIC | Schema explicito | `StructType([StructField("col", Tipo(), nullable)])` |
# MAGIC | Visualizar dados | `display(df)`, `df.show(n, truncate=False)` |
# MAGIC | Inspecionar schema | `df.printSchema()`, `df.dtypes` |
# MAGIC | Listar colunas | `df.columns` |
# MAGIC | Contar linhas | `df.count()` |
# MAGIC | Estatisticas | `df.describe()` |
# MAGIC | SQL direto | `spark.sql("SELECT ...")` |
# MAGIC | Criar temp view | `df.createOrReplaceTempView("nome")` |

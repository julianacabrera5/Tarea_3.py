from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Inicialización de la Sesión de Spark optimizada
spark = SparkSession.builder \
    .appName("AnalisisPreciosSipsa2024") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ==============================================================================
# PIPELINE 1: INGESTA, LIMPIEZA Y NORMALIZACIÓN DE DATOS (HDFS)
# Descripción: Carga el dataset de Precios Mensuales SIPSA/DANE desde HDFS,
# normaliza los nombres de las columnas, limpia caracteres especiales y realiza
# el casteo de precios e identificadores de clasificación agronómica (CPC).
# Muestra: Total de registros procesados y esquema unificado (printSchema).
# ==============================================================================
print("\n" + "="*80)
print(">>> PIPELINE 1: INGESTA Y ESTRUCTURACIÓN DE PRECIOS SIPSA DANE (HDFS) <<<")
print("="*80)

hdfs_path = "hdfs://localhost:9000/Tarea3/mensual_24.csv"

# Lectura del DataFrame desde HDFS configurando el separador correcto ';'
df_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("sep", ";") \
    .option("encoding", "UTF-8") \
    .load(hdfs_path)

# Selección, limpieza de espacios/tildes y normalización de tipos
df_clean = df_raw.select(
    F.col("Fecha").alias("mes_anio"),
    F.col("Grupo").alias("grupo_alimento"),
    F.col("Producto").alias("producto"),
    F.col("CODIGO_CPC_AC").alias("codigo_cpc"),
    F.col("Mercado").alias("mercado_mayorista"),
    F.regexp_replace(F.col("Precio promedio por kilogramo*"), ",", ".").cast("double").alias("precio_prom_kg")
).filter(F.col("precio_prom_kg").isNotNull() & F.col("producto").isNotNull())

df_clean.cache()
print(f"Total de registros procesados y validados: {df_clean.count()}")
df_clean.printSchema()

# ==============================================================================
# PIPELINE 2: DIAGNÓSTICO ESTADÍSTICO POR GRUPO DE ALIMENTOS
# Descripción: Agrupa los precios por categoría agropecuaria y evalúa la media,
# desviación estándar, mediana (p50), percentil 95 y precio máximo por kilogramo.
# Muestra: Tabla comparativa de la estructura de precios por grupo agroalimentario.
# ==============================================================================
print("\n" + "="*80)
print(">>> PIPELINE 2: COMPORTAMIENTO ESTADÍSTICO DE LA CANASTA BÁSICA <<<")
print("="*80)

stats_grupo = df_clean.groupBy("grupo_alimento").agg(
    F.count("precio_prom_kg").alias("total_registros"),
    F.round(F.avg("precio_prom_kg"), 2).alias("precio_promedio_kg"),
    F.round(F.stddev("precio_prom_kg"), 2).alias("desv_estandar"),
    F.round(F.expr("percentile_approx(precio_prom_kg, 0.5)"), 2).alias("mediana_p50"),
    F.round(F.expr("percentile_approx(precio_prom_kg, 0.95)"), 2).alias("p95_alto"),
    F.round(F.max("precio_prom_kg"), 2).alias("max_precio_registrado")
).orderBy(F.col("precio_promedio_kg").desc())

stats_grupo.show(10, truncate=False)

# ==============================================================================
# PIPELINE 3: WINDOW FUNCTIONS - TOP MERCADOS MAYORISTAS MÁS COSTOSOS POR PRODUCTO
# Descripción: Aplica funciones de ventana (Window.partitionBy por Producto)
# para identificar los mercados o plazas de abasto que registran los precios
# más altos en productos de alta demanda (ej. Papa, Carne, Leche, Huevo).
# Muestra: Ranking Top 1 de mayor costo por producto en el país.
# ==============================================================================
print("\n" + "="*80)
print(">>> PIPELINE 3: RANKING DE MERCADOS MAYORISTAS CRÍTICOS (WINDOW FUNCTIONS) <<<")
print("="*80)

# Filtro en productos de consumo masivo
productos_clave = ["Carne de res", "Leche liquida", "Papa negra", "Huevo", "Arroz de primera"]
df_filtrado_prod = df_clean.filter(F.col("producto").isin(productos_clave))

windowSpec = Window.partitionBy("producto").orderBy(F.col("precio_prom_kg").desc())

df_ranked = df_filtrado_prod.withColumn("rank", F.rank().over(windowSpec)) \
    .filter(F.col("rank") == 1) \
    .drop("rank") \
    .select("producto", "grupo_alimento", "mercado_mayorista", "mes_anio", "precio_prom_kg") \
    .orderBy(F.col("precio_prom_kg").desc())

df_ranked.show(15, truncate=False)

# ==============================================================================
# PIPELINE 4: EVOLUCIÓN MENSUAL Y TENDENCIA DE PRECIOS EN 2024
# Descripción: Evalúa la trayectoria mensual del costo promedio por kg en el país
# para detectar picos inflacionarios o variaciones de oferta durante el año 2024.
# Muestra: Serie temporal consolidada por mes.
# ==============================================================================
print("\n" + "="*80)
print(">>> PIPELINE 4: TENDENCIA MENSUAL DE PRECIOS AGROPECUARIOS EN 2024 <<<")
print("="*80)

df_tendencia = df_clean.groupBy("mes_anio").agg(
    F.round(F.avg("precio_prom_kg"), 2).alias("promedio_nacional_kg"),
    F.count("producto").alias("total_reportes_mes")
).orderBy("mes_anio")

df_tendencia.show(12)

spark.stop()

# Databricks notebook source
with open("/dbfs/databricks-datasets/README.md") as f:
    x = ''.join(f.readlines())

print(x)

# COMMAND ----------

display(dbutils.fs.ls("/databricks-datasets"))

# comment by ashish

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS diamonds;
# MAGIC 
# MAGIC CREATE TABLE diamonds USING CSV OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true")

# COMMAND ----------

diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true")
diamonds.write.format("delta").save("/mnt/delta/diamonds")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS diamonds;
# MAGIC 
# MAGIC CREATE TABLE diamonds USING DELTA LOCATION '/mnt/delta/diamonds/'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT color, avg(price) AS price FROM diamonds GROUP BY color ORDER BY COLOR

# COMMAND ----------

# MAGIC %scala
# MAGIC val path = "abcdefg"

# COMMAND ----------

# MAGIC %r
# MAGIC assign("x", c(10.4, 5.6, 3.1, 6.4, 21.7))

# COMMAND ----------


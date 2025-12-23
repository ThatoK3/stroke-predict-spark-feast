# Inside Jupyter (http://localhost:8888) or any docker exec PySpark shell
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MinIO-Test") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# sample stroke data
data = [(1, 65, 180, 1), (2, 45, 160, 0)]
cols = ["patient_id", "age", "weight", "stroke"]
df = spark.createDataFrame(data, cols)

# write to MinIO bucket (creates bucket automatically if not exists)
df.write.mode("overwrite") \
  .parquet("s3a://stroke-data/patients.parquet")

# read back
read_df = spark.read.parquet("s3a://stroke-data/patients.parquet")
read_df.show()

spark.stop()


"""
# list buckets via MinIO client
docker run --rm --network stroke-network \
  quay.io/minio/mc:latest \
  mc alias set local http://stroke-minio:9000 admin password
docker run --rm --network stroke-network \
  quay.io/minio/mc:latest \
  mc ls local
"""

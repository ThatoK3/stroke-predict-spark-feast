from pyspark.sql import SparkSession
import time

def test_jmx_metrics():
    """Test Spark job with JMX monitoring"""
    spark = SparkSession.builder \
        .appName("JMXMetricsTest") \
        .master("spark://spark-master:7077") \
        .config("spark.driver.extraJavaOptions", 
                "-javaagent:/opt/jars/jmx-exporter/jmx_prometheus_javaagent-0.20.0.jar=7071:/opt/jars/jmx-exporter/spark-metrics.yml") \
        .config("spark.executor.extraJavaOptions",
                "-javaagent:/opt/jars/jmx-exporter/jmx_prometheus_javaagent-0.20.0.jar=7072:/opt/jars/jmx-exporter/spark-metrics.yml") \
        .getOrCreate()
    
    # Create some workload to generate metrics
    df = spark.range(1000000)
    df = df.withColumn("square", df["id"] * df["id"])
    
    # Force computation
    count = df.count()
    print(f"Processed {count} records")
    
    # Keep running to allow metric collection
    time.sleep(30)
    
    spark.stop()

if __name__ == "__main__":
    test_jmx_metrics()

from pyspark.sql import SparkSession

def test_spark():
    """Test basic Spark connectivity"""
    spark = SparkSession.builder \
        .appName("TestSpark") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    data = [("patient1", 45), ("patient2", 60)]
    df = spark.createDataFrame(data, ["id", "age"])
    df.show()
    
    print("Spark connection successful")
    spark.stop()

if __name__ == "__main__":
    test_spark()

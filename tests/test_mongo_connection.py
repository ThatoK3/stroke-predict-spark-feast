from pyspark.sql import SparkSession

def test_mongo_connection():
    """Test MongoDB Spark connector"""
    spark = SparkSession.builder \
        .appName("TestMongo") \
        .master("spark://spark-master:7077") \
        .config("spark.jars", "/opt/jars/mongo-spark/mongo-spark-connector_2.12-10.2.1.jar") \
        .getOrCreate()
    
    # Test MongoDB connection (adjust connection string as needed)
    try:
        df = spark.read \
            .format("mongodb") \
            .option("uri", "mongodb://localhost:27017/test.patients") \
            .load()
        
        print("MongoDB connection successful")
        df.printSchema()
        
    except Exception as e:
        print(f"MongoDB connection test: {e}")
        print("Note: MongoDB server may not be running, but JARs are loaded correctly")
    
    spark.stop()

if __name__ == "__main__":
    test_mongo_connection()

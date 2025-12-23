#!/bin/bash

echo "Setting up Spark-compatible JAR files..."

# Create directories
mkdir -p jars/mongo-spark
mkdir -p jars/jmx-exporter
mkdir -p jars/hadoop-aws

# Download MongoDB Spark Connector and dependencies
echo "Downloading MongoDB Spark Connector..."
cd jars/mongo-spark

# MongoDB Spark Connector for Spark 3.x
wget -q https://repo1.maven.org/maven2/org/mongodb/spark/mongo-spark-connector_2.12/10.2.1/mongo-spark-connector_2.12-10.2.1.jar

# MongoDB Java driver dependencies
wget -q https://repo1.maven.org/maven2/org/mongodb/bson/4.11.1/bson-4.11.1.jar
wget -q https://repo1.maven.org/maven2/org/mongodb/mongodb-driver-core/4.11.1/mongodb-driver-core-4.11.1.jar
wget -q https://repo1.maven.org/maven2/org/mongodb/mongodb-driver-sync/4.11.1/mongodb-driver-sync-4.11.1.jar

echo "MongoDB Spark Connector version: $(ls mongo-spark-connector*.jar | cut -d'-' -f4)"
cd ../..





# Download JMX Exporter for monitoring
echo "Downloading JMX Exporter for monitoring..."
cd jars/jmx-exporter
wget -q https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/0.20.0/jmx_prometheus_javaagent-0.20.0.jar

# Create basic JMX config
cat > config.yml << EOF
lowercaseOutputName: true
rules:
  - pattern: ".*"
EOF
cd ../..






# Download Minio/S3 Spark Connector
echo "Downloading Minio/S3 Spark Connecor..."
cd jars/hadoop-aws
# Hadoop AWS connector
wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
# AWS SDK bundle (includes S3 client)
wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
cd ../..



echo "Spark-compatible JARs downloaded successfully"
echo "Installed JARs:"
find jars -name "*.jar" -exec basename {} \;

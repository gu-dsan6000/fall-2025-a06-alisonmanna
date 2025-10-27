"""
Analyze the distribution of log levels (INFO, WARN, ERROR, DEBUG) across all log files. 
This problem requires basic PySpark operations and simple aggregations.

recommended approach:
1. Load log files as text files into Spark RDD or DataFrame 
2. Parse log entries (timestamp, log level, component, message) 
3. Apply transformations to extract insights 
4. Use Spark SQL for aggregations and queries 
5. Visualize results using matplotlib or similar tools

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, rand

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("Log_Level_Distribution_Cluster")

    # Cluster Configuration
    # .master(master_url)

    # Memory Configuration
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .config("spark.driver.maxResultSize", "2g")

    # Executor Configuration
    .config("spark.executor.cores", "2")
    .config("spark.cores.max", "6")  # Use all available cores 

    # S3 Configuration - Use S3A for AWS S3 access
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

    # Performance settings for cluster execution
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    # Serialization
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    # Arrow optimization for Pandas conversion
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")

    .getOrCreate()
)

def parse_logs(logs):
    # Parse log entries
    parsed_logs = logs.select(
        regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
        regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)', 1).alias('level'),
        regexp_extract('value', r'(INFO|WARN|ERROR|DEBUG)\s+([^:]+):', 2).alias('component'),
        col('value').alias('message')
    )
    parsed_logs = parsed_logs.filter(col('level') != "")
    return parsed_logs

def count_levels(parsed_logs):
    log_level_counts = parsed_logs.groupBy('level').count()
    log_level_counts.show()

    log_level_counts.toPandas().to_csv("data/output/problem1_counts.csv", index = False)

def sample_entries(parsed_logs):
    sample = parsed_logs[['message', 'level']].orderBy(rand()).limit(10)
    sample.toPandas().to_csv("data/output/problem1_sample.csv", index = False)

# # Find most active components
# component_counts = parsed_logs.groupBy('component').count().orderBy('count', ascending=False)
# component_counts.show(20)

def main():
    # Load log files (use data/sample/ for testing)
    logs_df = spark.read.text("data/sample/application_*/*.log")

    # Parse logs
    parsed_logs = parse_logs(logs_df)

    # Get log level counts
    count_levels(parsed_logs)

    # Sample 10 random log entries w/ levels
    sample_entries(parsed_logs)

if __name__ == "__main__":
    main()
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, rand

"""
Analyzes the distribution of log levels (INFO, WARN, ERROR, DEBUG) across all log files. 
This problem requires basic PySpark operations and simple aggregations.
"""


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

    log_level_counts.toPandas().to_csv("problem1_counts.csv", index = False)

    return log_level_counts

def sample_entries(parsed_logs):
    sample = parsed_logs[['message', 'level']].orderBy(rand()).limit(10)
    sample.toPandas().to_csv("problem1_sample.csv", index = False)

def summary_stats(logs, parsed_logs, level_counts):
    total_lines = logs.count()
    total_log_lines = parsed_logs.count()
    unique_levels = level_counts.count()

    output_file = 'problem1_summary.txt'
    with open(output_file, 'w') as f:
        print(f"Total log lines processed: {total_lines:,}", file=f)
        print(f"Total lines with log levels: {total_log_lines:,}", file=f)
        print(f"Unique log levels found: {unique_levels}", file=f)

        print("\nLog level distribution:", file=f)
        for row in level_counts.collect():
            percentage = (row['count'] / total_log_lines) * 100
            print(f"  {row['level']:<6}: {row['count']:>10,} ({percentage:5.2f}%)", file=f)

def main():
    # Load log files (use data/sample/ for testing)
    # logs_df = spark.read.text("data/sample/application_*/*.log")

    master_private_ip = os.getenv("MASTER_PRIVATE_IP", "172.31.80.218")
    if master_private_ip:
        master_url = f"spark://{master_private_ip}:7077"
    else:
        print("Error: No master URL provided")
        return 1
    
    # Initialize Spark
    spark = (
        SparkSession.builder
        .appName("Log_Level_Distribution_Cluster")

        # Cluster Configuration
        .master(master_url)

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

    try:
        s3_path = "s3a://abm119-assignment-spark-cluster-logs/data/*/*.log"
        logs_df = spark.read.text(s3_path)
        logs_df.cache()
    except Exception as e:
        print(f"Error getting log files from S3: {str(e)}")
        return 0

    # Parse logs
    try:
        parsed_logs = parse_logs(logs_df)
        parsed_logs.cache()
    except Exception as e:
        print(f"Error parsing the log files: {str(e)}")
        return 0

    # Get log level counts
    try:
        level_counts_df = count_levels(parsed_logs)
    except Exception as e:
        print(f"Error counting log levels: {str(e)}")
        return 0

    # Sample 10 random log entries w/ levels
    try:
        sample_entries(parsed_logs)
    except Exception as e:
        print(f"Error sampling log entries: {str(e)}")
        return 0

    # Produce summary statistics
    try:
        summary_stats(logs_df, parsed_logs, level_counts_df)
    except Exception as e:
        print(f"Error producing summary statistics: {str(e)}")
        return 0

    logs_df.unpersist()
    parsed_logs.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
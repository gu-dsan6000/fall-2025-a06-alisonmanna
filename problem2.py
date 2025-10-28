import os
import sys
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    regexp_extract, col, input_file_name, to_timestamp, explode, split,
    min as spark_min, max as spark_max, count as spark_count
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Analyzes cluster usage patterns to understand which clusters are most heavily used over time.
Extracts cluster IDs, application IDs, and application start/end times to create 
time-series dataset suitable for visualization with Seaborn.
"""

def extract_application_data(logs_df):
    # Add file path column to extract IDs
    df = logs_df.withColumn('file_path', input_file_name())

    # DEBUG: Print a sample of file paths to see what they look like
    print("Sample file paths:")
    df.select('file_path').distinct().show(5, truncate=False)
    
    # Extract application_id (including "application_" prefix)
    df = df.withColumn('application_id', regexp_extract('file_path', r'application_(\d+_\d+)', 0))

    # DEBUG: Check if extraction worked
    print("Sample application_ids:")
    df.select('application_id').distinct().show(5, truncate=False)
    
    # Get cluster_id and app_number from full application_id
    df = df.withColumn('cluster_id', regexp_extract(col('application_id'), r'application_(\d+)_', 1))

    df = df.withColumn('app_number', regexp_extract(col('application_id'), r'application_\d+_(\d+)', 1))

    # Extract timestamp from log entry
    df = df.withColumn('timestamp_str', regexp_extract(col('value'), r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1))

    # Filter rows for valid timestamps
    df = df.filter(col('timestamp_str') != "")

    # Convert to proper datetime
    df = df.withColumn('timestamp', to_timestamp(col('timestamp_str'), 'yy/MM/dd HH:mm:ss'))

    df = df.filter(col("timestamp").isNotNull())
    
    return df

def create_timeline(df):
    # Group by cluster_id, application_id, app_number, get min/max timestamps
    timeline = df.groupBy('cluster_id', 'application_id', 'app_number').agg(
        spark_min('timestamp').alias('start_time'),
        spark_max('timestamp').alias('end_time')
    )
    
    # Sort by cluster_id and app_number
    timeline = timeline.orderBy('cluster_id', 'app_number')
    
    return timeline

def create_cluster_summary(timeline):
    cluster_summary = timeline.groupBy('cluster_id').agg(
        spark_count('application_id').alias('num_applications'),
        spark_min('start_time').alias('cluster_first_app'),
        spark_max('end_time').alias('cluster_last_app')
    )
    
    # Sort by num of applications descending
    cluster_summary = cluster_summary.orderBy(col('num_applications').desc())
    
    return cluster_summary

def write_summary_stats(timeline_df, cluster_summary_df):
    # Convert to pandas df 
    timeline_pd = timeline_df.toPandas()
    cluster_summary_pd = cluster_summary_df.toPandas()
    
    total_clusters = cluster_summary_pd.shape[0]
    total_apps = timeline_pd.shape[0]
    avg_apps_per_cluster = total_apps / total_clusters if total_clusters > 0 else 0
    
    output_file = 'problem2_stats.txt'
    with open(output_file, 'w') as f:
        print(f"Total unique clusters: {total_clusters}", file=f)
        print(f"Total applications: {total_apps}", file=f)
        print(f"Average applications per cluster: {avg_apps_per_cluster:.2f}", file=f)
        print("", file=f)
        print("Most heavily used clusters:", file=f)
        
        for _, row in cluster_summary_pd.iterrows():
            cluster_id = row['cluster_id']
            num_apps = row['num_applications']
            print(f"  Cluster {cluster_id}: {num_apps} applications", file=f)

def create_visualizations(skip_spark=False):
    # Load data from CSV files
    cluster_summary = pd.read_csv('problem2_cluster_summary.csv')
    timeline = pd.read_csv('problem2_timeline.csv')

    timeline['start_time'] = pd.to_datetime(timeline['start_time'])
    timeline['end_time'] = pd.to_datetime(timeline['end_time'])
    
    # Calculate job duration
    timeline['duration_seconds'] = (timeline['end_time'] - timeline['start_time']).dt.total_seconds()
    
    sns.set_style("whitegrid")
    
    # Bar Chart, applications per cluster
    plt.figure(figsize=(10, 6))
    
    cluster_summary_sorted = cluster_summary.sort_values('num_applications', ascending=False)
    
    ax = sns.barplot(
        data=cluster_summary_sorted,
        x='cluster_id',
        y='num_applications',
        palette='viridis'
    )
    
    # Add value labels on top of bars
    for i, (idx, row) in enumerate(cluster_summary_sorted.iterrows()):
        ax.text(i, row['num_applications'] + 2, str(int(row['num_applications'])),
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Applications', fontsize=12)
    plt.title('Applications per Cluster', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('problem2_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Density Plot, job duration distribution for largest cluster
    # Find the largest cluster (most applications)
    largest_cluster_id = cluster_summary_sorted.iloc[0]['cluster_id']
    largest_cluster_data = timeline[timeline['cluster_id'] == largest_cluster_id].copy()
    
    # Filter out invalid durations
    largest_cluster_data = largest_cluster_data[largest_cluster_data['duration_seconds'] > 0]
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram + KDE overlay using log scale
    ax = sns.histplot(
        data=largest_cluster_data,
        x='duration_seconds',
        kde=True,
        bins=30,
        color='steelblue',
        log_scale=True
    )
    
    plt.xlabel('Job Duration (seconds, log scale)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Job Duration Distribution - Cluster {largest_cluster_id} (n={len(largest_cluster_data)})',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('problem2_density_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved:")
    print("  - problem2_bar_chart.png")
    print("  - problem2_density_plot.png")


def main():
    # command line arguments
    parser = argparse.ArgumentParser(description='Problem 2: Cluster Usage Analysis')
    parser.add_argument('master_url', nargs='?', default=None)
    parser.add_argument('--net-id', required=False)
    parser.add_argument('--skip-spark', action='store_true',help='Skip Spark processing and only regenerate visualizations')
    
    args = parser.parse_args()
    
    # If skip-spark mode, just regenerate visualizations
    if args.skip_spark:
        print("Skipping Spark processing, regenerating visualizations from existing CSVs")
        try:
            create_visualizations(skip_spark=True)
            print("Done!")
            return 0
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            return 1
    
    # Validate arguments for Spark mode
    if not args.master_url:
        print("Error: Master URL is required (or use --skip-spark)")
        print("Usage: python problem2.py spark://MASTER_IP:7077 --net-id YOUR-NET-ID")
        return 1
    
    if not args.net_id:
        print("Error: --net-id is required")
        print("Usage: python problem2.py spark://MASTER_IP:7077 --net-id YOUR-NET-ID")
        return 1
    
    master_url = args.master_url
    net_id = args.net_id
    
    # Initialize Spark
    spark = (
        SparkSession.builder
        .appName("Cluster_Usage_Analysis")
        
        # Cluster Configuration
        .master(master_url)
        
        # Memory Configuration
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        
        # Executor Configuration
        .config("spark.executor.cores", "2")
        .config("spark.cores.max", "6")
        
        # S3 Configuration
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        
        # Performance settings
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # Serialization
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
        # Arrow optimization
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        .getOrCreate()
    )
    
    try:
        # Load log files from S3
        s3_path = f"s3a://{net_id}-assignment-spark-cluster-logs/data/*/*.log"
        print(f"Loading logs from: {s3_path}")
        logs_df = spark.read.text(s3_path)
        print(f"Loaded {logs_df.count():,} log lines")
    except Exception as e:
        print(f"Error loading log files from S3: {str(e)}")
        spark.stop()
        return 1

    # logs_df = spark.read.text("data/sample/application_*/*.log")
    
    try:
        # Extract application data with timestamps
        print("Extracting application data & timestamps")
        app_data = extract_application_data(logs_df)
        app_data.cache()
    except Exception as e:
        print(f"Error extracting application data: {str(e)}")
        logs_df.unpersist()
        spark.stop()
        return 1
    
    try:
        # Create timeline
        print("Creating timeline data")
        timeline_df = create_timeline(app_data)
        timeline_df.cache()
        
        # Save timeline to CSV
        timeline_df.toPandas().to_csv("problem2_timeline.csv", index=False)
        print("Saved problem2_timeline.csv")
    except Exception as e:
        print(f"Error creating timeline: {str(e)}")
        logs_df.unpersist()
        app_data.unpersist()
        spark.stop()
        return 1
    
    try:
        # Create cluster summary
        print("Creating cluster summary")
        cluster_summary_df = create_cluster_summary(timeline_df)
        cluster_summary_df.cache()
        
        # Save cluster summary to CSV
        cluster_summary_df.toPandas().to_csv("problem2_cluster_summary.csv", index=False)
        print("Saved problem2_cluster_summary.csv")
    except Exception as e:
        print(f"Error creating cluster summary: {str(e)}")
        logs_df.unpersist()
        app_data.unpersist()
        timeline_df.unpersist()
        spark.stop()
        return 1
    
    try:
        # Write summary statistics
        print("Writing summary statistics")
        write_summary_stats(timeline_df, cluster_summary_df)
    except Exception as e:
        print(f"Error writing summary statistics: {str(e)}")
        logs_df.unpersist()
        app_data.unpersist()
        timeline_df.unpersist()
        cluster_summary_df.unpersist()
        spark.stop()
        return 1
    
    logs_df.unpersist()
    app_data.unpersist()
    timeline_df.unpersist()
    cluster_summary_df.unpersist()
    spark.stop()
    
    # Create visualizations
    print("Creating visualizations")
    try:
        create_visualizations()
        print("Done!")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
spark_etl.py — Spark ETL Pipeline (RDD + SQL)
INT315 Cluster Computing | Step 2
Demonstrates RDD transformations (map, filter, reduceByKey, broadcast)
and Spark SQL (JOIN, GROUP BY, ORDER BY, aggregates) on financial data.
"""

from pyspark.sql import SparkSession

# ── Initialize Spark ─────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("FinancialFraudETL")
    .master("local[1]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext

print("=" * 60)
print("STAGE 2 — Spark ETL (RDD + SQL)")
print("=" * 60)

# =====================================================================
# SECTION A — RDD TRANSFORMATIONS (Unit III)
# =====================================================================
print("\n--- RDD Section ---")

# Load transactions as raw text RDD, skip header
raw_rdd = sc.textFile("data/raw/transactions.csv")
header = raw_rdd.first()
txn_rdd = raw_rdd.filter(lambda line: line != header)

# Parse each line into a tuple:
# (tx_id, src_id, dst_id, amount, tx_type, country_code, is_fraud)
def parse_txn(line):
    parts = line.split(",")
    return (
        int(parts[0]),       # tx_id
        int(parts[1]),       # src_id
        int(parts[2]),       # dst_id
        float(parts[3]),     # amount
        parts[4],            # tx_type
        parts[5],            # country_code
        int(parts[6]),       # is_fraud
    )

parsed_rdd = txn_rdd.map(parse_txn)

print(f"  Total transactions loaded into RDD: {parsed_rdd.count():,}")

# Broadcast high-risk country codes
high_risk_countries = sc.broadcast({"KP", "IR", "SY", "NG", "RU"})

# Filter: high-risk country OR high-value (amount > 5000)
high_risk_rdd = parsed_rdd.filter(
    lambda t: t[5] in high_risk_countries.value or t[3] > 5000
)
print(f"  High-risk / high-value transactions: {high_risk_rdd.count():,}")

# map + reduceByKey: total outgoing volume per src_id
outgoing_volume = (
    high_risk_rdd
    .map(lambda t: (t[1], t[3]))          # (src_id, amount)
    .reduceByKey(lambda a, b: a + b)       # sum amounts per src_id
)

# Top 5 accounts by outgoing volume
top5 = outgoing_volume.takeOrdered(5, key=lambda x: -x[1])
print("\n  Top 5 accounts by outgoing volume (high-risk/high-value):")
for rank, (src_id, vol) in enumerate(top5, 1):
    print(f"    {rank}. Account {src_id:>6}  ->  ${vol:>14,.2f}")



# =====================================================================
# SECTION B — SPARK SQL (Unit IV)
# =====================================================================
print("\n--- Spark SQL Section ---")

# Load CSVs as DataFrames
accounts_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/raw/accounts.csv")
)

transactions_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/raw/transactions.csv")
)

print(f"  Accounts DataFrame : {accounts_df.count():,} rows, "
      f"columns = {accounts_df.columns}")
print(f"  Transactions DF    : {transactions_df.count():,} rows, "
      f"columns = {transactions_df.columns}")

# Register temp views
accounts_df.createOrReplaceTempView("accounts")
transactions_df.createOrReplaceTempView("transactions")

# ── Aggregation Query: per-account behavioral features ───────────────
# Uses JOIN, GROUP BY, ORDER BY, COUNT, SUM, AVG, MAX
sql_features = spark.sql("""
    SELECT
        a.account_id,
        a.account_type,
        a.risk_tier,
        a.account_age_days,

        /* Behavioral aggregates from outgoing transactions */
        COALESCE(agg.tx_count, 0)            AS tx_count,
        COALESCE(agg.total_sent, 0.0)        AS total_sent,
        COALESCE(agg.avg_amount, 0.0)        AS avg_amount,
        COALESCE(agg.max_amount, 0.0)        AS max_amount,
        COALESCE(agg.high_risk_tx_count, 0)  AS high_risk_tx_count,

        /* Target label: 1 if account was source of ANY fraud tx */
        COALESCE(agg.is_fraud_account, 0)    AS is_fraud_account

    FROM accounts a
    LEFT JOIN (
        SELECT
            src_id,
            COUNT(*)                                            AS tx_count,
            SUM(amount)                                         AS total_sent,
            AVG(amount)                                         AS avg_amount,
            MAX(amount)                                         AS max_amount,
            SUM(CASE
                WHEN country_code IN ('KP','IR','SY','NG','RU')
                THEN 1 ELSE 0
            END)                                                AS high_risk_tx_count,
            MAX(is_fraud)                                       AS is_fraud_account
        FROM transactions
        GROUP BY src_id
    ) agg ON a.account_id = agg.src_id

    ORDER BY a.account_id
""")

sql_features.cache()
print(f"\n  SQL Features DataFrame: {sql_features.count():,} rows")
sql_features.show(10, truncate=False)

# Print fraud distribution
fraud_dist = spark.sql("""
    SELECT
        COALESCE(MAX(is_fraud), 0) AS is_fraud_account,
        COUNT(*) AS account_count
    FROM accounts a
    LEFT JOIN transactions t ON a.account_id = t.src_id
    GROUP BY a.account_id
""")
print("  Fraud account distribution:")
fraud_dist.groupBy("is_fraud_account").count().show()

# ── Save to Parquet ──────────────────────────────────────────────────
sql_features.write.mode("overwrite").parquet("data/processed/sql_features.parquet")
print("  ✓ Saved: data/processed/sql_features.parquet")

sql_features.unpersist()

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ Step 2 COMPLETE — Spark ETL finished successfully!")
print("=" * 60)

try:
    spark.stop()
except Exception:
    pass

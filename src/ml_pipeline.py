"""
src/ml_pipeline.py — PySpark MLlib Pipeline with GraphX Feature Ablation
INT315 Cluster Computing | Step 4
Compares Baseline (SQL only) vs Enhanced (+ GraphX metrics) models,
benchmarks CPU core scaling (local[1], local[2], local[4]), and saves plots.
"""

import sys
import os
import time

# Ensure Spark internal python libraries are discoverable when run directly via python
spark_home = os.environ.get("SPARK_HOME", r"C:\Spark\spark-4.2.0-bin-hadoop3")
spark_python = os.path.join(spark_home, "python")
py4j_zip = None
py4j_dir = os.path.join(spark_python, "lib")
if os.path.exists(py4j_dir):
    for f in os.listdir(py4j_dir):
        if f.startswith("py4j") and f.endswith(".zip"):
            py4j_zip = os.path.join(py4j_dir, f)
            break

for p in [spark_python, py4j_zip]:
    if p and p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ── Matplotlib Non-Interactive Agg Backend ───────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 4 — PySpark MLlib Fraud Detection Pipeline")
print("=" * 60)

# ── Initialize Spark Session ─────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("FraudDetectionML")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# ── 1. Load Parquet Data ─────────────────────────────────────────────
print("\n[1/4] Loading Parquet Features...")
sql_df = spark.read.parquet("data/processed/sql_features.parquet")
graph_df = spark.read.parquet("data/processed/graph_features.parquet")

print(f"  SQL Features Loaded   : {sql_df.count():,} rows")
print(f"  Graph Features Loaded : {graph_df.count():,} rows")

# Inner join on account_id + fill post-join nulls
joined = sql_df.join(graph_df, on="account_id", how="inner").na.fill(0)

# Feature Engineering: In/Out Ratio + Connected Component Size
joined = joined.withColumn(
    "in_out_ratio",
    F.col("in_degree") / (F.col("out_degree") + F.lit(1.0))
)

comp_counts = joined.groupBy("component_id").agg(F.count("*").alias("component_size"))
joined = joined.join(comp_counts, on="component_id", how="left").na.fill(0)

# Index Categoricals
for col_name in ["account_type", "risk_tier"]:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
    joined = indexer.fit(joined).transform(joined)

joined = joined.na.fill(0).cache()
print(f"  Joined Dataset Schema verified ({joined.count():,} rows)")

# ── 2. Experiment 1: GraphX Feature Ablation ─────────────────────────
print("\n[2/4] Running Experiment 1: GraphX Feature Ablation...")

baseline_features = [
    "tx_count", "total_sent", "avg_amount", "max_amount",
    "high_risk_tx_count", "account_age_days",
    "account_type_idx", "risk_tier_idx"
]

graphx_features = [
    "pagerank", "in_degree", "out_degree",
    "in_out_ratio", "component_size"
]

enhanced_features = baseline_features + graphx_features
label_col = "is_fraud_account"

train_df, test_df = joined.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

auc_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
f1_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
acc_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")

ablation_results = []
best_model_preds = None
best_auc = 0.0
best_model_name = ""

for fset_name, fcols in [("Baseline", baseline_features), ("GraphX-Enhanced", enhanced_features)]:
    assembler = VectorAssembler(inputCols=fcols, outputCol="features", handleInvalid="skip")
    
    # Train Logistic Regression
    lr = LogisticRegression(labelCol=label_col, featuresCol="features", maxIter=20)
    lr_pipeline = Pipeline(stages=[assembler, lr])
    lr_model = lr_pipeline.fit(train_df)
    lr_preds = lr_model.transform(test_df)
    
    lr_auc = auc_evaluator.evaluate(lr_preds)
    lr_f1 = f1_evaluator.evaluate(lr_preds)
    lr_acc = acc_evaluator.evaluate(lr_preds)
    fraud_precision = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0
    ).evaluate(lr_preds)
    fraud_recall = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0
    ).evaluate(lr_preds)
    
    ablation_results.append({
        "Model": "Logistic Regression",
        "FeatureSet": fset_name,
        "Accuracy": round(lr_acc, 4),
        "F1_Score": round(lr_f1, 4),
        "Fraud_Precision": round(fraud_precision, 4),
        "Fraud_Recall": round(fraud_recall, 4),
        "ROC_AUC": round(lr_auc, 4)
    })
    
    if lr_auc > best_auc:
        best_auc = lr_auc
        best_model_preds = lr_preds
        best_model_name = f"Logistic Regression ({fset_name})"
        
    # Train Decision Tree Classifier
    dt = DecisionTreeClassifier(labelCol=label_col, featuresCol="features", maxDepth=8)
    dt_pipeline = Pipeline(stages=[assembler, dt])
    dt_model = dt_pipeline.fit(train_df)
    dt_preds = dt_model.transform(test_df)
    
    dt_auc = auc_evaluator.evaluate(dt_preds)
    dt_f1 = f1_evaluator.evaluate(dt_preds)
    dt_acc = acc_evaluator.evaluate(dt_preds)
    
    ablation_results.append({
        "Model": "Decision Tree",
        "FeatureSet": fset_name,
        "Accuracy": round(dt_acc, 4),
        "F1_Score": round(dt_f1, 4),
        "ROC_AUC": round(dt_auc, 4)
    })
    
    if dt_auc > best_auc:
        best_auc = dt_auc
        best_model_preds = dt_preds
        best_model_name = f"Decision Tree ({fset_name})"

results_df = pd.DataFrame(ablation_results)
print("\n" + "=" * 65)
print("             GRAPHX FEATURE ABLATION STUDY RESULTS")
print("=" * 65)
print(results_df.to_string(index=False))
print("=" * 65)

# ── 3. Visualizations ────────────────────────────────────────────────
print("\n[3/4] Generating Visualizations...")

# 1. Confusion Matrix
preds_rows = best_model_preds.select(label_col, "prediction").collect()
preds_pd = pd.DataFrame([r.asDict() for r in preds_rows])
from sklearn.metrics import confusion_matrix as sk_cm
cm = sk_cm(preds_pd[label_col].astype(int), preds_pd["prediction"].astype(int))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"])
plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
plt.title(f"Confusion Matrix: {best_model_name}\n(ROC-AUC: {best_auc:.4f})", fontsize=12, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"  ✓ Saved: {cm_path}")

# 2. Correlation Heatmap
corr_cols = [
    "tx_count", "total_sent", "avg_amount", "max_amount",
    "high_risk_tx_count", "pagerank", "in_degree", "out_degree",
    "in_out_ratio", "component_size", "is_fraud_account"
]
corr_rows = joined.select(corr_cols).collect()
corr_pd = pd.DataFrame([r.asDict() for r in corr_rows]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_pd, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap: Behavioral + GraphX Features vs. Fraud", fontsize=12, fontweight="bold")
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=200)
plt.close()
print(f"  ✓ Saved: {heatmap_path}")

# 3. Ablation Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for idx, metric in enumerate(["ROC_AUC", "F1_Score"]):
    ax = axes[idx]
    pivot = results_df.pivot(index="Model", columns="FeatureSet", values=metric)
    pivot = pivot[["Baseline", "GraphX-Enhanced"]]
    pivot.plot(kind="bar", ax=ax, rot=0, color=["#7293CB", "#D35E60"], edgecolor="black")
    ax.set_title(f"Model Comparison: {metric}", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(0.7, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
ablation_path = os.path.join(OUTPUT_DIR, "ablation_comparison.png")
plt.savefig(ablation_path, dpi=200)
plt.close()
print(f"  ✓ Saved: {ablation_path}")

# ── 4. Experiment 2: Data-Volume Scaling Benchmark ───────────────────
print("\n[4/4] Running Experiment 2: Data-Volume Scaling Benchmark...")

benchmark_data = []

assembler_bench = VectorAssembler(inputCols=enhanced_features, outputCol="features", handleInvalid="skip")
dt_bench = DecisionTreeClassifier(labelCol=label_col, featuresCol="features", maxDepth=8)
pipe_bench = Pipeline(stages=[assembler_bench, dt_bench])

# Compare training on increasing fractions of the same fixed train/test split.
# This measures data-volume scaling on the configured local[2] session; it does
# not claim to compare cluster sizes or core counts.
for fraction in [0.25, 0.5, 1.0]:
    t_start = time.time()
    sample_train = train_df.sample(False, fraction, seed=42).cache()
    sample_count = sample_train.count()
    m = pipe_bench.fit(sample_train)
    preds = m.transform(test_df)
    auc_score = auc_evaluator.evaluate(preds)
    elapsed = round(time.time() - t_start, 2)
    sample_train.unpersist()
    
    benchmark_data.append({
        "Train_Fraction": fraction,
        "Train_Rows": sample_count,
        "Spark_Master": spark.sparkContext.master,
        "Execution_Time_Sec": elapsed,
        "ROC_AUC": round(auc_score, 4)
    })
    print(f"  {fraction:>5.0%} rows ({sample_count:,}) -> Time: {elapsed:5.2f}s | ROC-AUC: {auc_score:.4f}")

bench_df = pd.DataFrame(benchmark_data)
bench_csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
bench_df.to_csv(bench_csv_path, index=False)
print(f"\n  ✓ Saved: {bench_csv_path}")
print("=" * 60)
print(bench_df.to_string(index=False))
print("=" * 60)

print("\n  ✅ Step 4 COMPLETE — PySpark MLlib Pipeline finished successfully!")
print("=" * 60)

try:
    spark.stop()
except Exception:
    pass

# INT315 Cluster Computing CA2

## Fraud detection with Spark, GraphX, and MLlib

This project demonstrates a batch fraud analytics workflow over a reproducible synthetic dataset. It creates 15,000 accounts and 200,000 transfers, computes account behavior with Spark RDDs and Spark SQL, extracts suspicious-network metrics with Scala GraphX, and compares MLlib classifiers with and without graph features.

## Architecture

```text
Synthetic generator
       |
       +--> accounts.csv + transactions.csv
                    |
        +-----------+------------+
        |                        |
  PySpark RDD + SQL         Scala GraphX
  account behavior          suspicious-edge graph
        |                        |
        +------ Parquet features+
                    |
          PySpark MLlib models
                    |
   metrics, charts, scaling results
```

## Requirements

- Windows, Python 3.10 or newer, Java 17, Apache Spark 4.x with GraphX
- Python packages: `numpy`, `pandas`, `pyspark`, `matplotlib`, `seaborn`, `scikit-learn`
- Ensure `python`, `spark-submit`, and `spark-shell` are on `PATH`. If Python is not named `python`, update the interpreter setting at the top of `run_all.ps1`.

## Run

From this project directory, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

The script runs the four stages in order and stops if a stage fails:

1. `src/generate_data.py` creates the synthetic CSV files in `data/raw/`.
2. `src/spark_etl.py` demonstrates RDD transformations, broadcast variables, and Spark SQL joins and aggregations; writes `data/processed/sql_features.parquet/`.
3. `src/GraphAnalytics.scala` creates a graph of transfers of at least $5,000 and computes PageRank, connected components, and degrees; writes `data/processed/graph_features.parquet/`.
4. `src/ml_pipeline.py` evaluates logistic regression and decision trees, compares behavioral features with behavioral plus GraphX features, and writes figures and CSV results in `outputs/`.

## Evaluation

The model uses a seeded 80/20 account-level split. Reported measures include accuracy, F1, fraud-class precision and recall, ROC-AUC, a confusion matrix, and feature correlations. An ablation chart compares the baseline and GraphX-enhanced models. A separate experiment fits the decision tree on 25%, 50%, and 100% of the fixed training split using the same `local[2]` Spark session. This is a data-volume scaling check; it does not claim a multi-node or CPU-core scaling result.

The `outputs/` folder contains:

- `confusion_matrix.png`
- `correlation_heatmap.png`
- `ablation_comparison.png`
- `benchmark_results.csv`

## Data and limitations

All data is generated locally and is synthetic. Fraud labels and ring structure are planted by the generator to make the pipeline demonstrable; the scores are not evidence of real-world fraud detection performance. Transaction rows are used to derive an account-level label indicating whether the account sent at least one fraud transaction. The system is a local academic demonstration, not a production deployment or distributed-cluster benchmark.

## Course alignment

The implementation demonstrates the project brief's required Spark processing, GraphX analysis, Spark MLlib modeling, and experimental evaluation. It also covers syllabus topics including RDD operations, Spark SQL aggregation and joins, Scala GraphX, MLlib classification, a confusion matrix, and a correlation heatmap.

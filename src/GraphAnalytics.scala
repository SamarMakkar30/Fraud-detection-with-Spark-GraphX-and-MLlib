// src/GraphAnalytics.scala — Scala GraphX Analytics
// INT315 Cluster Computing | Step 3
// Run with: spark-shell -i src/GraphAnalytics.scala

sc.setLogLevel("ERROR")

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import spark.implicits._

println("=" * 60)
println("STAGE 3 — GraphX Analytics (Scala)")
println("=" * 60)

// ── 1. Read Raw CSVs ────────────────────────────────────────────────
val accountsDF = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("data/raw/accounts.csv")

val transactionsDF = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("data/raw/transactions.csv")

println(s"  Accounts loaded     : ${accountsDF.count()} rows")
println(s"  Transactions loaded : ${transactionsDF.count()} rows")

// ── 2. Construct Vertices and Edges RDDs ─────────────────────────────
val vertices: RDD[(VertexId, String)] = accountsDF.rdd.map { r =>
  val id = r.getAs[Number]("account_id").longValue()
  val atype = r.getAs[String]("account_type")
  (id, atype)
}

// Keep suspicious/high-value transfers for graph structure. The full transaction
// table is still used by the SQL feature stage; this focused graph avoids making
// every account part of one giant component through random low-value transfers.
val edges: RDD[Edge[Double]] = transactionsDF.rdd.filter { r =>
  r.getAs[Number]("amount").doubleValue() >= 5000.0
}.map { r =>
  val src = r.getAs[Number]("src_id").longValue()
  val dst = r.getAs[Number]("dst_id").longValue()
  val amount = r.getAs[Number]("amount").doubleValue()
  Edge(src, dst, amount)
}

// ── 3. Build Graph ──────────────────────────────────────────────────
val graph: Graph[String, Double] = Graph(vertices, edges, "Unknown").cache()

println(s"  Graph Vertices Count: ${graph.vertices.count()}")
println(s"  Suspicious Graph Edges Count: ${graph.edges.count()}")

// Print Top 5 Triplets by transaction amount
println("\n  Top 5 Triplets by Transaction Amount:")
graph.triplets
  .sortBy(_.attr, ascending = false)
  .take(5)
  .foreach { t =>
    println(f"    Account ${t.srcId}%6d (${t.srcAttr}%-10s) --[ $$${t.attr}%,10.2f ]--> Account ${t.dstId}%6d (${t.dstAttr}%-10s)")
  }

// ── 4. Compute Graph Metrics ────────────────────────────────────────
println("\n  Computing PageRank (tol = 0.001)...")
val prVertices: RDD[(VertexId, Double)] = graph.pageRank(0.001).vertices

println("  Computing Connected Components...")
val ccVertices: RDD[(VertexId, VertexId)] = graph.connectedComponents().vertices

println("  Computing In-Degrees and Out-Degrees...")
val inDeg: RDD[(VertexId, Int)] = graph.inDegrees
val outDeg: RDD[(VertexId, Int)] = graph.outDegrees

// ── 5. Combine via leftOuterJoin with getOrElse Defaults ────────────
println("  Combining vertex metrics via leftOuterJoin...")

val combined: RDD[(VertexId, (Double, Long, Int, Int))] = graph.vertices
  .leftOuterJoin(prVertices)
  .mapValues { case (_, prOpt) => prOpt.getOrElse(0.0) }
  .leftOuterJoin(ccVertices)
  .mapValues { case (pr, ccOpt) => (pr, ccOpt.getOrElse(0L)) }
  .leftOuterJoin(inDeg)
  .mapValues { case ((pr, cc), inOpt) => (pr, cc, inOpt.getOrElse(0)) }
  .leftOuterJoin(outDeg)
  .mapValues { case ((pr, cc, ind), outOpt) => (pr, cc, ind, outOpt.getOrElse(0)) }

// ── 6. Convert to DataFrame and Save Parquet ────────────────────────
val graphFeaturesDF = combined.map { case (vid, (pr, cc, ind, outd)) =>
  (vid, pr, cc, ind, outd)
}.toDF("account_id", "pagerank", "component_id", "in_degree", "out_degree")

println(s"\n  Graph Features Sample (Total Rows: ${graphFeaturesDF.count()}):")
graphFeaturesDF.show(10, truncate = false)

graphFeaturesDF.write
  .mode("overwrite")
  .parquet("data/processed/graph_features.parquet")

println("  ✓ Saved: data/processed/graph_features.parquet")
println("=" * 60)
println("  ✅ Step 3 COMPLETE — Scala GraphX finished successfully!")
println("=" * 60)

sys.exit(0)

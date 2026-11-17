"""
generate_data.py — Financial Fraud Detection Data Generator
INT315 Cluster Computing | Step 1
Generates 15,000 accounts and 200,000 transactions with 40 embedded
fraud rings (~4-6% overall fraud rate) so GraphX PageRank and
Connected Components strongly correlate with fraud.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
rng = np.random.RandomState(SEED)

# ── Parameters ───────────────────────────────────────────────────────
NUM_ACCOUNTS = 15_000
NUM_TRANSACTIONS = 200_000
NUM_FRAUD_RINGS = 40
RING_SIZE_MIN = 5
RING_SIZE_MAX = 10
# Target ~4-6% fraud rate; ring transactions + some extra scatter fraud
FRAUD_SCATTER_RATE = 0.005  # small extra scatter on top of ring fraud

ACCOUNT_TYPES = ["Savings", "Current", "Business"]
RISK_TIERS = ["Low", "Medium", "High"]
TX_TYPES = ["Transfer", "Payment", "CashOut"]
COUNTRY_CODES = ["US", "GB", "IN", "DE", "NG", "BR", "CN", "RU", "AE", "JP"]

OUTPUT_DIR = "data/raw"

# ── 1. Generate Accounts ────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 — Generating Accounts & Transactions")
print("=" * 60)

account_ids = np.arange(1, NUM_ACCOUNTS + 1)
accounts_df = pd.DataFrame({
    "account_id": account_ids,
    "account_type": rng.choice(ACCOUNT_TYPES, size=NUM_ACCOUNTS),
    "risk_tier": rng.choice(RISK_TIERS, size=NUM_ACCOUNTS, p=[0.5, 0.35, 0.15]),
    "account_age_days": rng.randint(30, 3650, size=NUM_ACCOUNTS),
})

print(f"  ✓ Accounts generated: {len(accounts_df):,} rows")

# ── 2. Build Fraud Rings ────────────────────────────────────────────
# Each ring: 5-10 accounts forming a dense circular money-mule chain
# with rapid high-value transfers between consecutive members.
fraud_ring_accounts = set()
fraud_ring_transactions = []
tx_counter = 0  # will be offset later

for ring_id in range(NUM_FRAUD_RINGS):
    ring_size = rng.randint(RING_SIZE_MIN, RING_SIZE_MAX + 1)
    # Pick ring members from the full account pool (no overlap between rings)
    available = list(set(account_ids) - fraud_ring_accounts)
    members = rng.choice(available, size=ring_size, replace=False)
    fraud_ring_accounts.update(members)

    # Each ring generates dense circular transfers:
    # member[0]->member[1]->...->member[n-1]->member[0]  (full cycle)
    # Plus random cross-links within the ring for extra density
    num_cycles = rng.randint(8, 20)  # repeat the cycle multiple times
    for _ in range(num_cycles):
        for i in range(ring_size):
            src = members[i]
            dst = members[(i + 1) % ring_size]
            amount = round(rng.uniform(5000, 50000), 2)  # high-value
            tx_type = rng.choice(["Transfer", "CashOut"], p=[0.7, 0.3])
            country = rng.choice(["NG", "RU", "AE", "CN"], p=[0.3, 0.3, 0.2, 0.2])
            fraud_ring_transactions.append({
                "src_id": int(src),
                "dst_id": int(dst),
                "amount": amount,
                "tx_type": tx_type,
                "country_code": country,
                "is_fraud": 1,
            })

    # Add extra random cross-links within ring (makes component dense)
    num_cross = rng.randint(5, 15)
    for _ in range(num_cross):
        src, dst = rng.choice(members, size=2, replace=False)
        amount = round(rng.uniform(8000, 60000), 2)
        fraud_ring_transactions.append({
            "src_id": int(src),
            "dst_id": int(dst),
            "amount": amount,
            "tx_type": rng.choice(["Transfer", "CashOut"]),
            "country_code": rng.choice(["NG", "RU", "AE"]),
            "is_fraud": 1,
        })

num_fraud_ring_txns = len(fraud_ring_transactions)
print(f"  ✓ Fraud rings created: {NUM_FRAUD_RINGS} rings, "
      f"{len(fraud_ring_accounts)} unique accounts, "
      f"{num_fraud_ring_txns:,} ring transactions")

# ── 3. Generate Legitimate Transactions ─────────────────────────────
num_legit_needed = NUM_TRANSACTIONS - num_fraud_ring_txns
# A few scattered fraud transactions outside rings
num_scatter_fraud = int(num_legit_needed * FRAUD_SCATTER_RATE)
num_clean = num_legit_needed - num_scatter_fraud

legit_transactions = []

# Clean transactions: normal amounts, diverse countries
for _ in range(num_clean):
    src = rng.randint(1, NUM_ACCOUNTS + 1)
    dst = rng.randint(1, NUM_ACCOUNTS + 1)
    while dst == src:
        dst = rng.randint(1, NUM_ACCOUNTS + 1)
    legit_transactions.append({
        "src_id": int(src),
        "dst_id": int(dst),
        "amount": round(rng.exponential(500) + 10, 2),  # mostly small
        "tx_type": rng.choice(TX_TYPES, p=[0.4, 0.4, 0.2]),
        "country_code": rng.choice(COUNTRY_CODES),
        "is_fraud": 0,
    })

# Scatter fraud: suspicious but not in rings
for _ in range(num_scatter_fraud):
    src = rng.randint(1, NUM_ACCOUNTS + 1)
    dst = rng.randint(1, NUM_ACCOUNTS + 1)
    while dst == src:
        dst = rng.randint(1, NUM_ACCOUNTS + 1)
    legit_transactions.append({
        "src_id": int(src),
        "dst_id": int(dst),
        "amount": round(rng.uniform(3000, 40000), 2),
        "tx_type": rng.choice(["Transfer", "CashOut"]),
        "country_code": rng.choice(["NG", "RU", "AE", "CN"]),
        "is_fraud": 1,
    })

# ── 4. Combine & Shuffle ────────────────────────────────────────────
all_txns = fraud_ring_transactions + legit_transactions
rng.shuffle(all_txns)

transactions_df = pd.DataFrame(all_txns)
transactions_df.insert(0, "tx_id", range(1, len(transactions_df) + 1))

# Ensure exact row count matches target
assert len(transactions_df) == NUM_TRANSACTIONS, \
    f"Expected {NUM_TRANSACTIONS} txns, got {len(transactions_df)}"

print(f"  ✓ Total transactions: {len(transactions_df):,} rows")

# ── 5. Save CSVs ────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

accounts_path = os.path.join(OUTPUT_DIR, "accounts.csv").replace("\\", "/")
transactions_path = os.path.join(OUTPUT_DIR, "transactions.csv").replace("\\", "/")

accounts_df.to_csv(accounts_path, index=False)
transactions_df.to_csv(transactions_path, index=False)

print(f"\n  ✓ Saved: {accounts_path}")
print(f"  ✓ Saved: {transactions_path}")

# ── 6. Verification ─────────────────────────────────────────────────
fraud_count = transactions_df["is_fraud"].sum()
fraud_pct = fraud_count / len(transactions_df) * 100

print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"  Accounts file rows      : {len(accounts_df):,}")
print(f"  Transactions file rows  : {len(transactions_df):,}")
print(f"  Fraud transactions      : {fraud_count:,}")
print(f"  Legitimate transactions : {len(transactions_df) - fraud_count:,}")
print(f"  Fraud rate              : {fraud_pct:.2f}%")
print(f"  Fraud rings embedded    : {NUM_FRAUD_RINGS}")
print(f"  Ring accounts           : {len(fraud_ring_accounts)}")
print(f"  Account types           : {accounts_df['account_type'].value_counts().to_dict()}")
print(f"  Risk tiers              : {accounts_df['risk_tier'].value_counts().to_dict()}")
print(f"  Transaction types       : {transactions_df['tx_type'].value_counts().to_dict()}")
print("=" * 60)
print("  ✅ Step 1 COMPLETE — Data generation successful!")
print("=" * 60)

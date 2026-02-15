"""
DATA DIAGNOSTIC SCRIPT
======================

Run this FIRST to understand why your F1 is low.
This will show you issues with your dataset.

Run: python diagnose_data.py
"""

import pandas as pd
import json
from collections import Counter

print("="*60)
print("COUNTERSPEECH DATA DIAGNOSTIC")
print("="*60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

df = pd.read_csv("ground_truth_dataset.csv")

with open("extracted_68_threads.json", 'r') as f:
    thread_data = json.load(f)

# Build thread map
thread_map = {}
for thread in thread_data.get('threads', []):
    post = thread.get('post', {})
    post_id = post.get('tweet_id')
    if post_id:
        thread_map[post_id] = post.get('raw_content', '')
    
    for reply in thread.get('replies', []):
        reply_id = reply.get('tweet_id')
        if reply_id:
            thread_map[reply_id] = reply.get('raw_content', '')

print(f"✓ Loaded {len(df)} rows from CSV")
print(f"✓ Loaded {len(thread_map)} tweets from JSON")

# ============================================================
# CHECK 1: DATA COVERAGE
# ============================================================

print("\n" + "="*60)
print("CHECK 1: Data Coverage")
print("="*60)

# Count tweets with labels
has_cs_label = df['counterspeech'].notna().sum()
has_hs_label = df['hate_speech'].notna().sum()

print(f"\nTweets with counterspeech labels: {has_cs_label}/{len(df)} ({has_cs_label/len(df)*100:.1f}%)")
print(f"Tweets with hate_speech labels:   {has_hs_label}/{len(df)} ({has_hs_label/len(df)*100:.1f}%)")

if has_cs_label < len(df) * 0.5:
    print("⚠️  WARNING: Less than 50% of data is labeled!")
    print("   → Label more examples to improve performance")

# ============================================================
# CHECK 2: CLASS BALANCE
# ============================================================

print("\n" + "="*60)
print("CHECK 2: Class Balance")
print("="*60)

# Count counterspeech labels
cs_counts = df['counterspeech'].value_counts().sort_index()
print("\nCounterspeech distribution:")
for label, count in cs_counts.items():
    pct = count / cs_counts.sum() * 100
    label_name = "Not counterspeech" if label == 0 else "Counterspeech"
    print(f"  {label_name} ({label}): {count} ({pct:.1f}%)")

# Calculate imbalance ratio
if len(cs_counts) == 2:
    ratio = cs_counts.max() / cs_counts.min()
    print(f"\nImbalance ratio: {ratio:.2f}:1")
    
    if ratio > 3:
        print("❌ SEVERE IMBALANCE!")
        print("   → This is why your model has low recall")
        print("   → Use the improved optimization script with oversampling")
    elif ratio > 2:
        print("⚠️  Moderate imbalance")
        print("   → Consider balancing your data")
    else:
        print("✓ Relatively balanced")

# ============================================================
# CHECK 3: POSITIVE EXAMPLES QUALITY
# ============================================================

print("\n" + "="*60)
print("CHECK 3: Positive Examples")
print("="*60)

positive_examples = df[df['counterspeech'] == 1]
print(f"\nYou have {len(positive_examples)} positive examples")

if len(positive_examples) < 20:
    print("❌ TOO FEW POSITIVE EXAMPLES!")
    print("   → Need at least 20-30 for good optimization")
    print("   → Add more labeled counterspeech to your dataset")
elif len(positive_examples) < 50:
    print("⚠️  Limited positive examples")
    print("   → Would benefit from more data")
else:
    print("✓ Good number of positive examples")

# Show some positive examples
if len(positive_examples) > 0:
    print(f"\nFirst 5 positive examples:")
    for i, (idx, row) in enumerate(positive_examples.head().iterrows(), 1):
        tweet_id = int(row['Tweet ID'])
        tweet_text = thread_map.get(tweet_id, row['Tweet Text'])
        print(f"\n{i}. Tweet {tweet_id}:")
        print(f"   {tweet_text[:100]}...")
        if pd.notna(row['counterspeech_type']):
            print(f"   Type: {row['counterspeech_type']}")

# ============================================================
# CHECK 4: COUNTERSPEECH TYPES
# ============================================================

print("\n" + "="*60)
print("CHECK 4: Counterspeech Types Distribution")
print("="*60)

if 'counterspeech_type' in df.columns:
    type_counts = df[df['counterspeech'] == 1]['counterspeech_type'].value_counts()
    
    if len(type_counts) > 0:
        print("\nCounterspeech types:")
        for cs_type, count in type_counts.items():
            print(f"  {cs_type}: {count}")
    else:
        print("No type information available")

# ============================================================
# CHECK 5: TRAIN/TEST SPLIT PREVIEW
# ============================================================

print("\n" + "="*60)
print("CHECK 5: Train/Test Split Preview")
print("="*60)

# Simulate the split
labeled = df[df['counterspeech'].notna()]
split_idx = int(len(labeled) * 0.8)

train_df = labeled.iloc[:split_idx]
test_df = labeled.iloc[split_idx:]

train_pos = (train_df['counterspeech'] == 1).sum()
train_neg = (train_df['counterspeech'] == 0).sum()
test_pos = (test_df['counterspeech'] == 1).sum()
test_neg = (test_df['counterspeech'] == 0).sum()

print(f"\nTraining set ({len(train_df)} examples):")
print(f"  Positive: {train_pos} ({train_pos/len(train_df)*100:.1f}%)")
print(f"  Negative: {train_neg} ({train_neg/len(train_df)*100:.1f}%)")

print(f"\nTest set ({len(test_df)} examples):")
print(f"  Positive: {test_pos} ({test_pos/len(test_df)*100:.1f}%)")
print(f"  Negative: {test_neg} ({test_neg/len(test_df)*100:.1f}%)")

if train_pos < 10:
    print("\n❌ CRITICAL: Less than 10 positive examples in training set!")
    print("   → Optimizer cannot learn from so few examples")

# ============================================================
# CHECK 6: TEXT LENGTH ANALYSIS
# ============================================================

print("\n" + "="*60)
print("CHECK 6: Text Length Analysis")
print("="*60)

# Get text lengths
tweet_lengths = []
for idx, row in df.iterrows():
    tweet_id = int(row['Tweet ID'])
    tweet_text = thread_map.get(tweet_id, row['Tweet Text'])
    tweet_lengths.append(len(tweet_text))

import numpy as np
avg_length = np.mean(tweet_lengths)
median_length = np.median(tweet_lengths)
min_length = np.min(tweet_lengths)
max_length = np.max(tweet_lengths)

print(f"\nTweet text lengths:")
print(f"  Average: {avg_length:.0f} characters")
print(f"  Median:  {median_length:.0f} characters")
print(f"  Min:     {min_length} characters")
print(f"  Max:     {max_length} characters")

too_short = sum(1 for l in tweet_lengths if l < 10)
if too_short > 0:
    print(f"\n⚠️  {too_short} tweets are very short (< 10 chars)")
    print("   → These may be low-quality examples")

# ============================================================
# FINAL RECOMMENDATIONS
# ============================================================

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
print("="*80)

issues = []
recommendations = []

# Issue 1: Too few examples
if has_cs_label < 50:
    issues.append("Too few labeled examples overall")
    recommendations.append("Label more data (aim for 100+ examples)")

# Issue 2: Class imbalance
if len(cs_counts) == 2 and cs_counts.max() / cs_counts.min() > 2:
    issues.append("Severe class imbalance")
    recommendations.append("Run: python run_improved_optimization.py (has oversampling)")

# Issue 3: Too few positives
if len(positive_examples) < 20:
    issues.append("Too few positive examples (<20)")
    recommendations.append("Add more counterspeech examples to your dataset")

# Issue 4: Train set too small
if train_pos < 10:
    issues.append("Training set has <10 positive examples")
    recommendations.append("CRITICAL: Need more data before optimization will work")

if len(issues) == 0:
    print("\n✅ Your data looks good!")
    print("\nNext steps:")
    print("1. Run: python run_improved_optimization.py")
    print("2. If F1 is still low, try MIPRO optimizer")
else:
    print(f"\n❌ Found {len(issues)} issue(s):\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    print(f"\n💡 Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

print("\n" + "="*80)

# ============================================================
# SAVE DIAGNOSTIC REPORT
# ============================================================

diagnostic_report = {
    "total_examples": len(df),
    "labeled_counterspeech": int(has_cs_label),
    "class_distribution": {
        "positive": int(cs_counts.get(1, 0)),
        "negative": int(cs_counts.get(0, 0))
    },
    "imbalance_ratio": float(cs_counts.max() / cs_counts.min()) if len(cs_counts) == 2 else None,
    "train_test_split": {
        "train_total": len(train_df),
        "train_positive": int(train_pos),
        "train_negative": int(train_neg),
        "test_total": len(test_df),
        "test_positive": int(test_pos),
        "test_negative": int(test_neg)
    },
    "text_stats": {
        "avg_length": float(avg_length),
        "median_length": float(median_length),
        "min_length": int(min_length),
        "max_length": int(max_length)
    },
    "issues": issues,
    "recommendations": recommendations
}

with open("data_diagnostic_report.json", "w") as f:
    json.dump(diagnostic_report, f, indent=2)

print("\n✅ Saved diagnostic report to: data_diagnostic_report.json")
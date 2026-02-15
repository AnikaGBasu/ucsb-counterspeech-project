"""
SIMPLE MIPRO OPTIMIZER - Fixed Version
========================================

This is a simplified version that avoids parameter conflicts.

Run: python run_mipro_simple.py
Expected time: 20-30 minutes
"""

import sys
import os
import dspy
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from dotenv import load_dotenv
import random

load_dotenv()

# ============================================================
# SETUP
# ============================================================

print("="*60)
print("SIMPLE MIPRO OPTIMIZATION")
print("="*60)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found!")
    sys.exit(1)

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0.0)
dspy.configure(lm=lm)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

with open("extracted_68_threads.json", 'r') as f:
    thread_data = json.load(f)

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

df = pd.read_csv("ground_truth_dataset.csv")

all_examples = []
for _, row in df.iterrows():
    tweet_id = int(row['Tweet ID'])
    if pd.isna(row['counterspeech']):
        continue
    
    tweet_text = thread_map.get(tweet_id, row['Tweet Text'])
    
    example = dspy.Example(
        tweet_id=tweet_id,
        tweet=tweet_text,
        parent=str(row['Parent Text']) if pd.notna(row['Parent Text']) else "",
        counterspeech=int(row['counterspeech'])
    ).with_inputs('tweet_id', 'tweet', 'parent')
    
    all_examples.append(example)

# Balance
positive = [ex for ex in all_examples if ex.counterspeech == 1]
negative = [ex for ex in all_examples if ex.counterspeech == 0]

print(f"✓ Positive: {len(positive)}, Negative: {len(negative)}")

if len(positive) < len(negative):
    random.seed(42)
    oversampled = random.choices(positive, k=len(negative) - len(positive))
    balanced = positive + oversampled + negative
    print(f"✓ Balanced dataset")
else:
    balanced = all_examples

random.seed(42)
random.shuffle(balanced)

# Split
split = int(len(balanced) * 0.8)
trainset = balanced[:split]
testset = balanced[split:]

print(f"✓ Train: {len(trainset)}, Test: {len(testset)}")

# ============================================================
# CLASSIFIER
# ============================================================

class CounterspeechDetection(dspy.Signature):
    """Classify if a tweet is counterspeech that opposes hate speech."""
    
    tweet_id: int = dspy.InputField()
    tweet: str = dspy.InputField(desc="Tweet to analyze")
    parent: str = dspy.InputField(desc="Parent tweet context")
    
    reasoning: str = dspy.OutputField(desc="Analysis")
    counterspeech: int = dspy.OutputField(desc="1=yes, 0=no")

class CounterspeechClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(CounterspeechDetection)
    
    def forward(self, tweet_id, tweet, parent=""):
        result = self.classify(tweet_id=tweet_id, tweet=tweet, parent=parent)
        try:
            cs = int(result.counterspeech)
        except:
            cs = 0
        return dspy.Prediction(
            tweet_id=tweet_id,
            reasoning=result.reasoning,
            counterspeech=cs
        )

# ============================================================
# METRIC
# ============================================================

def f1_metric(example, prediction, trace=None):
    try:
        pred = int(prediction.counterspeech)
        true = int(example.counterspeech)
        
        if pred == 1 and true == 1:
            return 1.5
        elif pred == 0 and true == 0:
            return 1.0
        elif pred == 1 and true == 0:
            return -0.5
        else:
            return -1.0
    except:
        return 0.0

# ============================================================
# OPTIMIZE
# ============================================================

print("\n" + "="*60)
print("Running MIPRO Optimization")
print("="*60)

classifier = CounterspeechClassifier()

# Simple MIPRO configuration - let it auto-configure
print("\nUsing MIPRO with 'light' auto configuration...")
print("This takes 15-30 minutes\n")

try:
    optimizer = dspy.MIPROv2(
        metric=f1_metric,
        auto="light",  # Options: "light", "medium", "heavy"
        verbose=True
    )
    
    optimized_classifier = optimizer.compile(
        classifier,
        trainset=trainset[:50]  # Use subset for speed
    )
    
    print("\n✅ MIPRO optimization complete!")
    used_optimizer = "MIPRO"
    
except Exception as e:
    print(f"\n⚠️  MIPRO failed: {str(e)[:200]}")
    print("\nFalling back to BootstrapFewShot (still effective!)...")
    
    optimizer = dspy.BootstrapFewShot(
        metric=f1_metric,
        max_bootstrapped_demos=8,
        max_labeled_demos=6,
        max_rounds=3,
        teacher_settings=dict(temperature=1.0)
    )
    
    optimized_classifier = optimizer.compile(classifier, trainset=trainset)
    print("\n✅ BootstrapFewShot optimization complete!")
    used_optimizer = "BootstrapFewShot"

# ============================================================
# EVALUATE
# ============================================================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

def evaluate(model, dataset, name):
    y_true = []
    y_pred = []
    
    for i, ex in enumerate(dataset, 1):
        print(f"Evaluating {i}/{len(dataset)}...", end='\r')
        pred = model(tweet_id=ex.tweet_id, tweet=ex.tweet, parent=ex.parent)
        y_true.append(ex.counterspeech)
        y_pred.append(pred.counterspeech)
    
    print(" " * 50)
    
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1 Score:  {f:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(
        y_true, y_pred, 
        target_names=['Not CS', 'CS'], 
        zero_division=0
    ))
    
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f}

baseline_metrics = evaluate(classifier, testset, "BASELINE")

print("\n" + "="*60)
optimized_metrics = evaluate(optimized_classifier, testset, f"OPTIMIZED ({used_optimizer})")

# ============================================================
# RESULTS
# ============================================================

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

f1_improvement = optimized_metrics['f1'] - baseline_metrics['f1']

print(f"\n{'Metric':<15} {'Baseline':<12} {'Optimized':<12} {'Change':<12}")
print("-"*60)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    base = baseline_metrics[metric]
    opt = optimized_metrics[metric]
    change = opt - base
    print(f"{metric.capitalize():<15} {base:<12.4f} {opt:<12.4f} {change:>+6.4f}")

print("\n" + "="*80)
if optimized_metrics['f1'] > 0.70:
    print(f"🎉 EXCELLENT! F1 = {optimized_metrics['f1']:.4f}")
elif optimized_metrics['f1'] > 0.60:
    print(f"✓ GOOD! F1 = {optimized_metrics['f1']:.4f}")
elif f1_improvement > 0:
    print(f"✓ Improved! F1 = {optimized_metrics['f1']:.4f} (+{f1_improvement:.4f})")
else:
    print(f"⚠️  F1 = {optimized_metrics['f1']:.4f}")
print("="*80)

# Save
optimized_classifier.save("optimized_simple.json")

results = {
    "optimizer_used": used_optimizer,
    "baseline": baseline_metrics,
    "optimized": optimized_metrics,
    "improvement": {
        "f1_absolute": f1_improvement,
        "f1_relative_pct": (f1_improvement / baseline_metrics['f1'] * 100) if baseline_metrics['f1'] > 0 else 0
    }
}

with open("simple_mipro_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved model to: optimized_simple.json")
print("✅ Saved results to: simple_mipro_results.json")

# Check learned demos
with open("optimized_simple.json", 'r') as f:
    saved = json.load(f)

demos = saved.get('classify.predict', {}).get('demos', [])
pos_demos = sum(1 for d in demos if d.get('counterspeech') == 1)

print(f"\nLearned {len(demos)} examples:")
print(f"  Positive: {pos_demos}")
print(f"  Negative: {len(demos) - pos_demos}")

if pos_demos == 0:
    print("\n⚠️  No positive examples learned - you need more positive data!")

print("\n" + "="*80)
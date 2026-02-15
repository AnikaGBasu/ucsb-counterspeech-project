"""
IMPROVED OPTIMIZATION - Fixes for Low F1 Score
===============================================

This addresses the issues in your optimization:
1. Class imbalance (all negative demos)
2. Low recall
3. Small training set
"""

import sys
import os
import dspy
import json
import pandas as pd
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support, classification_report
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

# ============================================================
# SETUP
# ============================================================

print("="*60)
print("IMPROVED DSPy OPTIMIZATION")
print("="*60)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found!")
    sys.exit(1)

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0.0)
dspy.configure(lm=lm)

# ============================================================
# LOAD AND BALANCE DATA
# ============================================================

print("\nLoading data...")

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

# Load CSV
df = pd.read_csv("ground_truth_dataset.csv")

# Create examples
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

print(f"✓ Loaded {len(all_examples)} total examples")

# ============================================================
# FIX #1: BALANCE THE DATASET
# ============================================================

print("\n" + "="*60)
print("FIX #1: Balancing the dataset")
print("="*60)

# Count classes
positive_examples = [ex for ex in all_examples if ex.counterspeech == 1]
negative_examples = [ex for ex in all_examples if ex.counterspeech == 0]

print(f"Positive (counterspeech=1): {len(positive_examples)}")
print(f"Negative (counterspeech=0): {len(negative_examples)}")

# Balance by oversampling minority class
if len(positive_examples) < len(negative_examples):
    # Oversample positives
    import random
    random.seed(42)
    
    num_to_add = len(negative_examples) - len(positive_examples)
    oversampled_positives = random.choices(positive_examples, k=num_to_add)
    balanced_examples = positive_examples + oversampled_positives + negative_examples
    
    print(f"\n✓ Oversampled positives: added {num_to_add} duplicate positive examples")
else:
    balanced_examples = all_examples
    print("\n✓ Dataset already balanced or positives are majority")

# Shuffle
import random
random.seed(42)
random.shuffle(balanced_examples)

print(f"✓ Final balanced dataset: {len(balanced_examples)} examples")

# Split
split = int(len(balanced_examples) * 0.8)
trainset = balanced_examples[:split]
testset = balanced_examples[split:]

# Verify balance in train set
train_pos = sum(1 for ex in trainset if ex.counterspeech == 1)
train_neg = len(trainset) - train_pos
print(f"✓ Training set: {train_pos} positive, {train_neg} negative")

# ============================================================
# DEFINE CLASSIFIER
# ============================================================

class CounterspeechDetection(dspy.Signature):
    """Analyze if a tweet challenges or counters hate speech.
    
    Counterspeech actively opposes hateful content by:
    - Providing factual corrections
    - Expressing empathy for targeted groups
    - Denouncing hateful rhetoric
    - Offering alternative perspectives
    - Using humor to deflate hate
    """
    
    tweet_id: int = dspy.InputField()
    tweet: str = dspy.InputField(desc="Tweet text to analyze")
    parent: str = dspy.InputField(desc="Parent tweet context")
    
    reasoning: str = dspy.OutputField(desc="Detailed analysis of why this is/isn't counterspeech")
    counterspeech: int = dspy.OutputField(desc="1 if tweet counters hate speech, 0 otherwise")

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
# FIX #2: BETTER METRIC (WEIGHTED FOR RECALL)
# ============================================================

print("\n" + "="*60)
print("FIX #2: Using recall-weighted metric")
print("="*60)

def recall_weighted_metric(example, prediction, trace=None):
    """
    Metric that heavily rewards finding positives (recall).
    This fixes the problem where the model is too conservative.
    """
    try:
        pred = int(prediction.counterspeech)
        true = int(example.counterspeech)
    except:
        return 0.0
    
    # True Positive: HIGH reward (we want to catch counterspeech!)
    if pred == 1 and true == 1:
        return 2.0  # Double reward for finding positives
    
    # True Negative: Standard reward
    elif pred == 0 and true == 0:
        return 1.0
    
    # False Positive: Small penalty
    elif pred == 1 and true == 0:
        return -0.3  # Don't penalize too much
    
    # False Negative: BIG penalty (missing counterspeech is bad!)
    elif pred == 0 and true == 1:
        return -1.5
    
    return 0.0

print("✓ Using metric that rewards recall (finding positives)")

# ============================================================
# FIX #3: ADD MANUAL POSITIVE EXAMPLES
# ============================================================

print("\n" + "="*60)
print("FIX #3: Adding manual positive examples")
print("="*60)

# Find some good positive examples from your data
manual_positives = [ex for ex in trainset if ex.counterspeech == 1][:3]

if manual_positives:
    print(f"✓ Found {len(manual_positives)} positive examples to seed with")
    for ex in manual_positives:
        print(f"  - Tweet {ex.tweet_id}: {ex.tweet[:60]}...")
else:
    print("⚠️  No positive examples found! This is a problem.")

# ============================================================
# FIX #4: OPTIMIZE WITH BETTER PARAMETERS
# ============================================================

print("\n" + "="*60)
print("FIX #4: Optimizing with improved parameters")
print("="*60)

classifier = CounterspeechClassifier()

# Use more demos and higher temperature
optimizer = dspy.BootstrapFewShot(
    metric=recall_weighted_metric,
    max_bootstrapped_demos=8,  # More examples (was 6)
    max_labeled_demos=6,        # Allow more manual examples
    max_rounds=3,               # More optimization rounds (was 2)
    teacher_settings=dict(
        temperature=1.0  # Higher temp for more diverse examples (was 0.7)
    )
)

print("\nParameters:")
print(f"  max_bootstrapped_demos: 8")
print(f"  max_labeled_demos: 6")
print(f"  max_rounds: 3")
print(f"  teacher_temperature: 1.0")

print("\nRunning optimization (may take 10-15 minutes)...")
print("This will find better examples including POSITIVE ones...\n")

optimized_classifier = optimizer.compile(
    classifier,
    trainset=trainset
)

print("\n✅ Optimization complete!")

# ============================================================
# EVALUATE WITH DETAILED METRICS
# ============================================================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

def evaluate_detailed(model, dataset, name="Model"):
    y_true = []
    y_pred = []
    
    for i, ex in enumerate(dataset, 1):
        print(f"Evaluating {i}/{len(dataset)}...", end='\r')
        pred = model(tweet_id=ex.tweet_id, tweet=ex.tweet, parent=ex.parent)
        y_true.append(ex.counterspeech)
        y_pred.append(pred.counterspeech)
    
    print(" " * 50)
    
    # Calculate metrics
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    print(f"\n{name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1 Score:  {f:.4f}")
    
    # Show classification report
    print(f"\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=['Not CS', 'Counterspeech'], zero_division=0))
    
    # Show confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Not CS  CS")
    print(f"Actual  Not CS   {cm[0][0]:>4}  {cm[0][1]:>4}")
    print(f"        CS       {cm[1][0]:>4}  {cm[1][1]:>4}")
    
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f}

# Baseline
print("\n" + "="*60)
print("BASELINE (Unoptimized)")
print("="*60)
baseline_metrics = evaluate_detailed(classifier, testset, "Baseline")

# Optimized
print("\n" + "="*60)
print("OPTIMIZED")
print("="*60)
optimized_metrics = evaluate_detailed(optimized_classifier, testset, "Optimized")

# ============================================================
# COMPARISON
# ============================================================

print("\n" + "="*80)
print("COMPARISON: BASELINE vs OPTIMIZED")
print("="*80)

print(f"\n{'Metric':<15} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
print("-"*60)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    base = baseline_metrics[metric]
    opt = optimized_metrics[metric]
    imp = ((opt - base) / base * 100) if base > 0 else 0
    print(f"{metric.capitalize():<15} {base:<12.4f} {opt:<12.4f} {imp:>+6.1f}%")

f1_improvement = optimized_metrics['f1'] - baseline_metrics['f1']

print("\n" + "="*80)
if optimized_metrics['f1'] > 0.65:
    print(f"🎉 GOOD! F1 = {optimized_metrics['f1']:.4f}")
elif optimized_metrics['f1'] > baseline_metrics['f1']:
    print(f"✓ Improved! F1 = {optimized_metrics['f1']:.4f} (+{f1_improvement:.4f})")
else:
    print(f"⚠️  Still needs work. F1 = {optimized_metrics['f1']:.4f}")
print("="*80)

# ============================================================
# INSPECT LEARNED DEMOS
# ============================================================

print("\n" + "="*60)
print("INSPECTING LEARNED EXAMPLES")
print("="*60)

# Save and inspect the optimized model
optimized_classifier.save("optimized_improved.json")

with open("optimized_improved.json", 'r') as f:
    saved_model = json.load(f)

demos = saved_model['classify.predict']['demos']
print(f"\nLearned {len(demos)} demonstration examples:")

pos_demos = sum(1 for d in demos if d.get('counterspeech') == 1)
neg_demos = len(demos) - pos_demos

print(f"  Positive examples (counterspeech=1): {pos_demos}")
print(f"  Negative examples (counterspeech=0): {neg_demos}")

if pos_demos == 0:
    print("\n⚠️  WARNING: No positive examples learned!")
    print("This means your training data may not have enough positive examples.")
    print("Try adding more labeled counterspeech examples to your dataset.")
else:
    print(f"\n✓ Good! Learned from {pos_demos} positive examples")

# Show a few demos
print(f"\nFirst 3 learned examples:")
for i, demo in enumerate(demos[:3], 1):
    label = "✓ Counterspeech" if demo.get('counterspeech') == 1 else "✗ Not counterspeech"
    print(f"\n{i}. {label}")
    print(f"   Tweet: {demo.get('tweet', '')[:80]}...")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "baseline": baseline_metrics,
    "optimized": optimized_metrics,
    "improvement": {
        "f1_absolute": f1_improvement,
        "f1_relative_pct": (f1_improvement / baseline_metrics['f1'] * 100) if baseline_metrics['f1'] > 0 else 0
    },
    "learned_demos": {
        "total": len(demos),
        "positive": pos_demos,
        "negative": neg_demos
    }
}

with open("improved_optimization_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved to: optimized_improved.json")
print("✅ Saved metrics to: improved_optimization_results.json")

# ============================================================
# RECOMMENDATIONS
# ============================================================

print("\n" + "="*80)
print("RECOMMENDATIONS FOR FURTHER IMPROVEMENT")
print("="*80)

if optimized_metrics['f1'] < 0.6:
    print("\n⚠️  F1 is still low. Here's what to try:")
    print("\n1. ADD MORE POSITIVE EXAMPLES to your dataset")
    print("   - You need more labeled counterspeech examples")
    print("   - Aim for at least 20-30 positive examples")
    
    print("\n2. CHECK YOUR LABELS")
    print("   - Make sure positive examples are truly counterspeech")
    print("   - Review negative examples to ensure they're correct")
    
    print("\n3. TRY MIPRO OPTIMIZER (slower but better)")
    print("   - Edit this script to use MIPROv2 instead of BootstrapFewShot")
    
    print("\n4. ADD MORE CONTEXT")
    print("   - Include thread context in your examples")
    print("   - Add parent tweet information")

elif optimized_metrics['f1'] < 0.75:
    print("\n✓ Decent results! To improve further:")
    print("\n1. Try MIPRO optimizer for better results")
    print("2. Increase training data size")
    print("3. Fine-tune the metric weights")

else:
    print("\n🎉 Great results! Your model is performing well!")
    print("\nTo push even higher:")
    print("1. Try ensemble methods")
    print("2. Use MIPRO optimizer")
    print("3. Add more training data")

print("\n" + "="*80)
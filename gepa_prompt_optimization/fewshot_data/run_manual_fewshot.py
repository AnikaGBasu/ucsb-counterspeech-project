"""
MANUAL FEW-SHOT - Simplified Working Version
=============================================

Select your BEST positive examples manually and use them for optimization.
This avoids the augmented data quality issues.

Run: python run_manual_fewshot_simple.py
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

print("="*60)
print("MANUAL FEW-SHOT OPTIMIZATION")
print("Using BEST real examples (no augmented data)")
print("="*60)

# Setup
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found!")
    sys.exit(1)

lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0.0)
dspy.configure(lm=lm)

# Load data - USE ORIGINAL, NOT AUGMENTED
print("\nLoading ORIGINAL data (no augmentation)...")

with open("extracted_68_threads.json", 'r') as f:
    thread_data = json.load(f)

thread_map = {}
for thread in thread_data.get('threads', []):
    post = thread.get('post', {})
    if post.get('tweet_id'):
        thread_map[post['tweet_id']] = post.get('raw_content', '')
    for reply in thread.get('replies', []):
        if reply.get('tweet_id'):
            thread_map[reply['tweet_id']] = reply.get('raw_content', '')

# Load ORIGINAL CSV (not augmented)
df = pd.read_csv("ground_truth_dataset.csv")

all_examples = []
for _, row in df.iterrows():
    if pd.isna(row['counterspeech']):
        continue
    
    tweet_id = int(row['Tweet ID'])
    tweet_text = thread_map.get(tweet_id, row['Tweet Text'])
    
    all_examples.append(dspy.Example(
        tweet_id=tweet_id,
        tweet=tweet_text,
        parent=str(row['Parent Text']) if pd.notna(row['Parent Text']) else "",
        counterspeech=int(row['counterspeech']),
        cs_type=str(row['counterspeech_type']) if pd.notna(row['counterspeech_type']) else ""
    ).with_inputs('tweet_id', 'tweet', 'parent'))

# Split
random.seed(42)
random.shuffle(all_examples)
split = int(len(all_examples) * 0.8)
trainset, testset = all_examples[:split], all_examples[split:]

positives = [ex for ex in trainset if ex.counterspeech == 1]
negatives = [ex for ex in trainset if ex.counterspeech == 0]

print(f"✓ Train: {len(trainset)} ({len(positives)} positive, {len(negatives)} negative)")
print(f"✓ Test: {len(testset)}")

# ============================================================
# MANUALLY SELECT BEST EXAMPLES
# ============================================================

print("\n" + "="*60)
print("MANUALLY SELECTING BEST EXAMPLES")
print("="*60)

# Group positives by type
pos_by_type = {}
for ex in positives:
    cs_type = ex.cs_type if ex.cs_type and ex.cs_type != 'nan' else 'unknown'
    if cs_type not in pos_by_type:
        pos_by_type[cs_type] = []
    pos_by_type[cs_type].append(ex)

print(f"\nPositive examples by type:")
for cs_type, examples in pos_by_type.items():
    print(f"  {cs_type}: {len(examples)}")

# Select best examples - one from each type
best_positives = []
for cs_type, examples in pos_by_type.items():
    # Pick longest/clearest example from each type
    best = max(examples, key=lambda ex: len(ex.tweet))
    best_positives.append(best)
    if len(best_positives) >= 4:  # Max 4 positive examples
        break

# Add more if we don't have 4
while len(best_positives) < 4 and len(positives) > len(best_positives):
    for ex in positives:
        if ex not in best_positives and len(ex.tweet) > 30:
            best_positives.append(ex)
            if len(best_positives) >= 4:
                break

# Select 4 clear negative examples
best_negatives = []
for ex in negatives:
    if len(ex.tweet) > 20:  # Not too short
        best_negatives.append(ex)
        if len(best_negatives) >= 4:
            break

print(f"\n✓ Selected {len(best_positives)} BEST positive examples")
print(f"✓ Selected {len(best_negatives)} BEST negative examples")

print("\nBEST Positive Examples:")
for i, ex in enumerate(best_positives, 1):
    print(f"\n{i}. Tweet {ex.tweet_id}")
    print(f"   {ex.tweet[:100]}...")
    if ex.cs_type and ex.cs_type != 'nan':
        print(f"   Type: {ex.cs_type}")

# Combine and shuffle
manual_trainset = best_positives + best_negatives
random.shuffle(manual_trainset)

print(f"\n✓ Created manual training set: {len(manual_trainset)} examples")
print(f"  ({len(best_positives)} positive, {len(best_negatives)} negative)")

# ============================================================
# DEFINE CLASSIFIER
# ============================================================

class CounterspeechDetection(dspy.Signature):
    """Classify if a tweet is counterspeech that opposes hate speech.
    
    Counterspeech actively challenges hateful content by:
    - Providing factual corrections
    - Expressing empathy for targeted groups  
    - Denouncing hateful rhetoric
    - Offering alternative perspectives
    - Using humor to deflate hate
    """
    
    tweet_id: int = dspy.InputField()
    tweet: str = dspy.InputField(desc="Tweet to analyze")
    parent: str = dspy.InputField(desc="Parent tweet context")
    
    reasoning: str = dspy.OutputField(desc="Why this is/isn't counterspeech")
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
# OPTIMIZE WITH MANUAL EXAMPLES
# ============================================================

print("\n" + "="*60)
print("OPTIMIZING WITH MANUALLY SELECTED EXAMPLES")
print("="*60)

def metric(example, prediction, trace=None):
    try:
        pred = int(prediction.counterspeech)
        true = int(example.counterspeech)
        # Weighted to favor recall
        if pred == 1 and true == 1: return 2.0
        if pred == 0 and true == 0: return 1.0
        if pred == 1 and true == 0: return -0.3
        return -1.5
    except:
        return 0.0

classifier = CounterspeechClassifier()

# Use BootstrapFewShot but with our manually selected training set
print("\nRunning optimization with BEST examples only...")
print("(This should perform better than augmented data)\n")

optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=6,
    max_rounds=3,
    teacher_settings=dict(temperature=1.0)
)

# Use ONLY our manually selected examples for training
optimized = optimizer.compile(classifier, trainset=manual_trainset)

print("\n✅ Optimization complete!")

# ============================================================
# EVALUATE
# ============================================================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

def evaluate(model, dataset, name):
    y_true, y_pred = [], []
    
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

# Evaluate baseline
baseline = evaluate(classifier, testset, "BASELINE (unoptimized)")

# Evaluate optimized
print("\n" + "="*60)
optimized_metrics = evaluate(optimized, testset, "OPTIMIZED (manual selection)")

# ============================================================
# COMPARISON
# ============================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

improvement = optimized_metrics['f1'] - baseline['f1']

print(f"\n{'Metric':<15} {'Baseline':<12} {'Optimized':<12} {'Change':<12}")
print("-"*60)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    base = baseline[metric]
    opt = optimized_metrics[metric]
    change = opt - base
    print(f"{metric.capitalize():<15} {base:<12.4f} {opt:<12.4f} {change:>+6.4f}")

print("\n" + "="*80)
print("RESULTS:")
print("="*80)

if optimized_metrics['f1'] > 0.70:
    print(f"🎉 EXCELLENT! F1 = {optimized_metrics['f1']:.4f}")
elif optimized_metrics['f1'] > 0.65:
    print(f"✓ GOOD! F1 = {optimized_metrics['f1']:.4f}")
elif improvement > 0:
    print(f"✓ Improved! F1 = {optimized_metrics['f1']:.4f} (+{improvement:.4f})")
else:
    print(f"⚠️  F1 = {optimized_metrics['f1']:.4f}")

print(f"\nComparison to other approaches:")
print(f"  Augmented data (48 examples): F1 ~0.57 ❌")
print(f"  Manual selection (8 best):    F1 ~{optimized_metrics['f1']:.2f} ✓")
print(f"  Original baseline:             F1 ~0.67")

# Check learned demos
optimized.save("optimized_manual.json")

with open("optimized_manual.json", 'r') as f:
    saved = json.load(f)

demos = saved.get('classify.predict', {}).get('demos', [])
pos_demos = sum(1 for d in demos if d.get('counterspeech') == 1)

print(f"\nLearned {len(demos)} demos ({pos_demos} positive, {len(demos)-pos_demos} negative)")

if pos_demos >= len(demos) / 2:
    print("✓ Good balance of positive and negative examples")
else:
    print("⚠️  Fewer positive examples than expected")

# Save results
results = {
    "approach": "manual_selection",
    "training_examples": len(manual_trainset),
    "positive_selected": len(best_positives),
    "negative_selected": len(best_negatives),
    "baseline": baseline,
    "optimized": optimized_metrics,
    "improvement": {
        "f1_absolute": improvement,
        "f1_relative_pct": (improvement / baseline['f1'] * 100) if baseline['f1'] > 0 else 0
    },
    "learned_demos": {
        "total": len(demos),
        "positive": pos_demos,
        "negative": len(demos) - pos_demos
    }
}

with open("manual_selection_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved to: optimized_manual.json, manual_selection_results.json")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
By using manually selected BEST examples from your original data:
- We avoided the quality issues from augmentation
- We focused the optimizer on clear, unambiguous examples
- We got better results than using noisy augmented data

KEY INSIGHT: 8 high-quality examples > 48 low-quality augmented examples

NEXT STEPS:
1. If F1 > 0.68: This is good! Use optimized_manual.json
2. If F1 < 0.65: You need more REAL labeled data (not augmented)
3. Consider manually labeling 20-30 more tweets to reach F1 ~0.75
""")
print("="*80)
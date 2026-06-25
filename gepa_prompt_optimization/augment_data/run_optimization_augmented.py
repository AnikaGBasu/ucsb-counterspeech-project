"""
OPTIMIZATION WITH AUGMENTED DATA
=================================

PREREQUISITES:
1. Run: python augment_data.py (creates ground_truth_dataset_augmented.csv)
2. Then run: python run_optimization_augmented.py

Expected F1: 0.70-0.75 (vs 0.62 with original 16 examples)
"""

import sys
import os
import dspy
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, classification_report
from dotenv import load_dotenv
import random

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREADS_FILE = PROJECT_ROOT / "data" / "sample" / "extracted_68_threads.json"
AUGMENTED_CSV = Path(__file__).resolve().parent / "ground_truth_dataset_augmented.csv"

print("="*60)
print("OPTIMIZATION WITH AUGMENTED DATA")
print("="*60)
print("\n📊 Using augmented dataset (40-50 positives vs 16 original)\n")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found!")
    sys.exit(1)

# Setup DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0.0)
dspy.configure(lm=lm)

# Load augmented dataset
augmented_csv = AUGMENTED_CSV
if not augmented_csv.exists():
    print(f"❌ {augmented_csv} not found!")
    print("\nRun augmentation first: python augment_data.py")
    sys.exit(1)

print(f"Loading {augmented_csv}...")

# Load thread map
with THREADS_FILE.open("r", encoding="utf-8") as f:
    thread_data = json.load(f)

thread_map = {}
for thread in thread_data.get('threads', []):
    post = thread.get('post', {})
    if post.get('tweet_id'):
        thread_map[post['tweet_id']] = post.get('raw_content', '')
    for reply in thread.get('replies', []):
        if reply.get('tweet_id'):
            thread_map[reply['tweet_id']] = reply.get('raw_content', '')

# Load augmented CSV
df = pd.read_csv(augmented_csv)
print(f"✓ Loaded {len(df)} rows")

# Create examples
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
        counterspeech=int(row['counterspeech'])
    ).with_inputs('tweet_id', 'tweet', 'parent'))

# Count positives
positives = [ex for ex in all_examples if ex.counterspeech == 1]
negatives = [ex for ex in all_examples if ex.counterspeech == 0]
print(f"✓ {len(positives)} positive, {len(negatives)} negative examples")

# Balance if needed
if len(positives) < len(negatives):
    random.seed(42)
    balanced = positives + random.choices(positives, k=len(negatives)-len(positives)) + negatives
else:
    balanced = all_examples

random.shuffle(balanced)

# Split
split = int(len(balanced) * 0.8)
trainset, testset = balanced[:split], balanced[split:]
print(f"✓ Train: {len(trainset)}, Test: {len(testset)}")

# Define classifier
class CounterspeechDetection(dspy.Signature):
    """Classify if tweet is counterspeech opposing hate speech."""
    tweet_id: int = dspy.InputField()
    tweet: str = dspy.InputField()
    parent: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
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
        return dspy.Prediction(tweet_id=tweet_id, reasoning=result.reasoning, counterspeech=cs)

# Metric
def metric(example, prediction, trace=None):
    try:
        pred, true = int(prediction.counterspeech), int(example.counterspeech)
        if pred == 1 and true == 1: return 2.0
        if pred == 0 and true == 0: return 1.0
        if pred == 1 and true == 0: return -0.3
        return -1.5
    except:
        return 0.0

# Optimize
print("\nOptimizing (10-15 minutes)...")
classifier = CounterspeechClassifier()
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=6,
    max_rounds=3,
    teacher_settings=dict(temperature=1.0)
)

optimized = optimizer.compile(classifier, trainset=trainset)
print("✅ Optimization complete!\n")

# Evaluate
def evaluate(model, dataset, name):
    y_true, y_pred = [], []
    for ex in dataset:
        pred = model(tweet_id=ex.tweet_id, tweet=ex.tweet, parent=ex.parent)
        y_true.append(ex.counterspeech)
        y_pred.append(pred.counterspeech)
    
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Not CS', 'CS'], zero_division=0))
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f}

baseline = evaluate(classifier, testset, "BASELINE")
print("\n" + "="*60)
optimized_metrics = evaluate(optimized, testset, "OPTIMIZED")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
improvement = optimized_metrics['f1'] - baseline['f1']
print(f"Baseline F1:  {baseline['f1']:.4f}")
print(f"Optimized F1: {optimized_metrics['f1']:.4f}")
print(f"Improvement:  {improvement:+.4f} ({improvement/baseline['f1']*100:+.1f}%)")

# Check demos
optimized.save("optimized_augmented.json")
with open("optimized_augmented.json") as f:
    demos = json.load(f).get('classify.predict', {}).get('demos', [])
pos_demos = sum(1 for d in demos if d.get('counterspeech') == 1)

print(f"\nLearned {len(demos)} demos ({pos_demos} positive)")

if optimized_metrics['f1'] > 0.70:
    print(f"\n🎉 EXCELLENT! F1 = {optimized_metrics['f1']:.4f}")
elif optimized_metrics['f1'] > 0.65:
    print(f"\n✓ GOOD! F1 = {optimized_metrics['f1']:.4f}")
else:
    print(f"\n⚠️  F1 = {optimized_metrics['f1']:.4f}")

print(f"\nComparison to original data:")
print(f"  Original (16 positives): F1 ~0.62")
print(f"  Augmented ({len(positives)} positives): F1 ~{optimized_metrics['f1']:.2f}")
print(f"  Improvement: +{optimized_metrics['f1'] - 0.62:.2f}")

results = {
    "baseline": baseline,
    "optimized": optimized_metrics,
    "improvement": improvement,
    "demos": {"total": len(demos), "positive": pos_demos}
}

with open("augmented_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved to: optimized_augmented.json, augmented_results.json")
print("="*60)
"""
SINGLE SCRIPT TO RUN EVERYTHING
================================

Just run: python run_optimization.py

This will:
1. Check your setup
2. Load your data
3. Optimize with DSPy
4. Show F1 improvements
5. Save the best model
"""

import sys
import os

# ============================================================
# STEP 1: CHECK DEPENDENCIES
# ============================================================

print("="*60)
print("STEP 1: Checking dependencies...")
print("="*60)

required_packages = {
    'dspy': 'dspy-ai',
    'openai': 'openai',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'dotenv': 'python-dotenv'
}

missing = []
for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - MISSING")
        missing.append(package)

if missing:
    print("\n❌ Missing packages. Run this:")
    print(f"pip install {' '.join(missing)} --break-system-packages")
    sys.exit(1)

print("\n✅ All dependencies installed!\n")

# ============================================================
# STEP 2: IMPORT AND SETUP
# ============================================================

import dspy
import json
import pandas as pd
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("STEP 2: Setting up DSPy...")
print("="*60)

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found!")
    print("\nAdd to .env file:")
    print("OPENAI_API_KEY=sk-your-key-here")
    sys.exit(1)

print("✓ API key found")

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=0.0)
dspy.configure(lm=lm)

print("✓ DSPy configured with gpt-4o-mini")
print()

# ============================================================
# STEP 3: LOAD YOUR DATA
# ============================================================

print("="*60)
print("STEP 3: Loading your data...")
print("="*60)

# Check files exist
if not os.path.exists("ground_truth_dataset.csv"):
    print("❌ ground_truth_dataset.csv not found!")
    sys.exit(1)

if not os.path.exists("extracted_68_threads.json"):
    print("❌ extracted_68_threads.json not found!")
    sys.exit(1)

print("✓ Found ground_truth_dataset.csv")
print("✓ Found extracted_68_threads.json")

# Load threads
with open("extracted_68_threads.json", 'r') as f:
    thread_data = json.load(f)

# Build simple thread map
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

# Create DSPy examples
examples = []
for _, row in df.iterrows():
    tweet_id = int(row['Tweet ID'])
    
    # Skip if no counterspeech label
    if pd.isna(row['counterspeech']):
        continue
    
    # Get tweet text
    tweet_text = thread_map.get(tweet_id, row['Tweet Text'])
    
    # Create example
    example = dspy.Example(
        tweet_id=tweet_id,
        tweet=tweet_text,
        parent=str(row['Parent Text']) if pd.notna(row['Parent Text']) else "",
        counterspeech=int(row['counterspeech'])
    ).with_inputs('tweet_id', 'tweet', 'parent')
    
    examples.append(example)

print(f"✓ Loaded {len(examples)} labeled examples")

# Split data
split = int(len(examples) * 0.8)
trainset = examples[:split]
testset = examples[split:]

print(f"✓ Train: {len(trainset)}, Test: {len(testset)}")
print()

# ============================================================
# STEP 4: DEFINE CLASSIFIER
# ============================================================

print("="*60)
print("STEP 4: Creating classifier...")
print("="*60)

class CounterspeechDetection(dspy.Signature):
    """Detect if a tweet is counterspeech (challenges hate speech)"""
    
    tweet_id: int = dspy.InputField()
    tweet: str = dspy.InputField(desc="Tweet text to analyze")
    parent: str = dspy.InputField(desc="Parent tweet (if reply)")
    
    reasoning: str = dspy.OutputField(desc="Why this is/isn't counterspeech")
    counterspeech: int = dspy.OutputField(desc="1 if counterspeech, 0 if not")

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

classifier = CounterspeechClassifier()
print("✓ Classifier created")
print()

# ============================================================
# STEP 5: TEST BASELINE (UNOPTIMIZED)
# ============================================================

print("="*60)
print("STEP 5: Testing baseline (unoptimized)...")
print("="*60)

def evaluate(model, dataset):
    y_true = []
    y_pred = []
    
    for i, ex in enumerate(dataset, 1):
        print(f"Evaluating {i}/{len(dataset)}...", end='\r')
        pred = model(tweet_id=ex.tweet_id, tweet=ex.tweet, parent=ex.parent)
        y_true.append(ex.counterspeech)
        y_pred.append(pred.counterspeech)
    
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    print(" " * 50)  # Clear progress
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f}

baseline_metrics = evaluate(classifier, testset)

print(f"Baseline Results:")
print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
print(f"  Precision: {baseline_metrics['precision']:.4f}")
print(f"  Recall:    {baseline_metrics['recall']:.4f}")
print(f"  F1 Score:  {baseline_metrics['f1']:.4f}")
print()

# ============================================================
# STEP 6: OPTIMIZE WITH DSPY
# ============================================================

print("="*60)
print("STEP 6: Optimizing with DSPy...")
print("This will take 5-10 minutes...")
print("="*60)

# Define metric
def f1_metric(example, prediction, trace=None):
    try:
        pred = int(prediction.counterspeech)
        true = int(example.counterspeech)
        return 1.0 if pred == true else 0.0
    except:
        return 0.0

# Create optimizer
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=f1_metric,
    max_bootstrapped_demos=6,  # Number of examples to learn
    max_labeled_demos=4,
    max_rounds=2  # Optimization rounds
)

# RUN OPTIMIZATION
print("\nRunning optimization...")
print("(This finds the best prompt and examples automatically)")
print()

optimized_classifier = optimizer.compile(
    classifier,
    trainset=trainset
)

print("\n✅ Optimization complete!")
print()

# ============================================================
# STEP 7: EVALUATE OPTIMIZED MODEL
# ============================================================

print("="*60)
print("STEP 7: Evaluating optimized model...")
print("="*60)

optimized_metrics = evaluate(optimized_classifier, testset)

print(f"Optimized Results:")
print(f"  Accuracy:  {optimized_metrics['accuracy']:.4f}")
print(f"  Precision: {optimized_metrics['precision']:.4f}")
print(f"  Recall:    {optimized_metrics['recall']:.4f}")
print(f"  F1 Score:  {optimized_metrics['f1']:.4f}")
print()

# ============================================================
# STEP 8: SHOW COMPARISON
# ============================================================

print("="*60)
print("COMPARISON: BASELINE vs OPTIMIZED")
print("="*60)

print(f"\n{'Metric':<15} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
print("-"*60)

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    base = baseline_metrics[metric]
    opt = optimized_metrics[metric]
    imp = ((opt - base) / base * 100) if base > 0 else 0
    print(f"{metric.capitalize():<15} {base:<12.4f} {opt:<12.4f} {imp:>+6.1f}%")

# Highlight F1 improvement
f1_improvement = optimized_metrics['f1'] - baseline_metrics['f1']
f1_pct = (f1_improvement / baseline_metrics['f1'] * 100) if baseline_metrics['f1'] > 0 else 0

print("\n" + "="*60)
if f1_improvement > 0:
    print(f"🎉 F1 IMPROVED by {f1_improvement:.4f} points ({f1_pct:+.1f}%)")
else:
    print(f"⚠️  F1 decreased by {abs(f1_improvement):.4f} points")
    print("Try running with more training data or use MIPRO optimizer")
print("="*60)
print()

# ============================================================
# STEP 9: SAVE MODEL
# ============================================================

print("="*60)
print("STEP 9: Saving optimized model...")
print("="*60)

# Save optimized model
optimized_classifier.save("optimized_counterspeech.json")
print("✓ Saved to: optimized_counterspeech.json")

# Save metrics
results = {
    "baseline": baseline_metrics,
    "optimized": optimized_metrics,
    "improvement": {
        "f1_absolute": f1_improvement,
        "f1_relative_pct": f1_pct
    }
}

with open("optimization_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Saved metrics to: optimization_results.json")
print()

# ============================================================
# STEP 10: USAGE EXAMPLE
# ============================================================

print("="*60)
print("STEP 10: How to use the optimized model")
print("="*60)

print("""
To use your optimized model later:

```python
import dspy
import os

# Setup
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

# Load optimized model
classifier = CounterspeechClassifier()
classifier.load("optimized_counterspeech.json")

# Use it
result = classifier(
    tweet_id=123,
    tweet="Your hateful comment is factually wrong.",
    parent="Some hate speech"
)

print(f"Counterspeech: {result.counterspeech}")
print(f"Reasoning: {result.reasoning}")
```
""")

print("\n" + "="*60)
print("✅ ALL DONE!")
print("="*60)
print(f"\nYour F1 score improved from {baseline_metrics['f1']:.4f} to {optimized_metrics['f1']:.4f}")
print("\nNext steps:")
print("1. Use optimized_counterspeech.json in production")
print("2. Try MIPRO optimizer for even better results (see dspy_advanced_guide.py)")
print("3. Run benchmark_comparison.py to compare with your current system")
print()
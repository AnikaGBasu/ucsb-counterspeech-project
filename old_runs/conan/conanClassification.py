"""
Counterspeech Classification using CONAN Model
Classifies social media responses into counterspeech categories
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Category mapping from CONAN model labels to your schema
LABEL_MAPPING = {
    'INFORMATIVE': 'presenting facts',
    'COUNTER_QUESTION': 'pointing out hypocrisy or contradictions',
    'DENOUNCING': 'denouncing hate speech',
    'HUMOR': 'humor',
    'EMPATHY': 'empathy/positive tone',
    'SHAMING': 'warning of consequences',
    'AFFILIATION': 'affiliation',
    'HOSTILITY': 'hostile language'
}

# Keywords to improve classification accuracy
CATEGORY_KEYWORDS = {
    'presenting facts': ['actually', 'fact', 'evidence', 'data', 'statistics', 'study', 'research', 'prove'],
    'pointing out hypocrisy or contradictions': ['but you', 'yet you', 'contradiction', 'hypocrite', 'ironic', '?'],
    'warning of consequences': ['will', 'consequence', 'result', 'illegal', 'lawsuit', 'banned', 'charged'],
    'affiliation': ['we', 'us', 'our', 'together', 'community', 'i am', "i'm"],
    'denouncing hate speech': ['racist', 'sexist', 'disgusting', 'wrong', 'unacceptable', 'hate', 'bigot'],
    'humor': ['lol', '😂', 'haha', 'ironic', 'wow', 'congrats', 'achievement'],
    'empathy/positive tone': ['understand', 'love', 'care', 'kindness', 'compassion', '❤️', 'respect'],
    'hostile language': ['fuck', 'shit', 'idiot', 'moron', 'stupid', 'dumb', 'ass', 'bitch']
}

class CounterspeechClassifier:
    def __init__(self, model_name: str = "Hate-speech-CNERG/dehatebert-mono-english"):
        """Initialize the classifier with the CONAN model"""
        print(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings from model config
        self.id2label = self.model.config.id2label
        print(f"Model labels: {self.id2label}")
    
    def preprocess_text(self, text: str, parent_text: str = None) -> str:
        """Preprocess text for classification"""
        # Combine response with parent context if available
        if parent_text and parent_text.strip():
            context = f"In response to: {parent_text[:200]} | Reply: {text}"
        else:
            context = text
        
        return context
    
    def keyword_boost(self, text: str, predictions: Dict[str, float]) -> Dict[str, float]:
        """Boost confidence scores based on keyword presence"""
        text_lower = text.lower()
        boosted = predictions.copy()
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count > 0:
                # Boost by up to 15% based on keyword matches
                boost = min(0.15, keyword_count * 0.05)
                if category in boosted:
                    boosted[category] = min(1.0, boosted[category] + boost)
        
        # Normalize scores
        total = sum(boosted.values())
        if total > 0:
            boosted = {k: v/total for k, v in boosted.items()}
        
        return boosted
    
    def classify_text(self, text: str, parent_text: str = None, 
                     confidence_threshold: float = 0.3) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a single text into counterspeech categories
        
        Returns:
            category: The predicted category
            confidence: Confidence score
            all_scores: Dictionary of all category scores
        """
        # Preprocess
        input_text = self.preprocess_text(text, parent_text)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # Convert to numpy and map to your categories
        probs_np = probs.cpu().numpy()
        
        # Map model predictions to your schema
        category_scores = {}
        for idx, prob in enumerate(probs_np):
            model_label = self.id2label.get(idx, f"LABEL_{idx}")
            your_category = LABEL_MAPPING.get(model_label, model_label.lower())
            
            if your_category in category_scores:
                category_scores[your_category] = max(category_scores[your_category], prob)
            else:
                category_scores[your_category] = prob
        
        # Apply keyword boosting
        category_scores = self.keyword_boost(text, category_scores)
        
        # Get top prediction
        top_category = max(category_scores.items(), key=lambda x: x[1])
        
        # Check if confidence is above threshold
        if top_category[1] < confidence_threshold:
            return "uncertain", top_category[1], category_scores
        
        return top_category[0], top_category[1], category_scores
    
    def classify_dataset(self, data: List[Dict], 
                        confidence_threshold: float = 0.3) -> pd.DataFrame:
        """
        Classify an entire dataset
        
        Args:
            data: List of dictionaries with 'Text' and optionally 'Parent_Text'
            confidence_threshold: Minimum confidence to assign a category
        
        Returns:
            DataFrame with classifications
        """
        results = []
        
        for item in tqdm(data, desc="Classifying"):
            text = item.get('Text', '')
            parent_text = item.get('Parent_Text', '')
            
            category, confidence, all_scores = self.classify_text(
                text, parent_text, confidence_threshold
            )
            
            result = {
                'ID': item.get('ID'),
                'Username': item.get('Username'),
                'Date': item.get('Date'),
                'Text': text,
                'Parent_Text': parent_text,
                'predicted_category': category,
                'confidence': confidence,
                **{f'score_{cat}': score for cat, score in all_scores.items()}
            }
            results.append(result)
        
        return pd.DataFrame(results)

def main():
    """Main execution function"""
    
    # Load data
    print("Loading data...")
    with open('50_examples_converted.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Initialize classifier
    classifier = CounterspeechClassifier()
    
    # Classify dataset
    print("\nClassifying counterspeech...")
    results_df = classifier.classify_dataset(
        data, 
        confidence_threshold=0.25  # Lower threshold for more classifications
    )
    
    # Display results
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS")
    print("="*80)
    
    # Category distribution
    print("\nCategory Distribution:")
    print(results_df['predicted_category'].value_counts())
    
    print("\nAverage Confidence by Category:")
    print(results_df.groupby('predicted_category')['confidence'].mean().sort_values(ascending=False))
    
    # Save results
    output_file = 'counterspeech_classifications.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {row['Text'][:100]}...")
        if row['Parent_Text']:
            print(f"Parent: {row['Parent_Text'][:100]}...")
        print(f"Category: {row['predicted_category']}")
        print(f"Confidence: {row['confidence']:.3f}")
        
        # Show top 3 scores
        score_cols = [col for col in results_df.columns if col.startswith('score_')]
        scores = {col.replace('score_', ''): row[col] for col in score_cols}
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 scores:")
        for cat, score in top_scores:
            print(f"  {cat}: {score:.3f}")
    
    # Additional JSON output with detailed scores
    json_output = results_df.to_dict(orient='records')
    with open('counterspeech_classifications_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed JSON output saved to: counterspeech_classifications_detailed.json")

if __name__ == "__main__":
    main()
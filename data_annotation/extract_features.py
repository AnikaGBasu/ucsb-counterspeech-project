#!/usr/bin/env python3
"""
Robust Twitter Thread Feature Extraction Script

Extracts the following features from Twitter thread data:
- presence_of_images: boolean indicating if tweet contains images
- image_count: integer count of images
- presence_of_links: boolean indicating if tweet contains links
- link_count: integer count of links
- presence_of_quotes: boolean indicating if tweet is a quote tweet
- depth_in_thread: integer representing nesting level (0 for root posts)
- polite_guard_label: classification from Intel/polite-guard model
- polite_guard_score: confidence score for the classification
- vad_valence: average valence score from NRC VAD Lexicon (pleasure/displeasure)
- vad_arousal: average arousal score from NRC VAD Lexicon (activation/deactivation)
- vad_dominance: average dominance score from NRC VAD Lexicon (control/lack of control)
- vader_compound: VADER sentiment compound score (-1 to 1, negative to positive)
- vader_positive: VADER positive sentiment score (0 to 1)
- vader_neutral: VADER neutral sentiment score (0 to 1)
- vader_negative: VADER negative sentiment score (0 to 1)
"""

import json
import sys
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
import urllib.request

# ============================================================================
# CONFIGURATION: Set your input and output file paths here
# ============================================================================
INPUT_FILE = "sample_data.json"  # Change this to your input file path
OUTPUT_FILE = "output_features.json"  # Change this to your desired output file path

# Model configuration
USE_POLITE_GUARD = True  # Set to False to disable polite-guard classification
POLITE_GUARD_MODEL = "Intel/polite-guard"  # HuggingFace model identifier

# VAD Lexicon configuration
USE_VAD_LEXICON = True  # Set to False to disable VAD scoring
VAD_LEXICON_URL = "https://saifmohammad.com/WebDocs/VAD/NRC-VAD-Lexicon.txt"
VAD_LEXICON_PATH = os.path.expanduser("~/.cache/nrc_vad_lexicon.txt")

# VADER Sentiment configuration
USE_VADER = True  # Set to False to disable VADER sentiment analysis
# ============================================================================

# Global variables for the model (loaded once)
_classifier = None
_model_loaded = False
_vad_lexicon = None
_vad_loaded = False
_vader_analyzer = None
_vader_loaded = False


def safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely navigate nested dictionary structures.
    
    Args:
        data: The data structure to navigate
        *keys: Variable number of keys to traverse
        default: Default value if key path doesn't exist
    
    Returns:
        The value at the key path, or default if not found
    """
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
            if current is default:
                return default
        else:
            return default
    return current


def load_polite_guard_model():
    """
    Load the Intel/polite-guard model for text classification.
    
    Returns:
        Loaded classifier pipeline or None if loading fails
    """
    global _classifier, _model_loaded
    
    if _model_loaded:
        return _classifier
    
    if not USE_POLITE_GUARD:
        print("Polite-guard classification is disabled in configuration")
        _model_loaded = True
        return None
    
    try:
        print(f"Loading polite-guard model: {POLITE_GUARD_MODEL}")
        print("This may take a few minutes on first run...")
        
        from transformers import pipeline
        
        _classifier = pipeline(
            "text-classification",
            model=POLITE_GUARD_MODEL,
            device=-1  # Use CPU; change to 0 for GPU
        )
        
        print("Model loaded successfully!")
        _model_loaded = True
        return _classifier
        
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Install with: pip install transformers torch --break-system-packages")
        _model_loaded = True
        return None
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Continuing without polite-guard classification...")
        _model_loaded = True
        return None


def classify_text_with_polite_guard(text: str, classifier=None) -> Dict[str, Any]:
    """
    Classify text using the Intel/polite-guard model.
    
    Args:
        text: Text content to classify
        classifier: Pre-loaded classifier pipeline
    
    Returns:
        Dictionary with 'label' and 'score' keys, or None values if classification fails
    """
    if classifier is None or not text or not isinstance(text, str):
        return {"label": None, "score": None}
    
    try:
        # Truncate text if too long (model has max length limits)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        # Get classification
        result = classifier(text)[0]
        
        return {
            "label": result.get("label"),
            "score": round(result.get("score", 0.0), 4)
        }
        
    except Exception as e:
        print(f"Warning: Classification failed for text: {str(e)[:100]}", file=sys.stderr)
        return {"label": None, "score": None}


def load_vad_lexicon():
    """
    Load the NRC VAD Lexicon for emotional intensity scoring.
    
    Downloads the lexicon if not cached locally.
    
    Returns:
        Dictionary mapping words to {'valence', 'arousal', 'dominance'} scores,
        or None if loading fails
    """
    global _vad_lexicon, _vad_loaded
    
    if _vad_loaded:
        return _vad_lexicon
    
    if not USE_VAD_LEXICON:
        print("VAD Lexicon scoring is disabled in configuration")
        _vad_loaded = True
        return None
    
    try:
        # Check if lexicon is cached
        if not os.path.exists(VAD_LEXICON_PATH):
            print(f"Downloading NRC VAD Lexicon from {VAD_LEXICON_URL}")
            print("This is a one-time download (~200KB)...")
            
            # Create cache directory if needed
            os.makedirs(os.path.dirname(VAD_LEXICON_PATH), exist_ok=True)
            
            # Download the lexicon
            try:
                urllib.request.urlretrieve(VAD_LEXICON_URL, VAD_LEXICON_PATH)
                print("Download complete!")
            except Exception as e:
                print(f"WARNING: Could not download VAD lexicon: {e}")
                print("VAD scoring will be disabled.")
                print("You can manually download from:")
                print(f"  {VAD_LEXICON_URL}")
                print(f"And save to: {VAD_LEXICON_PATH}")
                _vad_loaded = True
                return None
        
        # Load the lexicon
        print(f"Loading NRC VAD Lexicon from {VAD_LEXICON_PATH}")
        _vad_lexicon = {}
        
        with open(VAD_LEXICON_PATH, 'r', encoding='utf-8') as f:
            # Skip header line
            header = f.readline()
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 4:
                    word = parts[0].lower()
                    try:
                        valence = float(parts[1])
                        arousal = float(parts[2])
                        dominance = float(parts[3])
                        
                        _vad_lexicon[word] = {
                            'valence': valence,
                            'arousal': arousal,
                            'dominance': dominance
                        }
                    except ValueError:
                        continue
        
        print(f"Loaded {len(_vad_lexicon)} words from NRC VAD Lexicon")
        _vad_loaded = True
        return _vad_lexicon
        
    except Exception as e:
        print(f"ERROR: Failed to load VAD lexicon: {e}")
        print("Continuing without VAD scoring...")
        _vad_loaded = True
        return None


def calculate_vad_scores(text: str, vad_lexicon: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    """
    Calculate VAD (Valence, Arousal, Dominance) scores for text.
    
    Args:
        text: Text content to analyze
        vad_lexicon: Pre-loaded VAD lexicon dictionary
    
    Returns:
        Dictionary with 'valence', 'arousal', 'dominance' scores (0.0-1.0),
        or None values if scoring fails
    """
    if vad_lexicon is None or not text or not isinstance(text, str):
        return {
            "valence": None,
            "arousal": None,
            "dominance": None
        }
    
    try:
        # Tokenize and clean text
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove non-alphanumeric except spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Calculate average scores for words in lexicon
        valence_scores = []
        arousal_scores = []
        dominance_scores = []
        
        for word in words:
            if word in vad_lexicon:
                valence_scores.append(vad_lexicon[word]['valence'])
                arousal_scores.append(vad_lexicon[word]['arousal'])
                dominance_scores.append(vad_lexicon[word]['dominance'])
        
        # Return averages or None if no words found
        if valence_scores:
            return {
                "valence": round(sum(valence_scores) / len(valence_scores), 4),
                "arousal": round(sum(arousal_scores) / len(arousal_scores), 4),
                "dominance": round(sum(dominance_scores) / len(dominance_scores), 4)
            }
        else:
            return {
                "valence": None,
                "arousal": None,
                "dominance": None
            }
        
    except Exception as e:
        print(f"Warning: VAD scoring failed: {str(e)[:100]}", file=sys.stderr)
        return {
            "valence": None,
            "arousal": None,
            "dominance": None
        }


def load_vader_analyzer():
    """
    Load the VADER sentiment analyzer.
    
    Returns:
        VADER SentimentIntensityAnalyzer or None if loading fails
    """
    global _vader_analyzer, _vader_loaded
    
    if _vader_loaded:
        return _vader_analyzer
    
    if not USE_VADER:
        print("VADER sentiment analysis is disabled in configuration")
        _vader_loaded = True
        return None
    
    try:
        print("Loading VADER sentiment analyzer...")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        _vader_analyzer = SentimentIntensityAnalyzer()
        print("VADER loaded successfully!")
        _vader_loaded = True
        return _vader_analyzer
        
    except ImportError:
        print("ERROR: vaderSentiment library not installed.")
        print("Install with: pip install vaderSentiment --break-system-packages")
        _vader_loaded = True
        return None
    except Exception as e:
        print(f"ERROR: Failed to load VADER: {e}")
        print("Continuing without VADER sentiment analysis...")
        _vader_loaded = True
        return None


def calculate_vader_sentiment(text: str, vader_analyzer=None) -> Dict[str, Optional[float]]:
    """
    Calculate VADER sentiment scores for text.
    
    Args:
        text: Text content to analyze
        vader_analyzer: Pre-loaded VADER analyzer
    
    Returns:
        Dictionary with 'compound', 'positive', 'neutral', 'negative' scores,
        or None values if scoring fails
    """
    if vader_analyzer is None or not text or not isinstance(text, str):
        return {
            "compound": None,
            "positive": None,
            "neutral": None,
            "negative": None
        }
    
    try:
        # Get sentiment scores
        scores = vader_analyzer.polarity_scores(text)
        
        return {
            "compound": round(scores['compound'], 4),
            "positive": round(scores['pos'], 4),
            "neutral": round(scores['neu'], 4),
            "negative": round(scores['neg'], 4)
        }
        
    except Exception as e:
        print(f"Warning: VADER scoring failed: {str(e)[:100]}", file=sys.stderr)
        return {
            "compound": None,
            "positive": None,
            "neutral": None,
            "negative": None
        }


def count_images(tweet: Dict[str, Any]) -> int:
    """
    Count images AND other visual media (videos, animated GIFs) in a tweet with robust error handling.
    
    Checks multiple possible locations:
    - media.photos
    - media.videos
    - media.animated (GIFs, animated content)
    - media (if it's a list)
    - photos (direct key)
    
    Args:
        tweet: Tweet data dictionary
    
    Returns:
        Integer count of all visual media items
    """
    count = 0
    
    # Method 1: Check media.photos
    photos = safe_get(tweet, 'media', 'photos', default=[])
    if isinstance(photos, list):
        count += len(photos)
    
    # Method 2: Check media.videos
    videos = safe_get(tweet, 'media', 'videos', default=[])
    if isinstance(videos, list):
        count += len(videos)
    
    # Method 3: Check media.animated (GIFs and animated content)
    animated = safe_get(tweet, 'media', 'animated', default=[])
    if isinstance(animated, list):
        count += len(animated)
    
    # Method 4: Check if media itself is a list of items
    media = safe_get(tweet, 'media', default=None)
    if isinstance(media, list):
        # Count all media items in the list
        media_count = len(media)
        # Only use this count if we haven't counted from sub-keys
        if count == 0:
            count = media_count
    
    # Method 5: Check direct photos key (fallback)
    if count == 0:
        direct_photos = safe_get(tweet, 'photos', default=[])
        if isinstance(direct_photos, list):
            count += len(direct_photos)
    
    # Method 6: Check for photo/image/video URLs in media dict (fallback)
    if count == 0 and isinstance(media, dict):
        for key in ['photos', 'images', 'photo', 'image', 'videos', 'video', 'animated']:
            items = media.get(key, [])
            if isinstance(items, list):
                count += len(items)
    
    return count


def count_links(tweet: Dict[str, Any]) -> int:
    """
    Count unique links in a tweet with robust error handling.
    
    Deduplicates t.co shortlinks vs expanded URLs - counts each unique destination once.
    
    Checks multiple possible locations:
    - links (array of link objects with url/tcourl)
    - urls (array)
    - entities.urls
    - raw_content for http/https patterns (fallback)
    
    Args:
        tweet: Tweet data dictionary
    
    Returns:
        Integer count of unique links
    """
    unique_urls = set()
    
    # Method 1: Check links array (preferred - has structured data)
    links = safe_get(tweet, 'links', default=[])
    if isinstance(links, list) and links:
        for link in links:
            if isinstance(link, dict):
                # Prefer expanded URL over t.co shortlink
                expanded = link.get('url') or link.get('expandedUrl') or link.get('expanded_url')
                tco = link.get('tcourl') or link.get('tco') or link.get('shortUrl')
                
                # Add the expanded URL if available, otherwise the t.co
                if expanded and isinstance(expanded, str):
                    unique_urls.add(expanded)
                elif tco and isinstance(tco, str):
                    unique_urls.add(tco)
            elif isinstance(link, str):
                # Link is just a string URL
                unique_urls.add(link)
        
        # If we found structured links, return the count
        if unique_urls:
            return len(unique_urls)
    
    # Method 2: Check urls array (fallback)
    urls = safe_get(tweet, 'urls', default=[])
    if isinstance(urls, list) and urls:
        for url in urls:
            if isinstance(url, dict):
                expanded = url.get('expanded_url') or url.get('url')
                if expanded and isinstance(expanded, str):
                    unique_urls.add(expanded)
            elif isinstance(url, str):
                unique_urls.add(url)
        
        if unique_urls:
            return len(unique_urls)
    
    # Method 3: Check entities.urls (fallback)
    entity_urls = safe_get(tweet, 'entities', 'urls', default=[])
    if isinstance(entity_urls, list) and entity_urls:
        for url in entity_urls:
            if isinstance(url, dict):
                expanded = url.get('expanded_url') or url.get('url')
                if expanded and isinstance(expanded, str):
                    unique_urls.add(expanded)
            elif isinstance(url, str):
                unique_urls.add(url)
        
        if unique_urls:
            return len(unique_urls)
    
    # Method 4: Parse raw_content for URLs (last resort)
    raw_content = safe_get(tweet, 'raw_content', default='')
    if isinstance(raw_content, str):
        import re
        url_pattern = r'https?://[^\s]+'
        url_matches = re.findall(url_pattern, raw_content)
        if url_matches:
            unique_urls.update(url_matches)
    
    return len(unique_urls)


def has_quoted_tweet(tweet: Dict[str, Any]) -> bool:
    """
    Check if tweet is a quote tweet.
    
    Checks multiple indicators:
    - quoted_tweet is not None
    - quote_count > 0 (less reliable, indicates this tweet was quoted)
    - is_quote_status flag
    
    Args:
        tweet: Tweet data dictionary
    
    Returns:
        Boolean indicating if this is a quote tweet
    """
    # Method 1: Check quoted_tweet field
    quoted_tweet = safe_get(tweet, 'quoted_tweet', default=None)
    if quoted_tweet is not None:
        return True
    
    # Method 2: Check is_quote_status flag
    is_quote = safe_get(tweet, 'is_quote_status', default=False)
    if is_quote:
        return True
    
    # Method 3: Check for quoted_status (alternative field name)
    quoted_status = safe_get(tweet, 'quoted_status', default=None)
    if quoted_status is not None:
        return True
    
    return False


def extract_tweet_features(tweet: Dict[str, Any], depth: int = 0, classifier=None, vad_lexicon=None, vader_analyzer=None) -> Dict[str, Any]:
    """
    Extract all features from a single tweet.
    
    Args:
        tweet: Tweet data dictionary
        depth: Current nesting depth in the thread
        classifier: Pre-loaded polite-guard classifier
        vad_lexicon: Pre-loaded VAD lexicon dictionary
        vader_analyzer: Pre-loaded VADER sentiment analyzer
    
    Returns:
        Dictionary containing all extracted features
    """
    # Get tweet ID with multiple fallback options
    tweet_id = (
        safe_get(tweet, 'tweet_id') or 
        safe_get(tweet, 'id') or 
        safe_get(tweet, 'id_str') or 
        'unknown'
    )
    
    # Extract image features
    image_count = count_images(tweet)
    presence_of_images = image_count > 0
    
    # Extract link features
    link_count = count_links(tweet)
    presence_of_links = link_count > 0
    
    # Extract quote features
    presence_of_quotes = has_quoted_tweet(tweet)
    
    # Depth is passed as parameter
    depth_in_thread = depth
    
    # Extract text and classify with polite-guard
    raw_content = safe_get(tweet, 'raw_content', default='')
    if not isinstance(raw_content, str):
        raw_content = str(raw_content) if raw_content else ''
    
    # Classify text
    classification = classify_text_with_polite_guard(raw_content, classifier)
    
    # Calculate VAD scores
    vad_scores = calculate_vad_scores(raw_content, vad_lexicon)
    
    # Calculate VADER sentiment
    vader_scores = calculate_vader_sentiment(raw_content, vader_analyzer)
    
    return {
        'id': tweet_id,
        'presence_of_images': presence_of_images,
        'image_count': image_count,
        'presence_of_links': presence_of_links,
        'link_count': link_count,
        'presence_of_quotes': presence_of_quotes,
        'depth_in_thread': depth_in_thread,
        'polite_guard_label': classification['label'],
        'polite_guard_score': classification['score'],
        'vad_valence': vad_scores['valence'],
        'vad_arousal': vad_scores['arousal'],
        'vad_dominance': vad_scores['dominance'],
        'vader_compound': vader_scores['compound'],
        'vader_positive': vader_scores['positive'],
        'vader_neutral': vader_scores['neutral'],
        'vader_negative': vader_scores['negative']
    }


def process_nested_replies(replies: List[Dict[str, Any]], depth: int, features_list: List[Dict[str, Any]], classifier=None, vad_lexicon=None, vader_analyzer=None) -> None:
    """
    Recursively process nested replies and extract features.
    
    Args:
        replies: List of reply tweet dictionaries
        depth: Current nesting depth
        features_list: List to append extracted features to
        classifier: Pre-loaded polite-guard classifier
        vad_lexicon: Pre-loaded VAD lexicon dictionary
        vader_analyzer: Pre-loaded VADER sentiment analyzer
    """
    if not isinstance(replies, list):
        return
    
    for reply in replies:
        if not isinstance(reply, dict):
            continue
        
        # Extract features for this reply
        features = extract_tweet_features(reply, depth, classifier, vad_lexicon, vader_analyzer)
        features_list.append(features)
        
        # Process nested replies recursively
        nested_replies = safe_get(reply, 'nested_replies', default=[])
        if nested_replies:
            process_nested_replies(nested_replies, depth + 1, features_list, classifier, vad_lexicon, vader_analyzer)


def process_thread(thread: Dict[str, Any], classifier=None, vad_lexicon=None, vader_analyzer=None) -> List[Dict[str, Any]]:
    """
    Process an entire thread and extract features from all tweets.
    
    Args:
        thread: Thread data dictionary containing 'post' and 'replies'
        classifier: Pre-loaded polite-guard classifier
        vad_lexicon: Pre-loaded VAD lexicon dictionary
        vader_analyzer: Pre-loaded VADER sentiment analyzer
    
    Returns:
        List of feature dictionaries for all tweets in the thread
    """
    features_list = []
    
    # Process root post (depth 0)
    post = safe_get(thread, 'post', default={})
    if isinstance(post, dict) and post:
        post_features = extract_tweet_features(post, depth=0, classifier=classifier, vad_lexicon=vad_lexicon, vader_analyzer=vader_analyzer)
        features_list.append(post_features)
    
    # Process direct replies (depth 1)
    replies = safe_get(thread, 'replies', default=[])
    if isinstance(replies, list):
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            
            # Extract features for this reply
            reply_features = extract_tweet_features(reply, depth=1, classifier=classifier, vad_lexicon=vad_lexicon, vader_analyzer=vader_analyzer)
            features_list.append(reply_features)
            
            # Process nested replies (depth 2+)
            nested_replies = safe_get(reply, 'nested_replies', default=[])
            if nested_replies:
                process_nested_replies(nested_replies, 2, features_list, classifier, vad_lexicon, vader_analyzer)
    
    return features_list


def process_json_file(input_path: str, output_path: str) -> None:
    """
    Process input JSON file and extract features to output file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    """
    try:
        # Load the polite-guard model if enabled
        classifier = load_polite_guard_model()
        
        # Load the VAD lexicon if enabled
        vad_lexicon = load_vad_lexicon()
        
        # Load the VADER analyzer if enabled
        vader_analyzer = load_vader_analyzer()
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_features = []
        
        # Check if data is a dictionary with 'threads' key
        if isinstance(data, dict):
            threads = data.get('threads', [])
            
            # Process each thread
            if isinstance(threads, list):
                total_threads = len(threads)
                for i, thread in enumerate(threads):
                    if not isinstance(thread, dict):
                        print(f"Warning: Thread {i} is not a dictionary, skipping", file=sys.stderr)
                        continue
                    
                    # Progress indicator
                    if (classifier or vad_lexicon or vader_analyzer) and total_threads > 1:
                        print(f"Processing thread {i+1}/{total_threads}...")
                    
                    thread_features = process_thread(thread, classifier, vad_lexicon, vader_analyzer)
                    all_features.extend(thread_features)
            
            # Also check if data itself has post/replies structure
            elif 'post' in data or 'replies' in data:
                thread_features = process_thread(data, classifier, vad_lexicon, vader_analyzer)
                all_features.extend(thread_features)
        
        # If data is a list of threads
        elif isinstance(data, list):
            total_threads = len(data)
            for i, thread in enumerate(data):
                if not isinstance(thread, dict):
                    print(f"Warning: Thread {i} is not a dictionary, skipping", file=sys.stderr)
                    continue
                
                # Progress indicator
                if (classifier or vad_lexicon or vader_analyzer) and total_threads > 1:
                    print(f"Processing thread {i+1}/{total_threads}...")
                
                thread_features = process_thread(thread, classifier, vad_lexicon, vader_analyzer)
                all_features.extend(thread_features)
        
        # Write output file
        output_data = {
            'total_tweets_processed': len(all_features),
            'model_used': POLITE_GUARD_MODEL if classifier else None,
            'vad_lexicon_used': vad_lexicon is not None,
            'vader_used': vader_analyzer is not None,
            'features': all_features
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully processed {len(all_features)} tweets")
        print(f"Output written to: {output_path}")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Total tweets: {len(all_features)}")
        print(f"Tweets with images: {sum(1 for f in all_features if f['presence_of_images'])}")
        print(f"Tweets with links: {sum(1 for f in all_features if f['presence_of_links'])}")
        print(f"Quote tweets: {sum(1 for f in all_features if f['presence_of_quotes'])}")
        print(f"Max depth: {max((f['depth_in_thread'] for f in all_features), default=0)}")
        
        # Print polite-guard statistics if available
        if classifier:
            classified_tweets = [f for f in all_features if f.get('polite_guard_label') is not None]
            print(f"\n=== Polite-Guard Classification ===")
            print(f"Successfully classified: {len(classified_tweets)}/{len(all_features)} tweets")
            
            if classified_tweets:
                # Count labels
                label_counts = {}
                for f in classified_tweets:
                    label = f.get('polite_guard_label', 'Unknown')
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                print("Label distribution:")
                for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(classified_tweets)) * 100
                    print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Print VAD statistics if available
        if vad_lexicon:
            vad_scored_tweets = [f for f in all_features if f.get('vad_valence') is not None]
            print(f"\n=== VAD Lexicon Scoring ===")
            print(f"Successfully scored: {len(vad_scored_tweets)}/{len(all_features)} tweets")
            
            if vad_scored_tweets:
                # Calculate average scores
                avg_valence = sum(f['vad_valence'] for f in vad_scored_tweets) / len(vad_scored_tweets)
                avg_arousal = sum(f['vad_arousal'] for f in vad_scored_tweets) / len(vad_scored_tweets)
                avg_dominance = sum(f['vad_dominance'] for f in vad_scored_tweets) / len(vad_scored_tweets)
                
                print("Average scores (0.0-1.0):")
                print(f"  Valence (pleasure):   {avg_valence:.4f}")
                print(f"  Arousal (activation): {avg_arousal:.4f}")
                print(f"  Dominance (control):  {avg_dominance:.4f}")
        
        # Print VADER statistics if available
        if vader_analyzer:
            vader_scored_tweets = [f for f in all_features if f.get('vader_compound') is not None]
            print(f"\n=== VADER Sentiment Analysis ===")
            print(f"Successfully scored: {len(vader_scored_tweets)}/{len(all_features)} tweets")
            
            if vader_scored_tweets:
                # Calculate average scores
                avg_compound = sum(f['vader_compound'] for f in vader_scored_tweets) / len(vader_scored_tweets)
                avg_positive = sum(f['vader_positive'] for f in vader_scored_tweets) / len(vader_scored_tweets)
                avg_neutral = sum(f['vader_neutral'] for f in vader_scored_tweets) / len(vader_scored_tweets)
                avg_negative = sum(f['vader_negative'] for f in vader_scored_tweets) / len(vader_scored_tweets)
                
                print("Average scores:")
                print(f"  Compound (-1 to 1):  {avg_compound:.4f}")
                print(f"  Positive (0 to 1):   {avg_positive:.4f}")
                print(f"  Neutral (0 to 1):    {avg_neutral:.4f}")
                print(f"  Negative (0 to 1):   {avg_negative:.4f}")
                
                # Count sentiment categories
                positive_count = sum(1 for f in vader_scored_tweets if f['vader_compound'] >= 0.05)
                neutral_count = sum(1 for f in vader_scored_tweets if -0.05 < f['vader_compound'] < 0.05)
                negative_count = sum(1 for f in vader_scored_tweets if f['vader_compound'] <= -0.05)
                
                print("\nSentiment distribution:")
                print(f"  Positive: {positive_count} ({positive_count/len(vader_scored_tweets)*100:.1f}%)")
                print(f"  Neutral:  {neutral_count} ({neutral_count/len(vader_scored_tweets)*100:.1f}%)")
                print(f"  Negative: {negative_count} ({negative_count/len(vader_scored_tweets)*100:.1f}%)")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the script."""
    # Use hardcoded paths from configuration section
    if len(sys.argv) < 2:
        # Use configured paths
        input_path = INPUT_FILE
        output_path = OUTPUT_FILE
        print(f"Using configured paths:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print()
    else:
        # Command line arguments override configured paths
        input_path = sys.argv[1]
        
        # Generate default output filename if not provided
        if len(sys.argv) >= 3:
            output_path = sys.argv[2]
        else:
            input_stem = Path(input_path).stem
            output_path = f"{input_stem}_features.json"
        
        print(f"Using command line arguments:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print()
    
    process_json_file(input_path, output_path)


if __name__ == '__main__':
    main()
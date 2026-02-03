"""
GPT-based UNIFIED Hate Speech + Counterspeech Classification
with FULL THREAD CONTEXT and Single-Task Consistency

IMPROVEMENTS:
- Single API call for both HS and CS (eliminates contradiction)
- Enforced consistency between classifications
- Clear decision flowchart
- Improved prompting with examples
- 50% cost reduction
"""

import json
import time
import os
import re
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODELS_TO_RUN = ["gpt-4o-mini"]

# Input files
INPUT_CSV_PATH = "ground_truth_dataset.csv"
THREAD_JSON_PATH = "extracted_68_threads.json"
OUTPUT_DIR = "results_unified_classification"

REQUEST_DELAY = 1
MAX_RETRIES = 3

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600}
}

# ============================================================
# THREAD DATA LOADING
# ============================================================

def load_thread_data(json_path: str) -> Dict[int, Dict]:
    """
    Load thread data and create a mapping of tweet_id -> thread context.
    Returns: dict mapping tweet_id to full thread structure
    """
    print(f"[load_threads] Loading {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    thread_map = {}
    
    # Process each thread
    for thread in data.get('threads', []):
        post = thread.get('post', {})
        replies = thread.get('replies', [])
        
        # Add main post to map
        post_id = post.get('tweet_id')
        if post_id:
            thread_map[post_id] = {
                'target_tweet': post,
                'thread': thread,
                'is_reply': False,
                'parent': None
            }
        
        # Process all replies (including nested)
        def process_replies(reply_list, parent_tweet):
            for reply in reply_list:
                reply_id = reply.get('tweet_id')
                if reply_id:
                    thread_map[reply_id] = {
                        'target_tweet': reply,
                        'thread': thread,
                        'is_reply': True,
                        'parent': parent_tweet
                    }
                
                # Process nested replies recursively
                nested = reply.get('nested_replies', [])
                if nested:
                    process_replies(nested, reply)
        
        process_replies(replies, post)
    
    print(f"[load_threads] Loaded {len(thread_map)} tweets across {len(data.get('threads', []))} threads")
    return thread_map


def load_data_from_csv_with_threads(csv_path: str, thread_map: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Load ground truth data and enrich with thread context.
    Returns: (data_for_classification, ground_truth)
    """
    print(f"[load_csv] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[load_csv] Loaded {len(df)} rows")
    
    data = []
    truth = []
    missing_threads = []
    
    for _, row in df.iterrows():
        tweet_id = int(row['Tweet ID'])
        
        # Get thread context
        thread_context = thread_map.get(tweet_id)
        
        if not thread_context:
            missing_threads.append(tweet_id)
            print(f"[load_csv] WARNING: No thread context for tweet {tweet_id}")
            continue
        
        target_tweet = thread_context['target_tweet']
        full_thread = thread_context['thread']
        
        # Build classification input with full thread context
        item = {
            "ID": tweet_id,
            "Text": target_tweet.get('raw_content', ''),
            "Parent_Text": thread_context['parent'].get('raw_content', '') if thread_context['parent'] else '',
            "Is_Reply": thread_context['is_reply'],
            "Thread_Context": build_thread_context_string(full_thread, tweet_id)
        }
        
        data.append(item)
        
        # Build ground truth
        gt_item = {
            "id": tweet_id,
            "hate_speech": int(row['hate_speech']) if pd.notna(row['hate_speech']) else None,
            "hate_speech_type": str(row['hate_speech_type']) if pd.notna(row['hate_speech_type']) else None,
            "identity_targeted": str(row['identity_targeted']) if pd.notna(row['identity_targeted']) else None,
            "counterspeech": int(row['counterspeech']) if pd.notna(row['counterspeech']) else None,
            "counterspeech_type": str(row['counterspeech_type']) if pd.notna(row['counterspeech_type']) else None,
            "dominant_counterspeech_type": str(row['dominant_counterspeech_type']) if pd.notna(row['dominant_counterspeech_type']) else None
        }
        truth.append(gt_item)
    
    if missing_threads:
        print(f"[load_csv] WARNING: {len(missing_threads)} tweets missing thread context: {missing_threads[:10]}...")
    
    print(f"[load_csv] Created {len(data)} classification inputs and {len(truth)} ground truth records")
    return data, truth


def build_thread_context_string(thread: Dict, target_tweet_id: int) -> str:
    """
    Build a readable string representation of the thread for context.
    Marks the target tweet that needs classification.
    """
    lines = []
    lines.append("=== THREAD CONTEXT ===")
    
    # Main post
    post = thread.get('post', {})
    post_id = post.get('tweet_id')
    is_target = (post_id == target_tweet_id)
    
    prefix = ">>> TARGET TWEET >>>" if is_target else "Original Post:"
    lines.append(f"\n{prefix}")
    lines.append(f"ID: {post_id}")
    lines.append(f"User: @{post.get('username', 'unknown')}")
    lines.append(f"Date: {post.get('date', 'unknown')}")
    lines.append(f"Text: {post.get('raw_content', '')}")
    
    # Replies
    def format_replies(reply_list, depth=0, parent_id=None):
        for reply in reply_list:
            reply_id = reply.get('tweet_id')
            is_target = (reply_id == target_tweet_id)
            
            indent = "  " * depth
            prefix = f"{indent}>>> TARGET TWEET >>>" if is_target else f"{indent}Reply to {parent_id}:"
            
            lines.append(f"\n{prefix}")
            lines.append(f"{indent}ID: {reply_id}")
            lines.append(f"{indent}User: @{reply.get('username', 'unknown')}")
            lines.append(f"{indent}Date: {reply.get('date', 'unknown')}")
            lines.append(f"{indent}Text: {reply.get('raw_content', '')}")
            
            # Process nested replies
            nested = reply.get('nested_replies', [])
            if nested:
                format_replies(nested, depth + 1, reply_id)
    
    replies = thread.get('replies', [])
    if replies:
        format_replies(replies, 0, post_id)
    
    lines.append("\n=== END THREAD CONTEXT ===")
    return "\n".join(lines)


# ============================================================
# UNIFIED CLASSIFICATION PROMPT
# ============================================================

unified_classification_prompt = """You are an expert content moderator specializing in hate speech detection and counterspeech identification.

TASK: Classify a single tweet for BOTH:
1. HATE SPEECH: Does this tweet itself contain hate speech?
2. COUNTERSPEECH: Does this tweet oppose/challenge hate speech that appeared earlier in the thread?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL CONSISTENCY REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your hate speech and counterspeech classifications MUST be logically consistent:

IF you determine a tweet is "opposition to hate" → counterspeech MUST = 1
IF you say "hostile to the hater" → counterspeech MUST = 1  
IF you say "mocks/satirizes the hate" → counterspeech MUST = 1
IF you say "questions the hater" → counterspeech MUST = 1
IF you say "defensive response to hate" → counterspeech MUST = 1

You cannot mark something as "opposition" and then say it's NOT counterspeech.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

==================================================
INPUT FORMAT
==================================================

You receive JSON with:
- tweet_id: numeric identifier of the TARGET tweet
- raw_content: the TARGET tweet text (the one you're classifying)
- parent_raw_content: immediate parent tweet (if this is a reply)
- thread_context: FULL conversation showing all messages in the thread

⚠️ CRITICAL: Classify ONLY the tweet marked ">>> TARGET TWEET >>>". 
Thread context helps you understand the conversation but is NOT evidence itself.

==================================================
DECISION FLOWCHART - FOLLOW THIS EXACTLY
==================================================

START: Review the target tweet and thread context

┌─ STEP 1: Does the TARGET TWEET itself contain hate speech?
│
│  Hate speech definition:
│  - Dehumanization (vermin, animals, disease)
│  - Calls for exclusion, segregation, removal, or harm
│  - Slurs/epithets attacking a protected group
│  - Claims of inherent inferiority or criminality
│  - Denial of group existence/legitimacy (erasure)
│  - Endorsement of violence or extremism against groups
│
│  ├─ YES, target tweet contains hate speech itself
│  │  → hate_speech = 1
│  │  → counterspeech = 0 (can't be both hater and counter-hater)
│  │  → Classify type and target
│  │  → DONE - Output result
│  │
│  └─ NO, target tweet does not contain hate speech
│     → hate_speech = 0
│     → Continue to STEP 2
│
├─ STEP 2: Is there hate speech EARLIER in the thread?
│
│  Review thread_context for earlier messages containing:
│  - Any of the hate speech elements above
│  - Hateful content (even if quoted/reported)
│
│  ├─ NO hate speech earlier
│  │  → counterspeech = 0 (nothing to counter)
│  │  → DONE - Output result
│  │
│  └─ YES, hate speech exists earlier in thread
│     → Continue to STEP 3
│
└─ STEP 3: Does the TARGET TWEET oppose/challenge the earlier hate?

   Use ANY of these criteria (liberal interpretation):
   
   ✓ QUESTIONING THE HATER:
     - "why would you say that?"
     - "what is wrong with you?"
     - "are you serious?"
     - Rhetorical questions challenging the hate
   
   ✓ HOSTILE LANGUAGE TOWARD THE HATER:
     - "fuck you, you bigot"
     - "you're disgusting for saying that"
     - "hope you get fired"
     - Insults/threats aimed at the hate speaker
   
   ✓ MOCKERY/SATIRE OF THE HATE:
     - Absurdist exaggeration making hate look ridiculous
     - Sarcastic escalation exposing absurdity
     - Humor that undermines the hateful premise
   
   ✓ DEFENSIVE RESPONSES:
     - "Shut up yourself"
     - "some of us have..."
     - Asserting individual action against stereotypes
   
   ✓ EXPLICIT DENOUNCEMENT:
     - "that's racist/sexist/homophobic"
     - "this is hate speech"
     - Direct condemnation
   
   ✓ PRESENTING COUNTER-FACTS:
     - Correcting misinformation with evidence
     - Providing data that refutes the hate
   
   ✓ CHALLENGING LOGIC:
     - Pointing out contradictions
     - "your reasoning is flawed"
   
   ✓ WARNING OF CONSEQUENCES:
     - "this will have repercussions"
     - "you just got fired"
   
   ✓ EMPATHY/HUMANIZATION:
     - Defending the targeted group
     - Promoting inclusion or compassion
   
   ├─ YES, target opposes earlier hate (ANY criterion above)
   │  → counterspeech = 1
   │  → Classify counterspeech type
   │  → DONE - Output result
   │
   └─ NO, target does not oppose earlier hate
      → counterspeech = 0
      → DONE - Output result

==================================================
HATE SPEECH CLASSIFICATION DETAILS
==================================================

PROTECTED GROUPS (non-exhaustive):
Race, ethnicity, nationality, religion, caste, disability, sexual orientation, 
gender identity, sex, immigrant status, age groups, indigenous peoples

EXPLICIT HATE SPEECH:
- Direct slurs ("f*ggot", "n*gger", "retard")
- Clear dehumanization ("they're animals", "vermin")
- Direct calls for harm ("kill all X", "deport them")
- Denial of existence ("Palestine doesn't exist")
- No plausible alternative interpretation

IMPLICIT HATE SPEECH:
- Coded language/dog whistles ("urban youth", "globalists")
- Stereotypes as fact ("all X are criminals")
- Dehumanization through "jokes"
- Gendered body-shaming patterns
- Age-based generalizations that demean
- Requires context to identify

IDENTITY TARGETS (choose primary):
Women | Men | Black | White | Jewish | Muslim | Middle Eastern | Christian |
Asian | Latino/Hispanic | LGBTQ+ | Immigrants/refugees | Disabled people | 
Elderly | Indigenous | Palestinian | Transgender | Other | Unspecified

NOT HATE SPEECH:
❌ Political criticism ("I disagree with this policy")
❌ Personal insults not based on identity ("John is an idiot")
❌ Venting without dehumanization ("ugh men are annoying today")
❌ Discussing/condemning hate ("using slurs is wrong")
❌ Calling out bigotry ("that's racist")
❌ Attacking the hate speaker ("you're disgusting for saying that")

CRITICAL RULE: Thread context is for UNDERSTANDING, not EVIDENCE.
- Don't use "aligns with earlier messages" as reasoning
- Don't use "in context of thread" to upgrade non-hate to hate
- The tweet must be hateful on its own

==================================================
COUNTERSPEECH CLASSIFICATION DETAILS
==================================================

COUNTERSPEECH TYPES (pick best match):

1) presenting facts
   - Corrects misinformation with evidence
   - Provides data that refutes hate

2) challenging logic
   - Points out inconsistencies or flaws
   - Asks pointed counter-questions

3) warning of consequences
   - Mentions social, legal, or personal repercussions
   - "you got fired", "this will backfire"

4) affiliation
   - Establishes solidarity with targeted group
   - "I am [group] and..."

5) denouncing hate speech
   - Explicit condemnation: "that's racist"
   - Labels it as bigotry directly

6) humor
   - Sarcasm or jokes undermining hate
   - Absurdist mockery

7) empathy/positive tone
   - Promotes compassion or humanization
   - Inclusion and understanding

8) hostile language
   - Insults/profanity aimed at the hate speaker
   - While opposing the hate

dominant_counterspeech_type = same as counterspeech_type (single category)

==================================================
SPECIAL CASES AND EDGE CASES
==================================================

VENTING vs HATE:
"God I hate men today" = venting (frustration, no dehumanization)
"Men are subhuman trash" = hate (dehumanization)

ENDORSEMENT vs OPPOSITION (for replies):
Endorsement: "So true!", "Exactly!", "Facts", agreeing WITH the hate
Opposition: ANY challenge, question, mockery, or hostility to the HATER

HOSTILITY TARGET:
Hostile to the HATER = counterspeech ("you're disgusting")
Hostile to the TARGETED GROUP = hate speech ("they're disgusting")

QUOTED/REPORTED HATE:
If thread contains hate (even quoted), target can still counter it.
Example: "Someone said [slur]" → Next tweet: "that's awful" = counterspeech

==================================================
REASONING REQUIREMENTS
==================================================

Your reasoning MUST:
1. State whether earlier thread messages contain hate (if relevant)
2. Explain what the TARGET TWEET itself contains/does
3. Be concise (2-4 sentences max)
4. NOT quote the original text verbatim
5. Reference which rule/category applies
6. Be CONSISTENT with your classifications

REQUIRED patterns:
✓ "The target tweet itself contains [element]"
✓ "Independent of context, this attacks [group]"
✓ "The tweet opposes earlier hate by [action]"
✓ "Earlier in thread, X said Y; target responds with Z"

FORBIDDEN patterns:
❌ "aligns with earlier messages"
❌ "in context of the thread, this becomes hate"
❌ "combined with earlier statements"
❌ Saying it's "opposition" but marking counterspeech=0

==================================================
SELF-CHECK BEFORE OUTPUTTING
==================================================

Before finalizing, verify:

1. ✓ If I said "opposition" anywhere → counterspeech = 1?
2. ✓ If I said "hostile to hater" → counterspeech = 1?
3. ✓ If I said "mocks/satirizes" → counterspeech = 1?
4. ✓ If I said "questions the hater" → counterspeech = 1?
5. ✓ hate_speech and counterspeech are not both 1?
6. ✓ Reasoning matches my classifications?

If any check fails, FIX IT before outputting.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON (no markdown, no extra text):

{
  "id": <tweet_id>,
  "hate_speech": 0 or 1,
  "hate_speech_type": "explicit" | "implicit" | null,
  "identity_targeted": "Women" | "Men" | ... | null,
  "counterspeech": 0 or 1,
  "counterspeech_type": "presenting facts" | "challenging logic" | ... | null,
  "dominant_counterspeech_type": "<same as counterspeech_type>" | null,
  "reasoning": "Brief unified explanation for both classifications"
}

Rules:
- If hate_speech=0, hate_speech_type and identity_targeted must be null
- If hate_speech=1, hate_speech_type and identity_targeted must be set
- If counterspeech=0, counterspeech_type and dominant_counterspeech_type must be null
- If counterspeech=1, both counterspeech types must be set (same value)
- hate_speech and counterspeech cannot both be 1
- reasoning must be a single coherent explanation for both classifications
"""

# ============================================================
# UNIFIED EXAMPLES
# ============================================================

unified_examples = [
    # Example 1: Clear hate speech (no counterspeech possible)
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1000,
  "raw_content": "All [immigrants] are criminals ruining our country. Deport them all!",
  "parent_raw_content": "",
  "thread_context": "=== THREAD CONTEXT ===\\n\\n>>> TARGET TWEET >>>\\nID: 1000\\nUser: @hater\\nText: All [immigrants] are criminals ruining our country. Deport them all!\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1000,
  "hate_speech": 1,
  "hate_speech_type": "explicit",
  "identity_targeted": "Immigrants/refugees",
  "counterspeech": 0,
  "counterspeech_type": null,
  "dominant_counterspeech_type": null,
  "reasoning": "The target tweet itself contains explicit hate speech through harmful stereotyping (all immigrants are criminals) and calls for forced removal (deport them all), directly targeting immigrants as a protected group."
}'''
    },
    
    # Example 2: Endorsement (is hate speech)
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1001,
  "raw_content": "So true 👍👍",
  "parent_raw_content": "Women are too emotional to be leaders.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @sexist\\nText: Women are too emotional to be leaders.\\n\\n>>> TARGET TWEET >>>\\nID: 1001\\nUser: @endorser\\nText: So true 👍👍\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1001,
  "hate_speech": 1,
  "hate_speech_type": "implicit",
  "identity_targeted": "Women",
  "counterspeech": 0,
  "counterspeech_type": null,
  "dominant_counterspeech_type": null,
  "reasoning": "Earlier in thread, a harmful stereotype about women was posted. The target tweet explicitly endorses this hate speech with 'so true' and affirmation emojis, making it implicit hate speech targeting women."
}'''
    },
    
    # Example 3: Hostile language to hater = counterspeech
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1002,
  "raw_content": "Dude you got fired! 😂 I hope you don't get hired anywhere. Best of luck dumb fuck! 😆",
  "parent_raw_content": "You know what I hate Niggers.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @hater\\nText: You know what I hate Niggers.\\n\\n>>> TARGET TWEET >>>\\nID: 1002\\nUser: @opposer\\nText: Dude you got fired! 😂 I hope you don't get hired anywhere. Best of luck dumb fuck! 😆\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1002,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 1,
  "counterspeech_type": "hostile language",
  "dominant_counterspeech_type": "hostile language",
  "reasoning": "Earlier message contains explicit hate speech with a racial slur. The target tweet opposes this by directing hostile language and warnings of consequences at the hate speaker ('you got fired', 'dumb fuck'), making it counterspeech via hostile language toward the hater."
}'''
    },
    
    # Example 4: Questioning = counterspeech
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1003,
  "raw_content": "what is wrong with you?",
  "parent_raw_content": "I hope all f*ggots burn in hell. Disgusting freaks.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @homophobe\\nText: I hope all f*ggots burn in hell. Disgusting freaks.\\n\\n>>> TARGET TWEET >>>\\nID: 1003\\nUser: @questioner\\nText: what is wrong with you?\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1003,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 1,
  "counterspeech_type": "challenging logic",
  "dominant_counterspeech_type": "challenging logic",
  "reasoning": "Earlier message contains explicit hate speech with slurs and calls for harm against LGBTQ+ people. The target tweet opposes this by questioning the hater's behavior, which challenges the hate speech and qualifies as counterspeech."
}'''
    },
    
    # Example 5: Absurdist mockery = counterspeech
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1004,
  "raw_content": "yeah they buy all the uncrustables and kid cuisine and stealing all the good titty implants",
  "parent_raw_content": "i HATE transgender people. THEY FREAKING SUCK!!!!!!!!!!!!!!!!!!!!!!!!!",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @transphobe\\nText: i HATE transgender people. THEY FREAKING SUCK!!!!!!!!!!!!!!!!!!!!!!!!!\\n\\n>>> TARGET TWEET >>>\\nID: 1004\\nUser: @mocker\\nText: yeah they buy all the uncrustables and kid cuisine and stealing all the good titty implants\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1004,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 1,
  "counterspeech_type": "humor",
  "dominant_counterspeech_type": "humor",
  "reasoning": "Earlier message contains explicit hate speech against transgender people. The target tweet uses absurdist humor (claiming they buy all the Uncrustables and steal implants) to mock and ridicule the hateful premise, making it counterspeech through satire."
}'''
    },
    
    # Example 6: Defensive response = counterspeech
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1005,
  "raw_content": "Shut up yourself. Some of us Mayo butts have voted democrat in every election since we were 18. I am 70. Fight me. Damn.",
  "parent_raw_content": "Shut up. I'm sick of white people who expect Democrats to do the job they themselves cannot or will not do. Maybe get off your mayonnaise asses and confront them.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @frustrated\\nText: Shut up. I'm sick of white people who expect Democrats to do the job they themselves cannot or will not do. Maybe get off your mayonnaise asses and confront them.\\n\\n>>> TARGET TWEET >>>\\nID: 1005\\nUser: @defender\\nText: Shut up yourself. Some of us Mayo butts have voted democrat in every election since we were 18. I am 70. Fight me. Damn.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1005,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 1,
  "counterspeech_type": "challenging logic",
  "dominant_counterspeech_type": "challenging logic",
  "reasoning": "Earlier message expresses frustration with white people's political behavior (borderline hate speech). The target tweet opposes this by providing a defensive counter-example ('some of us have voted democrat', 'I am 70'), challenging the sweeping generalization with personal evidence."
}'''
    },
    
    # Example 7: Venting (NOT hate, NOT counterspeech)
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1006,
  "raw_content": "God I hate men today #annoyed",
  "parent_raw_content": "",
  "thread_context": "=== THREAD CONTEXT ===\\n\\n>>> TARGET TWEET >>>\\nID: 1006\\nUser: @venter\\nText: God I hate men today #annoyed\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1006,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 0,
  "counterspeech_type": null,
  "dominant_counterspeech_type": null,
  "reasoning": "The target tweet expresses temporary frustration ('today', '#annoyed') without dehumanization or calls for harm. This is venting, not hate speech. No earlier hate speech exists in the thread, so counterspeech is not applicable."
}'''
    },
    
    # Example 8: Fact-checking = counterspeech
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1007,
  "raw_content": "That's just false. FBI reports show immigrant crime rates are lower than native-born citizens.",
  "parent_raw_content": "Immigrants commit all the crimes. It's obvious.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @xenophobe\\nText: Immigrants commit all the crimes. It's obvious.\\n\\n>>> TARGET TWEET >>>\\nID: 1007\\nUser: @factchecker\\nText: That's just false. FBI reports show immigrant crime rates are lower than native-born citizens.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1007,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 1,
  "counterspeech_type": "presenting facts",
  "dominant_counterspeech_type": "presenting facts",
  "reasoning": "Earlier message contains hate speech through harmful stereotyping of immigrants. The target tweet opposes this by presenting factual evidence (FBI reports) that directly refutes the hateful claim, making it counterspeech through fact presentation."
}'''
    },
    
    # Example 9: Denouncement = counterspeech
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1008,
  "raw_content": "This is literally homophobic. You're saying you prefer straight relationships over LGBT ones because of their orientation.",
  "parent_raw_content": "I just prefer straight ships over LGBT ships in my media. Nothing wrong with that.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @shipper\\nText: I just prefer straight ships over LGBT ships in my media. Nothing wrong with that.\\n\\n>>> TARGET TWEET >>>\\nID: 1008\\nUser: @caller-outer\\nText: This is literally homophobic. You're saying you prefer straight relationships over LGBT ones because of their orientation.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1008,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 1,
  "counterspeech_type": "denouncing hate speech",
  "dominant_counterspeech_type": "denouncing hate speech",
  "reasoning": "Earlier message contains implicit hate speech (preference against LGBT relationships based on orientation). The target tweet opposes this by explicitly labeling it as homophobic and explaining why, making it counterspeech through direct denouncement."
}'''
    },
    
    # Example 10: Not counterspeech (unrelated reply)
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1009,
  "raw_content": "I meant globally, not just in the US.",
  "parent_raw_content": "Trans women shouldn't be allowed to compete. It's unfair.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @transphobe\\nText: Trans women shouldn't be allowed to compete. It's unfair.\\n\\nReply:\\nID: 1001\\nText: She's ranked 462 in women's swimming.\\n\\n>>> TARGET TWEET >>>\\nID: 1009\\nUser: @clarifier\\nText: I meant globally, not just in the US.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": '''{
  "id": 1009,
  "hate_speech": 0,
  "hate_speech_type": null,
  "identity_targeted": null,
  "counterspeech": 0,
  "counterspeech_type": null,
  "dominant_counterspeech_type": null,
  "reasoning": "Earlier in thread, there is hate speech against trans women. However, the target tweet is a clarification about ranking scope ('I meant globally') that does not challenge, oppose, or engage with the hateful content. It is simply a factual clarification unrelated to countering the hate."
}'''
    }
]

# ============================================================
# ROBUST JSON PARSING
# ============================================================

def parse_json_response(text: str, tweet_id: int = None) -> Dict:
    """
    Parse JSON response with multiple fallback strategies.
    """
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith('```'):
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        else:
            raise ValueError(f"Expected dict, got: {type(parsed)}")
    except json.JSONDecodeError as e:
        print(f"[parse_json] Parse error at char {e.pos}: {e.msg}")
        print(f"[parse_json] Context: ...{text[max(0, e.pos-50):e.pos+50]}...")
        
        # Try to extract just the JSON object
        obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except:
                pass
        
        # Manual extraction as last resort
        print(f"[parse_json] Attempting manual field extraction...")
        try:
            result = {}
            
            # Extract id
            id_match = re.search(r'"id"\s*:\s*(\d+)', text)
            result['id'] = int(id_match.group(1)) if id_match else tweet_id
            
            # Extract hate_speech
            hs_match = re.search(r'"hate_speech"\s*:\s*(\d+)', text)
            if hs_match:
                result['hate_speech'] = int(hs_match.group(1))
            
            # Extract counterspeech
            cs_match = re.search(r'"counterspeech"\s*:\s*(\d+)', text)
            if cs_match:
                result['counterspeech'] = int(cs_match.group(1))
            
            # Extract hate_speech_type
            hst_match = re.search(r'"hate_speech_type"\s*:\s*"([^"]*)"', text)
            if hst_match:
                val = hst_match.group(1)
                result['hate_speech_type'] = None if val.lower() == 'null' else val
            else:
                if re.search(r'"hate_speech_type"\s*:\s*null', text):
                    result['hate_speech_type'] = None
            
            # Extract identity_targeted
            it_match = re.search(r'"identity_targeted"\s*:\s*"([^"]*)"', text)
            if it_match:
                val = it_match.group(1)
                result['identity_targeted'] = None if val.lower() == 'null' else val
            else:
                if re.search(r'"identity_targeted"\s*:\s*null', text):
                    result['identity_targeted'] = None
            
            # Extract counterspeech_type
            cst_match = re.search(r'"counterspeech_type"\s*:\s*"([^"]*)"', text)
            if cst_match:
                val = cst_match.group(1)
                result['counterspeech_type'] = None if val.lower() == 'null' else val
            else:
                if re.search(r'"counterspeech_type"\s*:\s*null', text):
                    result['counterspeech_type'] = None
            
            # Extract dominant_counterspeech_type
            dcst_match = re.search(r'"dominant_counterspeech_type"\s*:\s*"([^"]*)"', text)
            if dcst_match:
                val = dcst_match.group(1)
                result['dominant_counterspeech_type'] = None if val.lower() == 'null' else val
            else:
                if re.search(r'"dominant_counterspeech_type"\s*:\s*null', text):
                    result['dominant_counterspeech_type'] = None
            
            # Extract reasoning
            rs_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if rs_match:
                result['reasoning'] = rs_match.group(1).replace('\\"', '"')
            else:
                result['reasoning'] = "Manual extraction - reasoning unavailable"
            
            print(f"[parse_json] Manual extraction successful")
            return result
            
        except Exception as manual_error:
            print(f"[parse_json] Manual extraction failed: {manual_error}")
            raise ValueError(f"Could not parse JSON. Original error: {e.msg}. Text: {text[:300]}")


# ============================================================
# CONSISTENCY VALIDATION
# ============================================================

def validate_consistency(result: Dict) -> Tuple[bool, str]:
    """
    Validate that hate speech and counterspeech classifications are logically consistent.
    Returns: (is_valid, message)
    """
    reasoning = result.get('reasoning', '').lower()
    hs = result.get('hate_speech', 0)
    cs = result.get('counterspeech', 0)
    
    # Rule 1: Can't be both hate speech and counterspeech
    if hs == 1 and cs == 1:
        return False, "Both hate_speech and counterspeech are 1 (impossible)"
    
    # Rule 2: If reasoning mentions opposition, must be counterspeech
    opposition_keywords = [
        'opposition', 'opposes', 'challenges', 'questions the hater',
        'hostile to hater', 'hostile to the hater', 'absurdist', 'mockery', 
        'satire', 'defensive response', 'counter', 'refutes', 'denounc'
    ]
    
    has_opposition = any(kw in reasoning for kw in opposition_keywords)
    
    if has_opposition and hs == 0 and cs == 0:
        return False, f"Reasoning indicates opposition but counterspeech=0"
    
    # Rule 3: If hate_speech=1, type and target must be set
    if hs == 1:
        if not result.get('hate_speech_type') or not result.get('identity_targeted'):
            return False, "hate_speech=1 but type or target not set"
    
    # Rule 4: If hate_speech=0, type and target must be null
    if hs == 0:
        if result.get('hate_speech_type') or result.get('identity_targeted'):
            return False, "hate_speech=0 but type or target are set"
    
    # Rule 5: If counterspeech=1, type must be set
    if cs == 1:
        if not result.get('counterspeech_type') or not result.get('dominant_counterspeech_type'):
            return False, "counterspeech=1 but type not set"
    
    # Rule 6: If counterspeech=0, types must be null
    if cs == 0:
        if result.get('counterspeech_type') or result.get('dominant_counterspeech_type'):
            return False, "counterspeech=0 but types are set"
    
    return True, "Consistent"


# ============================================================
# API CALL WITH RETRY
# ============================================================

def call_gpt_api(client: OpenAI, messages: List[Dict], model: str, tweet_id: int = None) -> Tuple[str, int, int]:
    """
    Call GPT API with retry logic and JSON fix capability.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[call_gpt_api] model={model} attempt={attempt}/{MAX_RETRIES}")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=400
            )
            
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Try to parse the response
            try:
                parsed = parse_json_response(content, tweet_id)
                
                # Validate consistency
                is_valid, msg = validate_consistency(parsed)
                if not is_valid:
                    print(f"[call_gpt_api] Consistency error: {msg}")
                    
                    if attempt < MAX_RETRIES:
                        # Ask model to fix
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"""CONSISTENCY ERROR: {msg}

Please review your classification and fix the error. Remember:
- If you say the tweet is "opposition" or "opposes hate", counterspeech MUST = 1
- hate_speech and counterspeech cannot both be 1
- If hate_speech=0, hate_speech_type and identity_targeted must be null
- If counterspeech=0, counterspeech_type must be null

Output the corrected JSON."""
                        })
                        time.sleep(REQUEST_DELAY)
                        continue
                
                # If parsing and validation succeeded, return
                return content, prompt_tokens, completion_tokens
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"[call_gpt_api] JSON parse error on attempt {attempt}: {parse_error}")
                
                if attempt < MAX_RETRIES:
                    # Add a fix request
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"""Your JSON had an error: {str(parse_error)[:200]}

Please output ONLY valid JSON with no extra text:

{{
  "id": {tweet_id if tweet_id else "tweet_id"},
  "hate_speech": 0 or 1,
  "hate_speech_type": "explicit" or "implicit" or null,
  "identity_targeted": "Group" or null,
  "counterspeech": 0 or 1,
  "counterspeech_type": "presenting facts" or ... or null,
  "dominant_counterspeech_type": "<same as counterspeech_type>" or null,
  "reasoning": "explanation here"
}}"""
                    })
                    time.sleep(REQUEST_DELAY)
                    continue
                else:
                    # Last attempt - use manual extraction result
                    return content, prompt_tokens, completion_tokens
                    
        except Exception as e:
            print(f"[call_gpt_api] API error on attempt {attempt}: {repr(e)}")
            
            if attempt == MAX_RETRIES:
                raise
            
            time.sleep(REQUEST_DELAY * 2)
    
    raise Exception("Max retries exceeded")


# ============================================================
# HELPERS
# ============================================================

def save_json(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[save_json] Wrote {len(data)} records -> {path}")


# ============================================================
# UNIFIED CLASSIFICATION FUNCTION
# ============================================================

def classify_unified(client: OpenAI, data: List[Dict], model: str):
    """
    Perform unified classification for both hate speech and counterspeech in a single API call.
    """
    print(f"[UNIFIED] Starting classification | model={model} | n={len(data)}")
    results, in_tok, out_tok = [], 0, 0
    consistency_warnings = 0

    for i, item in enumerate(data, start=1):
        tweet_id = item["ID"]
        
        payload = {
            "tweet_id": tweet_id,
            "raw_content": item["Text"],
            "parent_raw_content": item.get("Parent_Text", ""),
            "thread_context": item.get("Thread_Context", "")
        }

        print(f"[UNIFIED] ({i}/{len(data)}) tweet_id={tweet_id}")

        messages = [{"role": "system", "content": unified_classification_prompt}]
        messages.extend(unified_examples)
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

        try:
            text, it, ot = call_gpt_api(client, messages, model, tweet_id=tweet_id)
            parsed = parse_json_response(text, tweet_id)
            
            # Ensure id is correct
            parsed["id"] = tweet_id
            
            # Final validation
            is_valid, msg = validate_consistency(parsed)
            if not is_valid:
                print(f"[UNIFIED] WARNING tweet {tweet_id}: {msg}")
                consistency_warnings += 1
            
            results.append(parsed)
            in_tok += it
            out_tok += ot
            
        except Exception as e:
            print(f"[UNIFIED] FATAL ERROR on tweet {tweet_id}: {e}")
            print(f"[UNIFIED] Adding placeholder result and continuing...")
            
            results.append({
                "id": tweet_id,
                "hate_speech": 0,
                "hate_speech_type": None,
                "identity_targeted": None,
                "counterspeech": 0,
                "counterspeech_type": None,
                "dominant_counterspeech_type": None,
                "reasoning": f"ERROR: {str(e)[:100]}"
            })
        
        time.sleep(REQUEST_DELAY)

    print(f"[UNIFIED] Finished | total_in={in_tok} total_out={out_tok}")
    print(f"[UNIFIED] Consistency warnings: {consistency_warnings}/{len(data)}")
    return results, in_tok, out_tok


# ============================================================
# EVALUATION
# ============================================================

def evaluate_classifications(preds: List[Dict], truth: List[Dict]) -> Dict:
    print("[eval] Aligning predictions with ground truth by ID")

    pred_df = pd.DataFrame(preds).set_index("id")
    truth_df = pd.DataFrame(truth).set_index("id")

    common_ids = pred_df.index.intersection(truth_df.index)

    print(f"[eval] preds={len(pred_df)} truth={len(truth_df)} overlap={len(common_ids)}")

    if len(common_ids) == 0:
        raise ValueError("No overlapping IDs between predictions and ground truth")

    pred_df = pred_df.loc[common_ids].sort_index()
    truth_df = truth_df.loc[common_ids].sort_index()

    metrics = {}

    for label in ["hate_speech", "counterspeech"]:
        mask = truth_df[label].notna()
        
        if not mask.any():
            print(f"[eval] Skipping {label}: no ground truth available")
            continue

        y_true = truth_df.loc[mask, label].astype(int).values
        y_pred = pred_df.loc[mask, label].astype(int).values

        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        metrics[label] = {
            "n_eval": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": p,
            "recall": r,
            "f1": f,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

        print(
            f"[eval] {label}: n={len(y_true)} "
            f"acc={metrics[label]['accuracy']:.4f} "
            f"p={p:.4f} r={r:.4f} f1={f:.4f}"
        )

    return metrics


# ============================================================
# RUNNER
# ============================================================

def run_model(model: str, client: OpenAI, data: List[Dict], truth: List[Dict]):
    print(f"\n[run_model] ===== {model} =====")
    start = time.time()

    # Run unified classification
    results, total_in, total_out = classify_unified(client, data, model)
    
    # Save results
    out_path = f"{OUTPUT_DIR}/classifications_{model}_unified.json"
    save_json(results, out_path)
    
    # Evaluate
    metrics = evaluate_classifications(results, truth)
    
    # Calculate cost
    cost = (
        (total_in / 1e6) * PRICING[model]["input"]
        + (total_out / 1e6) * PRICING[model]["output"]
    )

    elapsed = time.time() - start
    print(
        f"[run_model] DONE model={model} "
        f"time={elapsed:.1f}s cost=${cost:.4f}"
    )

    return metrics, cost


# ============================================================
# MAIN
# ============================================================

def main():
    print("[main] Starting UNIFIED CLASSIFICATION run")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load thread data
    thread_map = load_thread_data(THREAD_JSON_PATH)
    
    # Load ground truth and enrich with thread context
    data, truth = load_data_from_csv_with_threads(INPUT_CSV_PATH, thread_map)

    print(f"[main] Loaded {len(data)} tweets for classification")
    print(f"[main] Ground truth contains {len(truth)} records")

    # Run classification for each model
    all_results = {}
    for model in MODELS_TO_RUN:
        metrics, cost = run_model(model, client, data, truth)
        
        hs_f1 = metrics.get('hate_speech', {}).get('f1', 0)
        cs_f1 = metrics.get('counterspeech', {}).get('f1', 0)
        
        all_results[model] = {
            'metrics': metrics,
            'cost': cost,
            'hs_f1': hs_f1,
            'cs_f1': cs_f1
        }
        
        print(
            f"[main] SUMMARY {model}: "
            f"HS F1={hs_f1:.4f} "
            f"CS F1={cs_f1:.4f} "
            f"cost=${cost:.4f}"
        )

    # Save summary
    summary_path = f"{OUTPUT_DIR}/summary.json"
    save_json(all_results, summary_path)

    print("[main] All models complete - results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
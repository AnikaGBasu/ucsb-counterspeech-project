"""
GPT-based Hate Speech + Counterspeech Classification
with FULL THREAD CONTEXT and Rule-Based Chain-of-Thought Summaries
UPDATED: Improved prompt and robust JSON error handling
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
OUTPUT_DIR = "results_thread_context_improved"

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

# Add near your imports (top of file)
from datetime import datetime

try:
    from dateutil import parser as date_parser  # python-dateutil
except Exception:
    date_parser = None


def _parse_date_safe(date_str: str) -> Optional[datetime]:
    """
    Best-effort parse. Returns None if missing/unparseable.
    Works with ISO-ish strings and many natural date formats if python-dateutil is available.
    """
    if not date_str or date_str == "unknown":
        return None
    if date_parser is None:
        return None
    try:
        return date_parser.parse(date_str)
    except Exception:
        return None


def _flatten_thread(thread: Dict) -> List[Dict]:
    """
    Flatten thread into nodes with structural metadata.
    Each node: tweet_id, username, date, raw_content, depth, parent_id, kind, order_index
    """
    nodes: List[Dict] = []
    order = 0

    # Main post
    post = thread.get("post", {}) or {}
    post_id = post.get("tweet_id")
    nodes.append({
        "tweet_id": post_id,
        "username": post.get("username", "unknown"),
        "date": post.get("date", "unknown"),
        "raw_content": post.get("raw_content", ""),
        "depth": 0,
        "parent_id": None,
        "kind": "post",
        "order_index": order
    })
    order += 1

    def walk(replies: List[Dict], depth: int, parent_id: Optional[int]):
        nonlocal order
        for r in replies or []:
            rid = r.get("tweet_id")
            nodes.append({
                "tweet_id": rid,
                "username": r.get("username", "unknown"),
                "date": r.get("date", "unknown"),
                "raw_content": r.get("raw_content", ""),
                "depth": depth,
                "parent_id": parent_id,
                "kind": "reply",
                "order_index": order
            })
            order += 1
            nested = r.get("nested_replies", []) or []
            if nested:
                walk(nested, depth + 1, rid)

    replies = thread.get("replies", []) or []
    walk(replies, depth=0, parent_id=post_id)
    return nodes


def build_thread_context_string(thread: Dict, target_tweet_id: int) -> str:
    """
    Build a readable string representation of the thread for context,
    with an explicit split into context BEFORE and AFTER the target.

    BEFORE TARGET:
      - eligible for the counterspeech hate-speech gate

    AFTER TARGET:
      - provided for disambiguation only
      - NOT eligible for the gate
    """
    nodes = _flatten_thread(thread)

    # Locate target node
    target_node = None
    for n in nodes:
        if n.get("tweet_id") == target_tweet_id:
            target_node = n
            break

    # If target not found, fall back to old behavior: show full thread without split
    if target_node is None:
        lines = ["=== THREAD CONTEXT ===", "(warning: target not found in thread)"]
        for n in nodes:
            indent = "  " * int(n["depth"])
            if n["kind"] == "post":
                prefix = "Original Post:"
            else:
                prefix = f"{indent}Reply to {n['parent_id']}:"
            lines.append(f"\n{prefix}")
            lines.append(f"{indent}ID: {n['tweet_id']}")
            lines.append(f"{indent}User: @{n['username']}")
            lines.append(f"{indent}Date: {n['date']}")
            lines.append(f"{indent}Text: {n['raw_content']}")
        lines.append("\n=== END THREAD CONTEXT ===")
        return "\n".join(lines)

    # Parse dates (best-effort)
    target_dt = _parse_date_safe(target_node.get("date", "unknown"))
    parsed_any = False
    for n in nodes:
        n["_dt"] = _parse_date_safe(n.get("date", "unknown"))
        if n["_dt"] is not None:
            parsed_any = True

    # Decide BEFORE/AFTER sets
    before_nodes = []
    after_nodes = []

    if target_dt is not None and parsed_any:
        for n in nodes:
            if n["tweet_id"] == target_tweet_id:
                continue
            ndt = n.get("_dt")
            # If a node has no parseable date, treat as "unknown timing": keep it in BEFORE
            # (This is conservative for gating; you can flip this if you prefer.)
            if ndt is None or ndt <= target_dt:
                before_nodes.append(n)
            else:
                after_nodes.append(n)

        # Sort for readability: chronological when possible, stable by original order
        before_nodes.sort(key=lambda x: (x["_dt"] or datetime.min, x["order_index"]))
        after_nodes.sort(key=lambda x: (x["_dt"] or datetime.max, x["order_index"]))
    else:
        # Fallback: use structural dump order and split around the target's position in that order
        target_idx = next((i for i, n in enumerate(nodes) if n["tweet_id"] == target_tweet_id), None)
        for i, n in enumerate(nodes):
            if n["tweet_id"] == target_tweet_id:
                continue
            if target_idx is not None and i < target_idx:
                before_nodes.append(n)
            else:
                after_nodes.append(n)

    def render_node(n: Dict, is_target: bool = False) -> List[str]:
        indent = "  " * int(n["depth"])
        if is_target:
            prefix = ">>> TARGET TWEET >>>"
        else:
            if n["kind"] == "post":
                prefix = "Original Post:"
            else:
                prefix = f"{indent}Reply to {n['parent_id']}:"

        out = [f"\n{prefix}",
               f"{indent}ID: {n['tweet_id']}",
               f"{indent}User: @{n['username']}",
               f"{indent}Date: {n.get('date', 'unknown')}",
               f"{indent}Text: {n.get('raw_content', '')}"]
        return out

    lines = []
    lines.append("=== THREAD CONTEXT (PARTITIONED) ===")
    lines.append("=== CONTEXT BEFORE TARGET (ELIGIBLE FOR GATE) ===")
    for n in before_nodes:
        lines.extend(render_node(n, is_target=False))

    lines.append("\n=== TARGET TWEET ===")
    lines.extend(render_node(target_node, is_target=True))

    lines.append("\n=== CONTEXT AFTER TARGET (NOT ELIGIBLE FOR GATE) ===")
    for n in after_nodes:
        lines.extend(render_node(n, is_target=False))

    lines.append("\n=== END THREAD CONTEXT ===")
    return "\n".join(lines)



# ============================================================
# IMPROVED PROMPTS
# ============================================================

hate_speech_cot_prompt = """You are an expert content moderator specializing in hate speech detection.

Your task: Classify whether a tweet contains hate speech, determine its type and target, then explain your reasoning.

==================================================
INPUT FORMAT
==================================================

You will receive JSON with:
- tweet_id: numeric identifier of the TARGET tweet to classify
- raw_content: the TARGET tweet text to classify
- parent_raw_content: (optional) immediate parent if this is a reply
- thread_context: FULL conversation thread showing all messages leading up to and around the target tweet

⚠️ CRITICAL: You are classifying ONLY the tweet marked as ">>> TARGET TWEET >>>". The thread context is provided ONLY to help you understand the conversational context - NOT as evidence for classification.

==================================================
CORE PRINCIPLE
==================================================

🎯 THE TWEET MUST BE HATEFUL ON ITS OWN

Thread context helps you understand:
- What the conversation is about
- Whether earlier messages contained hate
- What the target tweet is responding to

Thread context does NOT:
- Make a non-hateful tweet hateful
- Turn opposition into endorsement
- Upgrade venting to hate speech

DECISION RULE: Would this tweet be hateful if you saw it standalone? If no, it's probably not hate speech.

==================================================
STEP 1: DOES THIS CONTAIN HATE SPEECH?
==================================================

Hate speech is any content that targets a protected class or identity group (or a person as a proxy for that group) with:
- dehumanization (e.g., vermin, animals, disease)
- calls for exclusion, segregation, or forced removal
- endorsement of violence, genocide, lynching, or harm
- slurs or epithets used to attack a group
- claims of inherent inferiority, criminality, or contamination
- conspiratorial scapegoating framed as collective guilt (esp. "they control…" used to justify hatred)
- praising extremist violence against groups
- denying group existence or legitimacy as erasure

Protected classes include (non-exhaustive): race, ethnicity, nationality, religion, caste, disability, sexual orientation, gender identity, sex, immigrant status (when used as an identity group), and similar.

CRITICAL DISTINCTION - What is NOT hate speech:
❌ Political criticism ("I disagree with this policy")
❌ Personal insults not based on identity ("John is an idiot")
❌ Venting without dehumanization ("ugh men are annoying today")
❌ Discussing/condemning hate ("using slurs is wrong")
❌ Calling out bigotry ("that's racist")
❌ Attacking the hate speaker ("you're disgusting for saying that")
❌ Defensive responses ("shut up yourself, I've been voting for 50 years")

==================================================
KEY REMINDERS - READ CAREFULLY
==================================================

1. ⚠️ THREAD CONTEXT IS FOR UNDERSTANDING, NOT EVIDENCE
   - Review the full thread to understand conversational flow
   - Use it to see what the target tweet is responding to
   - Do NOT use "appears in hateful thread" as classification evidence
   - Do NOT classify based on "alignment with earlier messages"

2. ⚠️ MOST REPLIES TO HATE ARE OPPOSITION, NOT ENDORSEMENT
   - Default assumption: replies oppose hate unless proven otherwise
   - Hostility toward the HATER = opposition
   - Only classify as endorsement with explicit agreement

3. ⚠️ Hostility toward the HATER ≠ Endorsement
   - Insults, threats, or consequences aimed at the hate speaker = OPPOSITION
   - Even harsh or profane attacks on the hater are NOT hate speech
   - Examples: "you're disgusting", "hope you get fired", "you're a bigot"

4. ⚠️ VENTING IS NOT HATE SPEECH (Even in a Critical Thread)
   - Frustrated expressions like "ugh men are annoying" are venting
   - These are NOT hate EVEN IF the thread contains other criticism
   - Only becomes hate with dehumanization or calls for harm
   - Mere frustration or dislike ≠ hate speech

5. ⚠️ Attacks on Protected Characteristics = Hate
   - Language targeting identity-linked traits (e.g., gendered appearance,
     disability, age, ethnicity) counts as hate speech
   - Framing it as "individual criticism" does NOT excuse identity-based attacks
   - Ask: would this language be used if the person were a different identity?

6. ⚠️ Denying Group Existence or Legitimacy = Hate
   - Claims that a people, identity, or group "doesn't exist" = erasure
   - Example: "There is no country called Palestine" = erasure/hate
   - Not political discourse when it denies humanity, rights, or identity
   - Thread context CAN help identify this (political debate vs. erasure)

7. ⚠️ Defensive Responses = Opposition
   - "Shut up yourself", "some of us...", "fight me" = defensive
   - These defend against criticism, not attack a group
   - Even if aggressive, they're opposing the speaker, not endorsing hate

8. ⚠️ Absurdist Mockery = Opposition
   - Sarcasm or exaggeration that makes the hateful claim look ridiculous
     counts as opposition, not endorsement

==================================================
STEP 2: CHECK FOR ENDORSEMENT (REPLIES ONLY)
==================================================

⚠️ CRITICAL: Most replies to hate speech are OPPOSITION, not endorsement!

If the target tweet is a REPLY (check thread context), apply this decision tree:

┌─ Does the thread contain hate speech earlier?
│  └─ NO → This cannot be endorsement → hate_speech = 0
│  └─ YES → Continue to next question
│
└─ Does the TARGET TWEET oppose or endorse the earlier hate?

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   OPPOSITION (= NOT hate speech)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   
   ✓ Questions the hater: "why would you say that?", "are you serious?"
   
   ✓ Hostile toward THE HATER (not the targeted group):
     - "you're a bigot", "disgusting behavior"
     - "hope you get fired", "you'll face consequences"
     - "dumb fuck for saying that", "you're trash"
     → Being hostile to the HATER = OPPOSITION, not endorsement!
   
   ✓ Warns of consequences: 
     - "this will have repercussions", "expect consequences"
     - "you got fired", "hope you lose your job"
   
   ✓ Challenges logic: "that's completely false", "your reasoning is flawed"
   
   ✓ Dismisses claim: "nobody believes this", "nonsense", "this is ridiculous"
   
   ✓ Defensive responses:
     - "shut up yourself", "some of us...", "fight me"
     - Asserting individual action against stereotypes
   
   ✓ Presents counter-facts: 
     - "actually, [different perspective]"
     - "we did X while you Y" (defensive response)
   
   ✓ Absurdist mockery of hate:
     - Ridiculous exaggerations that mock the premise
     - Sarcastic escalations that expose the absurdity
     → If it makes the hate look stupid = OPPOSITION
   
   ✓ Discusses critically: "prejudice is harmful", "discrimination is wrong"
   
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ENDORSEMENT (= IS hate speech):
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   
   ✗ EXPLICIT agreement about the hate itself: 
     - "so true!", "facts", "exactly", "correct", "this!", "agree"
     - Must be agreeing WITH THE HATE, not just the speaker
   
   ✗ Amplifies hate toward THE SAME TARGET: 
     - "yes, and [targeted group] also [more hate]"
   
   ✗ Celebrates THE HATE: 
     - "finally someone said it about [group]"
     - "thank you for this truth about [group]"
   
   ✗ Normalizes by treating extreme hate as reasonable

⚠️ CRITICAL DISTINCTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hostile/attacking language directed AT THE HATER = OPPOSITION
Hostile/attacking language directed AT THE TARGET = ENDORSEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Examples:
- "You're a piece of shit for saying that" → OPPOSITION (attacking hater)
- "They're pieces of shit" (about target) → ENDORSEMENT (attacking target)

RULE: If reply OPPOSES (even hostilely) → hate_speech = 0
RULE: If reply ENDORSES the hate itself → hate_speech = 1 (same target as earlier)
RULE: When unclear, check for EXPLICIT agreement with the hateful claim

==================================================
STEP 3: IF HATE SPEECH, CLASSIFY TYPE
==================================================

EXPLICIT hate speech:
- Direct slurs aimed at a group ("f*ggot", "n*gger", "retard")
- Clear dehumanization ("they're animals", "vermin", "disease")
- Direct calls for harm ("kill all X", "deport them all")
- Denial of group existence/statehood when clearly intended to erase
- No plausible alternative interpretation

IMPLICIT hate speech:
- Coded language or dog whistles ("urban youth", "globalists")
- Stereotypes presented as fact ("all X are criminals")
- Dehumanization through "jokes" or sarcasm
- Body-shaming with gendered patterns (attacking appearance based on gender)
- Age-based generalizations that demean
- Denial of group legitimacy (may require thread context to distinguish from political discourse)
- Requires context to identify as hateful

==================================================
STEP 4: IF HATE SPEECH, IDENTIFY TARGET
==================================================

Choose the PRIMARY target from this list:
Women | Men | Black | White | Jewish | Muslim | Middle Eastern | Christian |
Asian | Latino/Hispanic | LGBTQ+ | Immigrants/refugees | Disabled people | 
Elderly | Indigenous | Palestinian | Other | Unspecified

Target selection rules:
- Choose the group being attacked (not mentioned tangentially)
- If endorsing earlier hate in thread, use the SAME target
- If multiple groups targeted, pick the primary one
- Age-based attacks → use "Elderly" or relevant age category
- If unclear, use "Unspecified"

==================================================
SANITY CHECK - BEFORE FINALIZING
==================================================

If you classified as hate_speech = 1, ask yourself:

1. ❓ Would this tweet be hateful if I saw it standalone without thread context?
   → If NO, you may be wrongly using thread as evidence

2. ❓ Am I using "appears in hateful thread" or "aligns with earlier messages" as reasoning?
   → If YES, this is WRONG - the tweet must be hateful itself

3. ❓ Is this person attacking the HATER or the TARGETED GROUP?
   → If attacking the hater, this is OPPOSITION, not hate

4. ❓ Does this show explicit agreement ("so true!", "exactly!") or just hostility?
   → If just hostile toward hater, this is OPPOSITION, not hate

5. ❓ Is this venting frustration without dehumanization or calls for harm?
   → If YES, this falls under venting exception, not hate

6. ❓ Is this a defensive response ("shut up yourself", "some of us...")?
   → If YES, this is defending against criticism, not hate

If you answered unfavorably to any of these, RECONSIDER your classification.

==================================================
REASONING GUIDELINES
==================================================

Your reasoning_summary must:
1. Reference thread context if relevant (e.g., "Earlier in thread, X said...")
2. State whether earlier messages contain hate speech (if applicable)
3. Explain what THE TARGET TWEET ITSELF contains
4. Reference which rule/category applies
5. Be concise (2-4 sentences)

FORBIDDEN REASONING PATTERNS:
❌ "aligns with earlier messages in the thread"
❌ "in the context of the thread, this becomes hate"
❌ "combined with earlier statements, this is hate"
❌ "the thread contains hate, therefore this tweet..."

REQUIRED REASONING PATTERNS:
✓ "The target tweet itself contains [specific element]"
✓ "Independent of thread context, this attacks [group] by [mechanism]"
✓ "The tweet shows explicit agreement with earlier hate through [quote]"
✓ "While the thread contains hate, this tweet opposes it by [action]"
✓ "This denies the existence/legitimacy of [group], which constitutes erasure"
✓ "This uses body-shaming language in a gendered pattern targeting [group]"

Do NOT:
- Quote the original text verbatim
- Repeat the full text
- Include markdown formatting

==================================================
OUTPUT FORMAT
==================================================

Return ONLY this JSON (no other text):

[
  {
    "id": <tweet_id>,
    "hate_speech": 0 or 1,
    "hate_speech_type": "explicit" | "implicit" | null,
    "identity_targeted": "Women" | "Men" | ... | null,
    "reasoning_summary": "Brief explanation here referencing thread context if relevant"
  }
]

==================================================
DECISION FLOWCHART
==================================================

START
  │
  ├─→ Review THREAD CONTEXT (for understanding only)
  │
  ├─→ Would this tweet be hateful standalone?
  │     NO → Probably hate_speech = 0 (unless subtle erasure/pattern)
  │     YES or UNCLEAR ↓
  │
  ├─→ Is target tweet a reply?
  │     NO → Evaluate tweet itself for hate
  │     YES ↓
  │
  ├─→ Does thread contain earlier hate speech?
  │     NO → Cannot be endorsement, evaluate tweet itself
  │     YES ↓
  │
  ├─→ Is the target tweet ATTACKING THE HATER?
  │     YES → This is OPPOSITION → hate_speech = 0, STOP
  │     NO ↓
  │
  ├─→ Is this venting/defensive language without dehumanization?
  │     YES → Venting exception → hate_speech = 0, STOP
  │     NO ↓
  │
  ├─→ Does target show EXPLICIT AGREEMENT with the hate itself?
  │     YES → hate_speech = 1, use earlier target, STOP
  │     NO ↓
  │
  └─→ Does the tweet ITSELF attack/demean a protected group?
        → Evaluate for dehumanization, slurs, erasure, gendered body-shaming
        → Check edge cases above
        → Classify type & target if hate
        → If borderline, err toward hate_speech = 0
"""

hate_speech_cot_examples = [
    # Example 1: Clear endorsement with emoji
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1001,
  "raw_content": "So true 👍👍",
  "parent_raw_content": "All [group] are lazy and ruin our country.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @hater\\nText: All [group] are lazy and ruin our country.\\n\\n>>> TARGET TWEET >>>\\nID: 1001\\nUser: @endorser\\nText: So true 👍👍\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1001,
    "hate_speech": 1,
    "hate_speech_type": "implicit",
    "identity_targeted": "Immigrants/refugees",
    "reasoning_summary": "Earlier in thread, a hateful stereotyping statement was made about a protected group. The target tweet shows explicit agreement ('so true') with affirmation emoji, endorsing the hate speech."
  }
]"""
    },
    
    # Example 2: Opposition via challenge
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1002,
  "raw_content": "That's completely false and you know it",
  "parent_raw_content": "Women are too emotional to lead companies.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @sexist\\nText: Women are too emotional to lead companies.\\n\\n>>> TARGET TWEET >>>\\nID: 1002\\nUser: @challenger\\nText: That's completely false and you know it\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1002,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "The thread shows an earlier message using gender stereotypes. The target tweet directly challenges this claim rather than endorsing it, making this opposition, not hate speech."
  }
]"""
    },
    
    # Example 3: Venting is NOT hate
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1003,
  "raw_content": "God I hate men today #annoyed",
  "parent_raw_content": "men be like 'ew i hate gay people so gross ew don't look at me' then be like 'mm yummy bi girls kiss each other so hot mm sexy'",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @critic\\nText: men be like 'ew i hate gay people so gross ew don't look at me' then be like 'mm yummy bi girls kiss each other so hot mm sexy'\\n\\nReply 1:\\nID: 1001\\nText: For real, so frustrating\\n\\n>>> TARGET TWEET >>>\\nID: 1003\\nUser: @venter\\nText: God I hate men today #annoyed\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1003,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "This is venting frustration ('I hate men today', '#annoyed') without dehumanization or calls for harm. Even though the thread contains criticism of men's behavior, the target tweet itself falls under the venting exception and is not hate speech."
  }
]"""
    },
    
    # Example 4: Hostile language toward hater = opposition
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1004,
  "raw_content": "Dude you got fired! 😂 I hope you don't get hired anywhere, and you and your family suffer for your stupid actions. Best of luck dumb fuck! 😆",
  "parent_raw_content": "You know what I hate Niggers.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @hater\\nText: You know what I hate Niggers.\\n\\n>>> TARGET TWEET >>>\\nID: 1004\\nUser: @opposer\\nText: Dude you got fired! 😂 I hope you don't get hired anywhere, and you and your family suffer for your stupid actions. Best of luck dumb fuck! 😆\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1004,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "The parent contains explicit hate speech with a racial slur. The target tweet uses hostile language and warnings of consequences directed at the hate speaker ('you got fired', 'dumb fuck'), not the targeted group. This is opposition, not hate speech."
  }
]"""
    },
    
    # Example 5: Defensive response = opposition
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1005,
  "raw_content": "Shut up yourself. Some of us Mayo butts have voted democrat in every election since we were 18. I am 70. Fight me. Damn.",
  "parent_raw_content": "Shut up. I'm sick of white people who expect Democrats to do the job they themselves cannot or will not do. These are YOUR friends, family, neighbors and coworkers falling for this stupid RW propaganda. Maybe get off your mayonnaise asses, confront them and stop being cowards",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @frustrated\\nText: Shut up. I'm sick of white people who expect Democrats to do the job they themselves cannot or will not do. These are YOUR friends, family, neighbors and coworkers falling for this stupid RW propaganda. Maybe get off your mayonnaise asses, confront them and stop being cowards\\n\\n>>> TARGET TWEET >>>\\nID: 1005\\nUser: @defender\\nText: Shut up yourself. Some of us Mayo butts have voted democrat in every election since we were 18. I am 70. Fight me. Damn.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1005,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "The parent expresses frustration with white people's political behavior. The target tweet is a defensive response ('Shut up yourself', 'some of us', 'I am 70') asserting individual political action. This defends against criticism, not attacking a group. Opposition, not hate speech."
  }
]"""
    },
    
    # Example 6: Political erasure (thread context helps)
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1006,
  "raw_content": "Palestinian Territories * There is no country under the name palestine. #educationiskey",
  "parent_raw_content": "You're exhausting, I *am* advocating for peace, hamas doesn't stand for all of Palestine, it's not the only political party. Go off if you want. Have fun or whatever 🤷🏻‍♀️😂",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @debater\\nText: You're exhausting, I *am* advocating for peace, hamas doesn't stand for all of Palestine, it's not the only political party. Go off if you want. Have fun or whatever 🤷🏻‍♀️😂\\n\\n>>> TARGET TWEET >>>\\nID: 1006\\nUser: @denier\\nText: Palestinian Territories * There is no country under the name palestine. #educationiskey\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1006,
    "hate_speech": 1,
    "hate_speech_type": "explicit",
    "identity_targeted": "Middle Eastern",
    "reasoning_summary": "The target tweet denies the existence and legitimacy of Palestine as a state ('There is no country under the name palestine'), which constitutes erasure of Palestinian identity. This denial of group existence is hate speech regardless of political perspectives."
  }
]"""
    },
    
    # Example 7: Gendered body-shaming
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1007,
  "raw_content": "You know, she's... tubby. Where I have a problem is in the 'Image of Ivankaronga' (sleek, shiny, in control), versus the 'Truth of Ivankaronga', which is that she's tubby and wants ice cream and may really like girls -- which is fine, but she should just say so. There we are.",
  "parent_raw_content": "This bitch still talking when I already told her to shut the fuck up.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @hostile\\nText: This bitch still talking when I already told her to shut the fuck up.\\n\\n>>> TARGET TWEET >>>\\nID: 1007\\nUser: @bodyshamer\\nText: You know, she's... tubby. Where I have a problem is in the 'Image of Ivankaronga' (sleek, shiny, in control), versus the 'Truth of Ivankaronga', which is that she's tubby and wants ice cream and may really like girls -- which is fine, but she should just say so. There we are.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1007,
    "hate_speech": 1,
    "hate_speech_type": "explicit",
    "identity_targeted": "Women",
    "reasoning_summary": "The target tweet uses body-shaming language ('tubby', references to food cravings) in a gendered pattern targeting a woman. The thread context shows this is part of broader hostility toward women. The body-shaming itself attacks based on gender identity, making this hate speech."
  }
]"""
    },
    
    # Example 8: Absurdist mockery
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1008,
  "raw_content": "yeah they buy all the uncrustables and kid cuisine and stealing all the good titty implants",
  "parent_raw_content": "i HATE transgender people. THEY FREAKING SUCK!!!!!!!!!!!!!!!!!!!!!!!!!",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @hater\\nText: i HATE transgender people. THEY FREAKING SUCK!!!!!!!!!!!!!!!!!!!!!!!!!\\n\\n>>> TARGET TWEET >>>\\nID: 1008\\nUser: @mocker\\nText: yeah they buy all the uncrustables and kid cuisine and stealing all the good titty implants\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1008,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "The parent contains explicit hate speech against transgender people. The target tweet uses absurdist humor (claiming they buy all the Uncrustables and steal implants) to mock the hateful premise and make it look ridiculous. This is opposition through satire, not endorsement or hate."
  }
]"""
    },
    
    # Example 9: Venting about behavior
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1009,
  "raw_content": "The things I see on here, God I hate men #CringeWorthy 😖",
  "parent_raw_content": "",
  "thread_context": "=== THREAD CONTEXT ===\\n\\n>>> TARGET TWEET >>>\\nID: 1009\\nUser: @venter\\nText: The things I see on here, God I hate men #CringeWorthy 😖\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1009,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "This expresses frustration with observed behavior ('The things I see', '#CringeWorthy') without dehumanization or calls for harm. Falls under the venting exception - expressing annoyance is not hate speech."
  }
]"""
    },
    
    # Example 10: Warning of consequences
    {
        "role": "user",
        "content": '''{
  "tweet_id": 1010,
  "raw_content": "You might want to remove 'teacher' from your profile... since you know... your done with that.",
  "parent_raw_content": "You know what I hate Niggers.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 1000\\nUser: @teacher\\nText: You know what I hate Niggers.\\n\\n>>> TARGET TWEET >>>\\nID: 1010\\nUser: @warner\\nText: You might want to remove 'teacher' from your profile... since you know... your done with that.\\n\\n=== END THREAD CONTEXT ==="
}'''
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 1010,
    "hate_speech": 0,
    "hate_speech_type": null,
    "identity_targeted": null,
    "reasoning_summary": "The parent contains explicit hate speech with a racial slur. The target tweet warns of professional consequences for the hate speaker ('your done with that [teaching job]'). Warning of consequences for hateful behavior is opposition, not hate speech."
  }
]"""
    }
]
counterspeech_cot_prompt = """ROLE: You are a careful content moderator and dataset labeler.

TASK:
Label ONE reply (the TARGET tweet) for COUNTERSPEECH.
Input JSON provides:
- tweet_id (TARGET id)
- raw_content (TARGET text)
- parent_raw_content (immediate parent text)
- thread_context (FULL thread string; includes explicit section headers and one marked TARGET)

CRITICAL:
- Classify ONLY the TARGET tweet marked as ">>> TARGET TWEET >>>".
- Thread context is evidence. Your decision MUST be supported by citing specific in-thread text.

------------------------------------------------------------
STEP 0 — IDENTIFY ELIGIBLE CONTEXT (NEW PARTITION RULE)
------------------------------------------------------------
The thread_context is partitioned into explicit sections:

- "=== CONTEXT BEFORE TARGET (ELIGIBLE FOR GATE) ==="
- "=== TARGET TWEET ===" (contains the ">>> TARGET TWEET >>>" marker)
- "=== CONTEXT AFTER TARGET (NOT ELIGIBLE FOR GATE) ==="

ELIGIBLE FOR THE GATE (hate-speech anchor evidence):
- Use ONLY the "CONTEXT BEFORE TARGET (ELIGIBLE FOR GATE)" section to decide whether a hate-speech anchor exists.
- Do NOT use anything in the AFTER section to claim the gate passes.
  (You may read AFTER only to disambiguate what the target meant, but it MUST NOT affect whether the gate passes.)

Fallback:
- If thread_context is empty/missing OR the BEFORE section is missing, use ONLY parent_raw_content for the gate.

------------------------------------------------------------
STEP 1 — HATE-SPEECH ANCHOR (GATE)
------------------------------------------------------------
Counterspeech can be 1 ONLY IF there exists at least one HATE-SPEECH ANCHOR in eligible context.

Define a HATE-SPEECH ANCHOR as content that attacks, demeans, dehumanizes, excludes, or incites harm
against an identity group (protected or identity-based group), OR endorses such hate.

IMPORTANT EXCLUSIONS (do NOT count as hate-speech anchors):
- self-hate / self-deprecation ("I hate myself")
- generic rudeness or personal insults not tied to identity ("idiot", "whore" used as a personal insult)
- non-identity political disagreement without group attack
- mere discussion OF hate speech without attacking a group (unless it repeats slurs as an attack)

QUOTED / REPORTED HATE:
- If eligible context contains hateful content as a quote/paraphrase/reported speech ("someone said 'X'"),
  it still COUNTS as a hate-speech anchor for gating.

EVIDENCE REQUIREMENT (MANDATORY):
If you claim an anchor exists, you MUST provide:
- the anchor tweet id (from thread_context), AND
- a short direct quote (<= 12 words) from that anchor showing group-directed hate, AND
- the identity group being targeted.

If you cannot provide all three pieces of evidence, then the gate FAILS.

If the gate FAILS:
- counterspeech = 0
- counterspeech_type = null
- dominant_counterspeech_type = null

------------------------------------------------------------
STEP 2 — DOES THE TARGET OPPOSE THE ANCHOR?
------------------------------------------------------------
If (and only if) a hate-speech anchor exists:

counterspeech = 1 if the TARGET tweet opposes the anchor’s hateful content in ANY of these ways:
- condemns / rejects / disapproves
- refutes or corrects
- challenges assumptions (including rhetorical questions)
- defends or humanizes the targeted group
- warns of consequences (reporting, banning, job loss, etc.)
- uses humor/satire to mock the hateful premise
- uses hostile language toward the hater (insults aimed at the speaker) WHILE opposing hate

MINIMAL COUNTERSPEECH SHORTCUTS (COUNT AS COUNTERSPEECH when clearly aimed at the hater or hate):
- direct call-outs: "what is wrong with you", "this is disgusting", "this is what you stand for?"
- direct hostile rebuke at the hateful speaker: "fuck you", "shut up", "you're a bigot"
- consequence threats: "reported", "banned", "blocked", "you’ll get fired"

ENDORSEMENT / SUPPORT OF COUNTERSPEECH:
The target tweet can be counterspeech even if it is not directly addressing the hateful claim, if it:
- explicitly endorses an earlier counterspeech reply that is opposing the anchor
  (e.g., "exactly", "this", "well said", "facts", "period") AND the endorsed message is itself anti-hate.

To avoid false positives, only count endorsement if you can:
- quote a short snippet from the endorsed counterspeech (<= 12 words), AND
- explain how that endorsed message opposes the hate anchor.

NOT COUNTERSPEECH (counterspeech = 0):
- agrees with the hate
- amplifies the hate or attacks the targeted group
- is unrelated / topic shift with no anti-hate function
- is only supportive of self-hate or non-identity negativity (not hate speech)

CRITICAL EXCLUSION:
If the TARGET tweet itself contains hate speech (attacks a group) OR endorses hate, counterspeech MUST be 0.

------------------------------------------------------------
COUNTERSPEECH CATEGORIES (if counterspeech=1)
------------------------------------------------------------
Choose ONE:
1) presenting facts
2) challenging logic
3) warning of consequences
4) affiliation
5) denouncing hate speech
6) humor
7) empathy/positive tone
8) hostile language

Set dominant_counterspeech_type = counterspeech_type.

------------------------------------------------------------
REASONING SUMMARY (CONCISE BUT SHOW LOGIC)
------------------------------------------------------------
You MAY quote short snippets.

reasoning_summary MUST include three parts in one short paragraph:
1) HATE_ANCHOR: (anchor_id) "anchor quote" -> targets <group>
2) TARGET_LINK: "target snippet" -> how it responds (oppose/endorse/ignore)
3) DECISION: counterspeech=<0/1>, type=<category or null>

Keep it <= 3 sentences.

------------------------------------------------------------
OUTPUT FORMAT (STRICT)
------------------------------------------------------------
Return a valid JSON array with exactly one object:

[
  {
    "id": <tweet_id number>,
    "counterspeech": 0 or 1,
    "counterspeech_type": "<one category above>" | null,
    "dominant_counterspeech_type": "<same as counterspeech_type>" | null,
    "reasoning_summary": "<<=3 sentences, includes HATE_ANCHOR, TARGET_LINK, DECISION>"
  }
]

Return ONLY the JSON array. No extra text.
"""


# Updated examples list with new edge cases
counterspeech_cot_examples = [
    # EXISTING EXAMPLES (keep all 10 from before)
    # Example 1: Direct facts
    {
        "role": "user",
        "content": """{
  "tweet_id": 2001,
  "raw_content": "That's just false. FBI reports show immigrant crime rates are lower than native-born citizens.",
  "parent_raw_content": "Immigrants commit all the crimes. It's obvious.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: Immigrants commit all the crimes.\\n\\n>>> TARGET TWEET >>>\\nID: 2001\\nText: That's just false. FBI reports show immigrant crime rates are lower.\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2001,
    "counterspeech": 1,
    "counterspeech_type": "presenting facts",
    "dominant_counterspeech_type": "presenting facts",
    "reasoning_summary": "The thread contains hate speech making blanket claims about immigrants. The target tweet provides factual evidence from FBI reports that directly refutes this claim. This counters the hate with concrete data."
  }
]"""
    },
    
    # Example 2: Subtle questioning
    {
        "role": "user",
        "content": """{
  "tweet_id": 2002,
  "raw_content": "wtf",
  "parent_raw_content": "Women are too emotional to lead companies.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: Women are too emotional to lead companies.\\n\\n>>> TARGET TWEET >>>\\nID: 2002\\nText: wtf\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2002,
    "counterspeech": 1,
    "counterspeech_type": "challenging logic",
    "dominant_counterspeech_type": "challenging logic",
    "reasoning_summary": "The thread contains hate speech using gender stereotypes about women. The target tweet expresses disbelief ('wtf'). This signals disagreement and challenges the premise through confusion."
  }
]"""
    },
    
    # Example 3: Agreement with criticism
    {
        "role": "user",
        "content": """{
  "tweet_id": 2003,
  "raw_content": "God I hate men today #annoyed",
  "parent_raw_content": "men be like 'ew i hate gay people so gross' then be like 'mm yummy bi girls kiss so hot'",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: men be like 'ew i hate gay people' then 'bi girls so hot'\\n\\n>>> TARGET TWEET >>>\\nID: 2003\\nText: God I hate men today #annoyed\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2003,
    "counterspeech": 1,
    "counterspeech_type": "hostile language",
    "dominant_counterspeech_type": "hostile language",
    "reasoning_summary": "The thread contains criticism of men's homophobic and fetishizing behavior toward LGBTQ+ people. The target tweet expresses frustration with this bigoted behavior through alignment with the criticism. This is counterspeech expressing anger at hateful conduct."
  }
]"""
    },
    
    # NEW EXAMPLE 4: Confusion implies disagreement (addresses ID: 1542205641787334656)
    {
        "role": "user",
        "content": """{
  "tweet_id": 2004,
  "raw_content": "tf fuck i do",
  "parent_raw_content": "Women shouldn't have the right to vote because they're too emotional.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: Women shouldn't have the right to vote because they're too emotional.\\n\\n>>> TARGET TWEET >>>\\nID: 2004\\nText: tf fuck i do\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2004,
    "counterspeech": 1,
    "counterspeech_type": "challenging logic",
    "dominant_counterspeech_type": "challenging logic",
    "reasoning_summary": "The thread contains hate speech against women using gender stereotypes to justify disenfranchisement. The target tweet expresses confusion and frustration ('tf fuck i do'). This signals disbelief and disagreement with the extreme statement, qualifying as counterspeech through questioning."
  }
]"""
    },
    
    # NEW EXAMPLE 5: Questions that imply skepticism (addresses ID: 1243403051009892352)
    {
        "role": "user",
        "content": """{
  "tweet_id": 2005,
  "raw_content": "What are you even talking about?",
  "parent_raw_content": "Jews control all the media and banks.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: Jews control all the media and banks.\\n\\n>>> TARGET TWEET >>>\\nID: 2005\\nText: What are you even talking about?\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2005,
    "counterspeech": 1,
    "counterspeech_type": "challenging logic",
    "dominant_counterspeech_type": "challenging logic",
    "reasoning_summary": "The thread contains antisemitic conspiracy theories. The target tweet asks a question ('What are you even talking about?') that signals confusion and skepticism. This challenges the premise by expressing disbelief in the conspiracy claim."
  }
]"""
    },
    
    # NEW EXAMPLE 6: Acknowledging + redirecting (addresses ID: 1393736168928284672)
    {
        "role": "user",
        "content": """{
  "tweet_id": 2006,
  "raw_content": "I get that you're frustrated, but maybe you just need to find the right person",
  "parent_raw_content": "All men are dumb",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: All men are dumb\\n\\n>>> TARGET TWEET >>>\\nID: 2006\\nText: I get that you're frustrated, but maybe you just need to find the right person\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2006,
    "counterspeech": 1,
    "counterspeech_type": "empathy/positive tone",
    "dominant_counterspeech_type": "empathy/positive tone",
    "reasoning_summary": "The thread contains negative generalizations about men. The target tweet validates the emotional response ('I get that you're frustrated') while redirecting toward individual variation ('find the right person'). This counters the blanket statement through empathetic redirection."
  }
]"""
    },
    
    # NEW EXAMPLE 7: Factual correction without condemnation (addresses ID: 1401948208188858384)
    {
        "role": "user",
        "content": """{
  "tweet_id": 2007,
  "raw_content": "Palestine has UN observer state status since 2012",
  "parent_raw_content": "There is no such thing as Palestine, it's not a real place",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: There is no such thing as Palestine, it's not a real place\\n\\n>>> TARGET TWEET >>>\\nID: 2007\\nText: Palestine has UN observer state status since 2012\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2007,
    "counterspeech": 1,
    "counterspeech_type": "presenting facts",
    "dominant_counterspeech_type": "presenting facts",
    "reasoning_summary": "The thread contains hate speech denying Palestinian identity and statehood. The target tweet provides factual information about Palestine's UN recognition status. This directly counters the denial with concrete evidence of international recognition."
  }
]"""
    },
    
    # Example 8: Hostile language
    {
        "role": "user",
        "content": """{
  "tweet_id": 2008,
  "raw_content": "Dude you got fired! Hope you suffer for your stupid actions. Best of luck dumb fuck! 😆",
  "parent_raw_content": "You know what I hate Niggers.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: You know what I hate Niggers.\\n\\n>>> TARGET TWEET >>>\\nID: 2008\\nText: Dude you got fired! Hope you suffer.\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2008,
    "counterspeech": 1,
    "counterspeech_type": "warning of consequences",
    "dominant_counterspeech_type": "warning of consequences",
    "reasoning_summary": "The thread contains explicit hate speech with racial slurs. The target tweet celebrates real-world consequences for the hater (job loss) and warns of future repercussions. This opposes the hate by highlighting social punishment for bigotry."
  }
]"""
    },
    
    # Example 9: Sarcasm/humor
    {
        "role": "user",
        "content": """{
  "tweet_id": 2009,
  "raw_content": "yeah they buy all the uncrustables and kid cuisine and stealing all the good titty implants",
  "parent_raw_content": "i HATE transgender people. THEY FREAKING SUCK!!!!",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: i HATE transgender people. THEY SUCK!\\n\\n>>> TARGET TWEET >>>\\nID: 2009\\nText: yeah they buy all the uncrustables...\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2009,
    "counterspeech": 1,
    "counterspeech_type": "humor",
    "dominant_counterspeech_type": "humor",
    "reasoning_summary": "The thread contains explicit hate speech against transgender people. The target tweet uses absurdist humor (claiming they buy all the Uncrustables and steal implants). This mocks the hate and makes it look ridiculous through satirical exaggeration."
  }
]"""
    },
    
    # Example 10: NOT counterspeech - endorsement
    {
        "role": "user",
        "content": """{
  "tweet_id": 2010,
  "raw_content": "So true 👍👍",
  "parent_raw_content": "All [group] are lazy and ruin our country.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: All [group] are lazy.\\n\\n>>> TARGET TWEET >>>\\nID: 2010\\nText: So true 👍👍\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2010,
    "counterspeech": 0,
    "counterspeech_type": null,
    "dominant_counterspeech_type": null,
    "reasoning_summary": "The thread contains hate speech making harmful generalizations about a group. The target tweet explicitly agrees with this hate ('So true' with affirmation emoji). This endorses rather than opposes the hate."
  }
]"""
    },
    
    # Example 11: NOT counterspeech - unrelated
    {
        "role": "user",
        "content": """{
  "tweet_id": 2011,
  "raw_content": "What time is the meeting tomorrow?",
  "parent_raw_content": "I can't stand immigrants, they're ruining everything.",
  "thread_context": "=== THREAD CONTEXT ===\\n\\nOriginal Post:\\nID: 2000\\nText: Immigrants are ruining everything.\\n\\n>>> TARGET TWEET >>>\\nID: 2011\\nText: What time is the meeting?\\n==="
}"""
    },
    {
        "role": "assistant",
        "content": """[
  {
    "id": 2011,
    "counterspeech": 0,
    "counterspeech_type": null,
    "dominant_counterspeech_type": null,
    "reasoning_summary": "The thread contains hate speech against immigrants. The target tweet asks about a meeting time without any connection to the hate. This is completely unrelated and ignores the hateful content entirely."
  }
]"""
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
        if isinstance(parsed, list) and len(parsed) == 1:
            return parsed[0]
        elif isinstance(parsed, dict):
            return parsed
        else:
            raise ValueError(f"Expected array with one object or dict, got: {type(parsed)}")
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
            
            # Extract reasoning_summary
            rs_match = re.search(r'"reasoning_summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if rs_match:
                result['reasoning_summary'] = rs_match.group(1).replace('\\"', '"')
            else:
                result['reasoning_summary'] = "Manual extraction - reasoning unavailable"
            
            print(f"[parse_json] Manual extraction successful")
            return result
            
        except Exception as manual_error:
            print(f"[parse_json] Manual extraction failed: {manual_error}")
            raise ValueError(f"Could not parse JSON. Original error: {e.msg}. Text: {text[:300]}")


# ============================================================
# IMPROVED API CALL WITH RETRY
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
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Try to parse the response
            try:
                parsed = parse_json_response(content, tweet_id)
                # If parsing succeeded, return the original content
                return content, prompt_tokens, completion_tokens
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(f"[call_gpt_api] JSON parse error on attempt {attempt}: {parse_error}")
                
                if attempt < MAX_RETRIES:
                    # Add a fix request to the conversation
                    print(f"[call_gpt_api] Asking model to fix JSON...")
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })
                    messages.append({
                        "role": "user",
                        "content": f"""Your JSON had an error: {str(parse_error)[:200]}

Please output ONLY valid JSON with no extra text. Use this exact format:

[
  {{
    "id": {tweet_id if tweet_id else "tweet_id"},
    "hate_speech": 0 or 1,
    "hate_speech_type": "explicit" or "implicit" or null,
    "identity_targeted": "Group" or null,
    "reasoning_summary": "explanation here"
  }}
]

Critical:
- Use double quotes for all strings
- No trailing commas
- Escape quotes in reasoning_summary with backslash
- Output ONLY the JSON array, nothing else"""
                    })
                    
                    time.sleep(REQUEST_DELAY)
                    continue
                else:
                    # Last attempt - use the parsed result from manual extraction
                    print(f"[call_gpt_api] Using manually extracted result")
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


def strip_reasoning(preds: List[Dict]) -> List[Dict]:
    cleaned = []
    for p in preds:
        q = p.copy()
        q.pop("counterspeech_reasoning_summary", None)
        q.pop("hate_speech_reasoning_summary", None)
        cleaned.append(q)
    return cleaned


# ============================================================
# CLASSIFICATION FUNCTIONS
# ============================================================

def classify_hate_speech_cot(client: OpenAI, data: List[Dict], model: str):
    print(f"[HS] Starting hate speech classification | model={model} | n={len(data)}")
    results, in_tok, out_tok = [], 0, 0

    for i, item in enumerate(data, start=1):
        tweet_id = item["ID"]
        
        payload = {
            "tweet_id": tweet_id,
            "raw_content": item["Text"],
            "parent_raw_content": item.get("Parent_Text", ""),
            "thread_context": item.get("Thread_Context", "")
        }

        print(f"[HS] ({i}/{len(data)}) tweet_id={tweet_id}")

        # Use the globally defined prompts
        messages = [{"role": "system", "content": hate_speech_cot_prompt}]
        messages.extend(hate_speech_cot_examples)
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

        try:
            text, it, ot = call_gpt_api(client, messages, model, tweet_id=tweet_id)
            parsed = parse_json_response(text, tweet_id)
            
            # Ensure id is correct
            parsed["id"] = tweet_id
            
            # Rename reasoning_summary
            parsed["hate_speech_reasoning_summary"] = parsed.pop("reasoning_summary", "")
            
            results.append(parsed)
            in_tok += it
            out_tok += ot
            
        except Exception as e:
            print(f"[HS] FATAL ERROR on tweet {tweet_id}: {e}")
            print(f"[HS] Adding placeholder result and continuing...")
            
            results.append({
                "id": tweet_id,
                "hate_speech": 0,
                "hate_speech_type": None,
                "identity_targeted": None,
                "hate_speech_reasoning_summary": f"ERROR: {str(e)[:100]}"
            })
        
        time.sleep(REQUEST_DELAY)

    print(f"[HS] Finished | total_in={in_tok} total_out={out_tok}")
    return results, in_tok, out_tok


def classify_counterspeech_cot(client: OpenAI, data: List[Dict], model: str):
    print(f"[CS] Starting counterspeech classification | model={model} | n={len(data)}")
    results, in_tok, out_tok = [], 0, 0

    for i, item in enumerate(data, start=1):
        tweet_id = item["ID"]
        
        payload = {
            "tweet_id": tweet_id,
            "raw_content": item["Text"],
            "parent_raw_content": item.get("Parent_Text", ""),
            "thread_context": item.get("Thread_Context", "")
        }

        print(f"[CS] ({i}/{len(data)}) tweet_id={tweet_id}")

        # Use the globally defined prompts
        messages = [{"role": "system", "content": counterspeech_cot_prompt}]
        messages.extend(counterspeech_cot_examples)
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

        try:
            text, it, ot = call_gpt_api(client, messages, model, tweet_id=tweet_id)
            parsed = parse_json_response(text, tweet_id)
            
            # Ensure id is correct
            parsed["id"] = tweet_id
            
            # Rename reasoning_summary
            parsed["counterspeech_reasoning_summary"] = parsed.pop("reasoning_summary", "")
            
            results.append(parsed)
            in_tok += it
            out_tok += ot
            
        except Exception as e:
            print(f"[CS] FATAL ERROR on tweet {tweet_id}: {e}")
            print(f"[CS] Adding placeholder result and continuing...")
            
            results.append({
                "id": tweet_id,
                "counterspeech": 0,
                "counterspeech_type": None,
                "dominant_counterspeech_type": None,
                "counterspeech_reasoning_summary": f"ERROR: {str(e)[:100]}"
            })
        
        time.sleep(REQUEST_DELAY)

    print(f"[CS] Finished | total_in={in_tok} total_out={out_tok}")
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

def run_model(model: str, client: OpenAI, data: List[Dict], truth: List[Dict], classify_both: bool = True):
    print(f"\n[run_model] ===== {model} =====")
    start = time.time()

    if classify_both:
        hs, hs_in, hs_out = classify_hate_speech_cot(client, data, model)
        cs, cs_in, cs_out = classify_counterspeech_cot(client, data, model)
        
        # Merge results
        hs_map = {x["id"]: x for x in hs}
        for c in cs:
            tid = c["id"]
            if tid in hs_map:
                hs_map[tid].update(c)
        
        combined = list(hs_map.values())
        
        out_path = f"{OUTPUT_DIR}/classifications_{model}_combined.json"
        save_json(combined, out_path)
        
        metrics = evaluate_classifications(strip_reasoning(combined), truth)
        
        cost = (
            ((hs_in + cs_in) / 1e6) * PRICING[model]["input"]
            + ((hs_out + cs_out) / 1e6) * PRICING[model]["output"]
        )
    else:
        cs, cs_in, cs_out = classify_counterspeech_cot(client, data, model)
        
        out_path = f"{OUTPUT_DIR}/classifications_{model}_cs_only.json"
        save_json(cs, out_path)
        
        metrics = evaluate_classifications(strip_reasoning(cs), truth)
        
        cost = (
            (cs_in / 1e6) * PRICING[model]["input"]
            + (cs_out / 1e6) * PRICING[model]["output"]
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
    print("[main] Starting IMPROVED THREAD CONTEXT run")
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

    # Run classification
    for model in MODELS_TO_RUN:
        metrics, cost = run_model(model, client, data, truth, classify_both=True)
        
        hs_f1 = metrics.get('hate_speech', {}).get('f1', 0)
        cs_f1 = metrics.get('counterspeech', {}).get('f1', 0)
        
        print(
            f"[main] SUMMARY {model}: "
            f"HS F1={hs_f1:.4f} "
            f"CS F1={cs_f1:.4f} "
            f"cost=${cost:.4f}"
        )

    print("[main] All models complete - results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
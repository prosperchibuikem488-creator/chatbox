import requests
import json
import time
import random
import re
import hashlib

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# CONFIG
# =========================================================

API_KEY = "[YOUR_API_KEY]"

MODEL = "mistral-small"

OUTPUT_FILE = "intent_dataset_v5.json"

INTENTS = [
    "venting",
    "seeking_advice",
    "greeting",
    "crisis",
    "general"
]

TARGET_PER_INTENT = 1400

BATCH_SIZE = 30

TEMPERATURE_RANGE = (0.8, 1.2)


# =========================================================
# SESSION (SSL + RETRY FIX)
# =========================================================

session = requests.Session()

retries = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retries)

session.mount("https://", adapter)


# =========================================================
# STYLE MODES
# =========================================================

STYLE_MODES = [
    "casual",
    "formal",
    "teen",
    "slang",
    "messy_typing",
    "short_fragment",
    "long_emotional",
    "sarcastic",
    "dry",
    "emotionally_numb",
    "internet_speak",
    "late_night_text",
    "social_media_style",
    "code_switching",
    "emoji_heavy"
]


# =========================================================
# DIVERSITY TOPICS
# =========================================================

TOPICS = [
    "school stress",
    "university pressure",
    "relationships",
    "family conflict",
    "loneliness",
    "social anxiety",
    "burnout",
    "self-esteem",
    "panic attacks",
    "sleep issues",
    "job stress",
    "friendship problems",
    "identity confusion",
    "future anxiety",
    "financial stress",
    "grief",
    "breakups",
    "isolation",
    "feeling emotionally numb",
    "anger issues"
]


# =========================================================
# PROMPTS
# =========================================================

def get_prompt(intent):

    style = random.choice(STYLE_MODES)
    topic = random.choice(TOPICS)

    base = f"""
You are generating HIGH-DIVERSITY training data for a mental health intent classifier.

STRICT RULES:
- Output ONLY valid JSON array
- Output format:
[
  "sample1",
  "sample2"
]

- NO markdown
- NO explanations
- NO numbering
- NO duplicate phrasing
- Every message must sound like a DIFFERENT real person
- Include realistic human texting behavior
- Include ambiguity SOMETIMES
- Avoid repetitive structures
- Avoid repetitive openings
- Use diverse sentence lengths
- Use slang occasionally
- Use typos occasionally
- Use emojis occasionally
- Use lowercase often
- Include subtle phrasing

CURRENT STYLE:
{style}

CURRENT TOPIC:
{topic}

Generate {BATCH_SIZE} samples.
"""

    # =====================================================
    # VENTING
    # =====================================================

    if intent == "venting":

        return base + """
INTENT: venting

Definition:
The user expresses emotions, frustration, sadness, exhaustion,
or emotional distress WITHOUT genuinely asking for guidance.

IMPORTANT:
- Emotional expression is the PRIMARY goal
- The user is NOT truly seeking solutions
- Some rhetorical questions are allowed
- Mild hopelessness is allowed
- Frustration is common

Include:
- stress
- loneliness
- emotional exhaustion
- sadness
- self criticism
- burnout
- passive hopelessness
- emotional dumping
- feeling overwhelmed

Examples:
- "everything feels exhausting lately"
- "im tired of pretending im okay"
- "another awful day lol"
- "why is life so draining"

HARD NEGATIVES:
These may LOOK like advice requests but are NOT:
- rhetorical questions
- vague uncertainty
- emotional statements ending with '?'

DO NOT INCLUDE:
- direct requests for help
- asking what to do
- seeking reassurance
- suicidal ideation
- severe existential despair
"""


    # =====================================================
    # SEEKING ADVICE
    # =====================================================

    elif intent == "seeking_advice":

        return base + """
INTENT: seeking_advice

Definition:
The user genuinely wants guidance, reassurance, coping strategies,
support, suggestions, or next steps.

IMPORTANT:
- The PRIMARY goal is getting help/advice
- Emotional distress may exist
- But the user is actively seeking input

Include:
- direct questions
- indirect requests
- uncertainty
- reassurance seeking
- coping strategy requests
- emotional confusion

Examples:
- "what should i do"
- "is this normal?"
- "how do people deal with this"
- "can i ask something"
- "should i talk to someone"
- "how do i stop overthinking"

HARD NEGATIVES:
Examples emotionally similar to venting
BUT clearly requesting guidance.

DO NOT INCLUDE:
- pure emotional dumping
- greeting-only messages
- severe suicidal hopelessness
"""


    # =====================================================
    # GREETING
    # =====================================================

    elif intent == "greeting":

        return base + """
INTENT: greeting

Definition:
The user is primarily starting or opening a conversation.

IMPORTANT:
- Conversation initiation is the MAIN purpose
- Emotional disclosure should be MINIMAL
- Tiny emotional hints are acceptable
- But emotional discussion should NOT dominate

Include:
- casual greetings
- awkward openings
- chatbot testing
- small intros
- checking if chatbot responds

Examples:
- "hey"
- "yo"
- "hello?"
- "sup"
- "hi there"
- "you there?"
- "good evening"

DO NOT INCLUDE:
- emotional dumping
- direct requests for help
- distress-heavy messages
- hopelessness
- severe emotional disclosure
"""


    # =====================================================
    # CRISIS
    # =====================================================

    elif intent == "crisis":

        return base + """
INTENT: crisis

Definition:
The user expresses SEVERE emotional distress,
possible danger,
emotional collapse,
or hopelessness.

IMPORTANT:
CRISIS MUST BE HIGH SEVERITY.

Include:
- suicidal ideation (non-graphic)
- severe hopelessness
- emotional collapse
- inability to continue
- existential despair
- feeling unsafe
- panic breakdown
- wanting existence to stop

Examples:
- "i dont think i can keep doing this"
- "everything feels pointless"
- "i want the pain to stop"
- "i genuinely dont want to be here anymore"
- "im scared of my own thoughts lately"

SUBTLE CRISIS EXAMPLES:
- indirect suicidal language
- complete emotional collapse
- severe despair

DO NOT INCLUDE:
- mild sadness
- burnout
- stress
- "im tired"
- "life sucks"
- "i feel lost"

Those belong to VENTING.
"""


    # =====================================================
    # GENERAL
    # =====================================================

    elif intent == "general":

        return base + """
INTENT: general

Definition:
The message does NOT clearly belong to:
- crisis
- venting
- seeking_advice
- greeting

This includes:
- neutral conversation
- random discussion
- chatbot exploration
- informational questions
- unclear intent
- vague chatter

Include:
- small talk
- random questions
- casual conversation
- boredom
- factual discussion
- internet chatter

Examples:
- "what can you do"
- "tell me a joke"
- "im bored"
- "what movie should i watch"
- "thats interesting"
- "how does therapy work"
- "okay cool"

DO NOT INCLUDE:
- emotional venting
- crisis signals
- direct requests for emotional help
"""


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    # Handle dict outputs from API
    if isinstance(text, dict):

        if "text" in text:
            text = text["text"]

        elif "message" in text:
            text = text["message"]

        else:
            return None

    if not isinstance(text, str):
        return None

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    text = text.replace("\n", " ")

    return text


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_for_similarity(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9 ]", "", text)

    return text


def hash_text(text):

    return hashlib.md5(
        normalize_for_similarity(text).encode()
    ).hexdigest()


# =========================================================
# DEDUPLICATION
# =========================================================

def deduplicate(dataset):

    seen = set()

    cleaned = []

    for item in dataset:

        h = hash_text(item["text"])

        if h not in seen:

            seen.add(h)

            cleaned.append(item)

    return cleaned


# =========================================================
# HARD CASES
# =========================================================

def add_hard_cases(dataset):

    hard_cases = [

        # =================================================
        # GREETING vs VENTING
        # =================================================

        ("hey... rough day today", "venting"),
        ("yo honestly everything sucks rn", "venting"),
        ("hey im exhausted lately", "venting"),
        ("hi, today has been awful", "venting"),
        ("yo im mentally drained", "venting"),

        # =================================================
        # VENTING vs SEEKING ADVICE
        # =================================================

        ("i dont know what to do anymore", "seeking_advice"),
        ("how do people survive feeling like this", "seeking_advice"),
        ("what should i do honestly", "seeking_advice"),
        ("is this normal?", "seeking_advice"),
        ("should i talk to someone", "seeking_advice"),
        ("im struggling lately what do i do", "seeking_advice"),

        ("im tired of everything lately", "venting"),
        ("another horrible day lol", "venting"),
        ("life just feels exhausting", "venting"),
        ("why is everything so draining", "venting"),

        # =================================================
        # VENTING vs CRISIS
        # =================================================

        ("i feel so lost right now", "venting"),
        ("im really tired of everything", "venting"),
        ("nothing feels enjoyable anymore", "venting"),
        ("i feel emotionally numb lately", "venting"),

        ("i genuinely dont wanna wake up tomorrow", "crisis"),
        ("i dont want to be here anymore", "crisis"),
        ("everything feels pointless now", "crisis"),
        ("i seriously cant keep going", "crisis"),
        ("i feel trapped in my own mind", "crisis"),
        ("i think im reaching my limit", "crisis"),

        # =================================================
        # GREETING
        # =================================================

        ("hello i guess", "greeting"),
        ("sup", "greeting"),
        ("yo", "greeting"),
        ("you there?", "greeting"),
        ("good evening", "greeting"),
        ("hello chatbot", "greeting"),

        # =================================================
        # GENERAL
        # =================================================

        ("what can you do", "general"),
        ("tell me a joke", "general"),
        ("im bored", "general"),
        ("thats interesting", "general"),
        ("okay cool", "general"),
        ("what movie should i watch", "general"),
        ("can you explain anxiety", "general"),
        ("do you like music", "general"),
        ("today was kinda normal", "general"),
        ("what time is it", "general"),
        ("im just chilling rn", "general"),
        ("random question", "general"),
        ("hmm alright", "general"),

        # =================================================
        # SLANG / TYPOS
        # =================================================

        ("bro im mentally cooked", "venting"),
        ("idk what to do tbh", "seeking_advice"),
        ("lowkey wanna disappear", "crisis"),
        ("yo wassup", "greeting"),

        # =================================================
        # CODE SWITCHING
        # =================================================

        ("idk man maisha inanichosha", "venting"),
        ("hola no sé qué hacer", "seeking_advice"),

        # =================================================
        # EMOJIS
        # =================================================

        ("lol i'm falling apart 😭", "venting"),
        ("i seriously cant keep going 💀", "crisis"),
        ("heyyy 👋", "greeting"),
        ("im bored 😭", "general"),
    ]

    for text, label in hard_cases:

        dataset.append({
            "text": text,
            "label": label
        })

    return dataset


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(text):

    try:
        return json.loads(text)

    except:

        match = re.search(r"\[.*\]", text, re.DOTALL)

        if match:

            try:
                return json.loads(match.group())

            except:
                return []

    return []


# =========================================================
# API CALL
# =========================================================

def call_mistral(prompt):

    url = "https://api.mistral.ai/v1/chat/completions"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": random.uniform(*TEMPERATURE_RANGE),
        "max_tokens": 1200
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:

        response = session.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:

            print("API ERROR:", response.text)

            return []

        text = response.json()["choices"][0]["message"]["content"]

        return extract_json(text)

    except requests.exceptions.SSLError as e:

        print("SSL ERROR:", e)

        time.sleep(10)

        return []

    except Exception as e:

        print("GENERAL ERROR:", e)

        time.sleep(5)

        return []


# =========================================================
# GENERATION LOOP
# =========================================================

def generate_dataset():

    dataset = []

    for intent in INTENTS:

        print(f"\nGenerating {intent}...")

        collected = 0

        while collected < TARGET_PER_INTENT:

            prompt = get_prompt(intent)

            samples = call_mistral(prompt)

            if not samples:

                print("Retrying batch...")
                continue

            for sample in samples:

                cleaned = clean_text(sample)

                if cleaned is not None:

                    dataset.append({
                        "text": cleaned,
                        "label": intent
                    })

            collected += len(samples)

            print(f"{intent}: {collected}/{TARGET_PER_INTENT}")

            time.sleep(random.uniform(1.5, 3.0))

    print("\nAdding hard cases...")
    dataset = add_hard_cases(dataset)

    print("Deduplicating...")
    dataset = deduplicate(dataset)

    random.shuffle(dataset)

    return dataset


# =========================================================
# SAVE
# =========================================================

def save_dataset(dataset):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(dataset)} samples to {OUTPUT_FILE}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    dataset = generate_dataset()

    save_dataset(dataset)
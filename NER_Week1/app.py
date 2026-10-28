
import json
from pathlib import Path

from transformers import pipeline


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
OUTPUT_PATH = BASE_DIR / "results.json"


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

with DATASET_PATH.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

feedback_records = dataset["feedback"]
policy = dataset["policy"]


# --------------------------------------------------
# 3. Load pretrained NLP models
# --------------------------------------------------

print("Loading NLP models. First run may take time...")

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=(
        "distilbert/"
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
)

ner_model = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

print("All models loaded successfully.")


# --------------------------------------------------
# 4. Text summarization
# --------------------------------------------------

print("\n" + "=" * 60)
print("TASK 1: TEXT SUMMARIZATION")
print("=" * 60)

policy_summary = summarizer(
    policy["text"],
    max_new_tokens=100,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

print("\nOriginal policy:")
print(policy["text"])

print("\nGenerated summary:")
print(policy_summary)


# --------------------------------------------------
# 5. Sentiment analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("TASK 2: SENTIMENT ANALYSIS")
print("=" * 60)

sentiment_results = []

for record in feedback_records:
    prediction = sentiment_analyzer(record["text"])[0]

    result = {
        "id": record["id"],
        "text": record["text"],
        "sentiment": prediction["label"],
        "score": round(float(prediction["score"]), 4)
    }

    sentiment_results.append(result)

    print(f"\nFeedback ID: {result['id']}")
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Score: {result['score']}")


# --------------------------------------------------
# 6. Named Entity Recognition (NER)
# --------------------------------------------------

print("\n" + "=" * 60)
print("TASK 3: NAMED ENTITY RECOGNITION")
print("=" * 60)

ner_results = []

for record in feedback_records:
    entities = ner_model(record["text"])

    extracted_entities = [
        {
            "text": entity["word"],
            "type": entity["entity_group"],
            "score": round(float(entity["score"]), 4)
        }
        for entity in entities
    ]

    result = {
        "id": record["id"],
        "text": record["text"],
        "entities": extracted_entities
    }

    ner_results.append(result)

    print(f"\nFeedback ID: {record['id']}")
    print(f"Text: {record['text']}")

    if extracted_entities:
        for entity in extracted_entities:
            print(
                f"  {entity['text']} "
                f"-> {entity['type']} "
                f"(score: {entity['score']})"
            )
    else:
        print("  No entities detected.")


# --------------------------------------------------
# 7. Save results as JSON
# --------------------------------------------------

output = {
    "policy": {
        "title": policy["title"],
        "summary": policy_summary
    },
    "sentiment_analysis": sentiment_results,
    "named_entity_recognition": ner_results
}

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(output, file, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"Processing completed. Results saved to: {OUTPUT_PATH}")
print("=" * 60)

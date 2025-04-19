from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json
import torch
import random
import os

app = Flask(__name__)

# Load Sentence Transformer Model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load LLM Model for fallback responses
llm = pipeline("text-generation", model="facebook/blenderbot-400M-distill")

# Load intents file
intents_path = os.path.join("dataset", "intents1.json")
if not os.path.exists(intents_path):
    raise FileNotFoundError("Intents file is missing!")

with open(intents_path, 'r', encoding="utf-8") as f:
    intents = json.load(f)

# Prepare intent patterns and embeddings
intent_texts = []
intent_tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        intent_texts.append(pattern)
        intent_tags.append(intent['tag'])

intent_embeddings = embedding_model.encode(intent_texts, convert_to_tensor=True)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.6  # Adjusted for better matching


def chatbot_response(user_input):
    """Finds the best-matching intent or generates a response using LLM."""
    if not user_input.strip():
        return "Please enter a valid query."

    user_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, intent_embeddings)[0]
    best_match_idx = torch.argmax(similarities).item()
    best_match_score = similarities[best_match_idx].item()

    print(f"\nUser Input: {user_input}")
    print(f"Best Match Pattern: {intent_texts[best_match_idx]}")
    print(f"Best Match Score: {best_match_score}\n")

    # Check if similarity is above threshold
    if best_match_score >= SIMILARITY_THRESHOLD:
        matched_tag = intent_tags[best_match_idx]
        for intent in intents['intents']:
            if intent['tag'] == matched_tag:
                return random.choice(intent['responses'])

    print("No suitable match found, using LLM for response.")
    llm_response = llm(user_input, max_length=50, do_sample=True)[0]['generated_text']
    return llm_response

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('user_input', '').strip()
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
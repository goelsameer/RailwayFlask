import spacy
import torch
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import math
app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

# Load the BLIP model (Better than CLIP for "writing" labels from scratch)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")


def extract_keywords(text):
    """Uses spaCy to filter out filler words and keep only Nouns/Adjectives"""
    doc = nlp(text)
    # We only want the core subjects (Nouns) and descriptors (Adjectives)
    keywords = [token.text.lower()
                for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    return list(set(keywords))  # Remove duplicates


def extract_contextual_tags(text):
    doc = nlp(text)
    context_tags = []

    keywords = [token.text.lower()
                for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]

    for token in doc:
        if token.dep_ in ("prep", "agent"):  # Look for "over", "under", "near"
            # Join the head word and the child word
            # e.g., "man" + "over" + "fence"
            relation = f"{token.head.text.lower()} {token.text.lower()} {token.head.head.text.lower() if token.head.head else ''}"
            context_tags.append(relation.strip())

    return list(set(keywords + context_tags))
# @app.route('/tag', methods=['POST'])
# def tag_image():
#     data = request.json
#     img_path = data.get('path')

#     raw_image = Image.open(img_path).convert('RGB')

#     # Generate the description automatically
#     inputs = processor(raw_image, return_tensors="pt")
#     out = model.generate(**inputs)
#     description = processor.decode(out[0], skip_special_tokens=True)

#     # description will be something like "a drone view of a rocky river with trees"
#     return jsonify({"tags": [description]})


model.to(device)


@app.route('/tag', methods=['POST'])
def tag_image():
    data = request.json
    img_path = data.get('path')

    drone_id = data.get('droneID', 'Unknown')
    gps = data.get('gps', {"lat": 0, "lng": 0})
    timestamp = data.get('timestamp', '00:00:00')

    try:
        raw_image = Image.open(img_path).convert('RGB')
        with torch.no_grad():
            inputs = processor(raw_image, return_tensors="pt").to(device)

            if device == "cuda":
                inputs = {k: v.to(torch.float16)
                          for k, v in inputs.items() if torch.is_floating_point(v)}

            out = model.generate(**inputs, max_new_tokens=20, min_new_tokens=5)
            description = processor.decode(out[0], skip_special_tokens=True)

        simple_tags = extract_keywords(description)
        context_tags = extract_contextual_tags(description)

        return jsonify({
            "status": "success",
            "tags": {"context_tags": context_tags, "simple_tags": simple_tags},
            "caption": description,
            "metadata": {
                "droneID": drone_id,
                "gps": gps,
                "timestamp": timestamp
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001)



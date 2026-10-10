
from transformers import pipeline
from PIL import Image

categories = [
    "a photograph of a plastic bottle",
    "a photograph of paper waste",
    "a photograph of cardboard waste",
    "a photograph of a glass bottle",
    "a photograph of a metal can",
    "a photograph of a household battery",
    "a photograph of electronic waste such as a mobile phone",
    "a photograph of general household waste"
]

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)

def classify_image(image):
    image = image.convert("RGB")

    predictions = classifier(
        image,
        candidate_labels=categories
    )

    result = predictions[0]

    return {
        "label": result["label"],
        "score": float(result["score"])
    }

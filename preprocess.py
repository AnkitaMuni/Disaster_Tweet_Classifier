import spacy # type: ignore
from spacy.lang.en.stop_words import STOP_WORDS # type: ignore

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Preprocesses the input text using spaCy.
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return ' '.join(tokens)

def extract_location(text):
    """
    Extracts location from the text using spaCy's NER.
    """
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations[0] if locations else "No location detected"

def extract_disaster_type(text):
    """
    Extracts the type of disaster from the text based on keywords.
    """
    disaster_keywords = {
        "flood": "Flood",
        "earthquake": "Earthquake",
        "wildfire": "Wildfire",
        "hurricane": "Hurricane",
        "tsunami": "Tsunami",
        "landslide": "Landslide",
        "volcano": "Volcano"
    }
    
    for keyword, disaster_type in disaster_keywords.items():
        if keyword in text:
            return disaster_type
    return "Unknown"
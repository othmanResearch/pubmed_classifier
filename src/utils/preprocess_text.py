import spacy
import re
# Load spaCy model once globally
nlp = spacy.load("en_core_web_sm")

# Load spaCy model (disable unnecessary components for speed)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def tokenize_text(text, placeholders=None):
    """
    Tokenize a text while preserving placeholders.
    
    Args:
        text (str): Input text.
        placeholders (list of str, optional): List of placeholder strings to preserve.
    
    Returns:
        list of str: Tokenized text.
    """
    if placeholders is None:
        placeholders = []

    # Step 1: Temporarily protect placeholders with a unique marker
    placeholder_map = {}
    for i, ph in enumerate(placeholders):
        key = f"__PH{i}__"
        placeholder_map[key] = ph
        text = text.replace(ph, key)

    # Step 2: Tokenize with spaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Step 3: Restore placeholders
    for key, ph in placeholder_map.items():
        tokens = [ph if t == key else t for t in tokens]

    return tokens


def remove_stopwords_punct(tokens):
    """
    Remove stopwords and punctuation from a list of tokens using spaCy.
    
    Args:
        tokens (list[str]): list of tokens (output of tokenize_text)
    
    Returns:
        list[str]: cleaned tokens (without stopwords/punctuation)
    """
    doc = nlp(" ".join(tokens))
    cleaned_tokens = [
        token.text for token in doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    return cleaned_tokens



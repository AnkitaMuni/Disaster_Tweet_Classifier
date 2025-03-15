import nltk
from nltk.tokenize import word_tokenize

# Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Test tokenization
text = "Flooding in Mumbai. People are evacuating."
tokens = word_tokenize(text)
print("Tokens:", tokens)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

# Load stopwords for both languages
french_stopwords = set(stopwords.words('french'))
english_stopwords = set(stopwords.words('english'))

# Load additional French stopwords
try:
    with open("data/stopWords/additional_stop_words", "r", encoding='utf8') as f:
        additionalFrenchStopWords = f.read().splitlines()
    french_stopwords.update(additionalFrenchStopWords)
except FileNotFoundError:
    print("Additional French stopwords file not found.")


def remove_accents(input_str):
    """Remove accents from a given string."""
    trans = str.maketrans("éàèùâêîôûçæœ", "eaeuaeioucao")
    return input_str.translate(trans)


def preprocess(sentence, language='auto'):
    """Preprocess a sentence in either French or English."""
    if language == 'auto':
        # Simple heuristic: if more French stopwords are present, assume French
        language = 'french' if len([word for word in sentence.split() if word in french_stopwords]) > \
                               len([word for word in sentence.split() if word in english_stopwords]) else 'english'

    # Initialize stemmer and stopwords based on detected or specified language
    if language == 'french':
        stemmer = SnowballStemmer("french")
        stopwords_set = french_stopwords
    else:
        stemmer = PorterStemmer()
        stopwords_set = english_stopwords

    # Preprocessing
    words = sentence.lower().split()
    no_punc = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    no_stops = [word for word in no_punc if word not in stopwords_set]
    stemmed = [stemmer.stem(remove_accents(word)) for word in no_stops]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

    return " ".join(lemmatized)


# Test the function with both French and English examples
french_sentence = "Ceci est un test pour vérifier le prétraitement des textes en français."
english_sentence = "This is a test to check text preprocessing in English."

print("French:", preprocess(french_sentence))
print("English:", preprocess(english_sentence))

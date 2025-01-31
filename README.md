# Natural Language Processing (NLP) Code Examples

This repository contains code examples and explanations for various Natural Language Processing (NLP) concepts and techniques. The code demonstrates the implementation of fundamental NLP tasks using popular libraries like NLTK, spaCy, Transformers, and Gensim.

## Table of Contents

1. [Basic NLP Concepts](#basic-nlp-concepts)
2. [Text Preprocessing](#text-preprocessing)
3. [Text Representation](#text-representation)
4. [Word Embeddings](#word-embeddings)
5. [Advanced Models](#advanced-models)
6. [Applications](#applications)

## Basic NLP Concepts

Natural Language Processing (NLP) is a branch of artificial intelligence focused on enabling computers to understand and process human language. This repository covers various NLP techniques and their implementations.

### Key Components:
- Text processing and analysis
- Language understanding
- Text generation
- Sentiment analysis
- Named Entity Recognition (NER)

## Text Preprocessing

### Tokenization
```python
from nltk.tokenize import word_tokenize, sent_tokenize
text = "This is a simple sentence."
word_tokens = word_tokenize(text)
sentence_tokens = sent_tokenize(text)
```

### Stopword Removal
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
```

### Stemming and Lemmatization
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stem = PorterStemmer()
lemmatizer = WordNetLemmatizer()
```

## Text Representation

### Bag of Words (BoW)
```python
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(texts)
```

### TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(texts)
```

## Word Embeddings

### Word2Vec
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
```

### FastText
```python
from gensim.models import FastText
fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1)
```

### GloVe
```python
# Loading pre-trained GloVe embeddings
glove_dict = {}
with open("glove.6B.100d.txt", 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        glove_dict[word] = vector
```

## Advanced Models

### BERT
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```

### GPT-2
```python
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
```

## Applications

### Text Classification
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# Train classifiers on vectorized text
```

### Named Entity Recognition
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Text to analyze")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

### Text Summarization
```python
from transformers import pipeline
summarizer = pipeline("summarization")
summary = summarizer(text, max_length=50, min_length=10)
```

## Requirements

- Python 3.7+
- NLTK
- spaCy
- Transformers
- Gensim
- scikit-learn
- NumPy
- pandas

## Installation

Install required packages using pip:

```bash
pip install nltk spacy transformers gensim scikit-learn numpy pandas
```

Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

Each section in this repository contains example code that can be run independently. Make sure to install all required dependencies and download necessary models before running the code.

## Note

This code is for educational purposes and demonstrates various NLP concepts and techniques. For production use, please consider:
- Data preprocessing requirements
- Model selection based on specific use cases
- Computational resources
- Performance optimization

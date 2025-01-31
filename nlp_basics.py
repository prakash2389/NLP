### nlp

# Natural Languga Processing (NLP) is a data science where interaction between
# computer and human language. It is a field of artificial intelligence that gives
# the machines the ability to read, understand and derive meaning from human languages.

# NLP is used to apply machine learning algorithms to text and speech.

# NLP is used in:
# 1. Sentiment Analysis
# 2. Chatbots
# 3. Speech Recognition
# 4. Machine Translation
# 5. Named Entity Recognition
# 6. Text Classification
# 7. Language Modelling
# 8. Text Summarization
# 9. Question Answering
# 10. Text Generation
# 11. Text to Speech
# 12. Speech to Text
# 13. Document Clustering
# 14. Document Classification
# 15. Text Similarity
# 16. Text Extraction
# 17. Text Segmentation
# 18. Text Annotation
# 19. Text Normalization
# 20. Text Correction
# 21. Text Completion
# 22. Text Generation
# 23. Text Analysis
# 24. Text Mining
# 25. Text Understanding
# 26. Text Processing
# 27. Text Preprocessing
# 28. Text Cleaning
# 29. Text Wrangling
# 30. Text Transformation
# 31. Text Conversion
# 32. Text Representation
# 33. Text Encoding
# 34. Text Decoding
# 35. Text Translation
# 36. Text Conversion

# NLP Libraries:
# 1. NLTK (Natural Language Toolkit)
# 2. TextBlob
# 3. Gensim
# 4. SpaCy
# 5. Pattern
# 6. Polyglot
# 7. Stanford NLP
# 8. OpenNLP
# 9. Apache Lucene
# 10. Apache OpenNLP
# 11. Apache UIMA
# 12. Stanford CoreNLP
# 13. Stanford Parser
# 14. Stanford POS Tagger
# 15. Stanford Named Entity Recognizer
# 16. Stanford OpenIE
# 17. Stanford Sentiment Analysis
# 18. Stanford Relation Extractor
# 19. Stanford Temporal Tagger

# NLP Techniques:
# 1. Tokenization
# 2. Stopword Removal
# 3. Stemming
# 4. Lemmatization
# 5. Bag of Words
# 6. TF-IDF
# 7. Word Embedding
# 8. Named Entity Recognition
# 9. Sentiment Analysis
# 10. Text Classification
# 11. Language Modelling
# 12. Text Summarization

# NLP Applications:
# 1. Chatbots

# NLP Challenges:
# 1. Ambiguity
# 2. Polysemy
# 3. Synonymy
# 4. Named Entity Recognition

# NLP Datasets:
# 1. IMDB Movie Review Dataset
# 2. Amazon Product Review Dataset
# 3. Yelp Review Dataset
# 4. Twitter Sentiment Analysis Dataset
# 5. SMS Spam Collection Dataset
# 6. News Headlines Dataset
# 7. Wikipedia Dataset
# 8. Stack Overflow Dataset
# 9. Quora Question Pairs Dataset
# 10. Cornell Movie Dialogs Corpus

#Lets start with the basics of NLP
# Tokenization
# Tokenization is the process of breaking down a text into words or sentences.
# It is the first step in NLP.
# Tokenization can be done at different levels:
# 1. Word Tokenization - Breaks text into words
# 2. Sentence Tokenization - Breaks text into sentences
# 3. Subword Tokenization - Breaks text into subwords
# 4. Character Tokenization - Breaks text into characters

# Tokenization can be done using the following libraries:
# 1. NLTK (Natural Language Toolkit)
# 2. SpaCy
# 3. TextBlob

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer, RegexpTokenizer
nltk.download('punkt')
text = "This is a simple sentence."
word_tokens = word_tokenize(text)
print(word_tokens)
print(" ".join(word_tokens))

#Try with imdb dataset
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')
type(movie_reviews)
print(movie_reviews.categories())
print(movie_reviews.fileids())
len(movie_reviews.fileids())
print(movie_reviews.fileids('neg'))
print(movie_reviews.fileids('pos'))
print(movie_reviews.raw('neg/cv000_29416.txt'))

# Sentence Tokenization
text = movie_reviews.raw('neg/cv000_29416.txt')
word_tokens = word_tokenize(text)
print(word_tokens)
sent_tokens = sent_tokenize(text)
print(sent_tokens)

# Tweet Tokenization
tweet_tokenizer = TweetTokenizer()
tweet = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
tweet_tokens = tweet_tokenizer.tokenize(tweet)
print(tweet_tokens)

import re
re.sub('[A-Z\da-z\s]+', '', tweet)
re.sub(r'\w+', '', tweet)
re.findall(r'\w+', tweet)

# Regular Expression Tokenization
text = "This is a simple ###;123 sentence."
tokenizer = RegexpTokenizer(r'\w+')  # Only words
word_tokens = tokenizer.tokenize(text)
print(word_tokens)

#next process after tokenization is Stopword Removal
# Stopword Removal
# Stopwords are common words that do not add much meaning to a text.
# Examples of stopwords: 'the', 'is
# Stopword removal is the process of removing stopwords from a text.
# It is done after tokenization.
from nltk.corpus import stopwords
nltk.download('stopwords')

text = "I am doing well. I am working on gardening. I will be watching the movies this weekends!!! I am looking forward to the weekend."
re_tokenizer = RegexpTokenizer(r'\w+')
word_tokens = re_tokenizer.tokenize(text)
print(word_tokens)
stop_words = set(stopwords.words('english'))
print(stop_words)
filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
print(filtered_words)

#next step in stemming and lemmatization
# Stemming and Lemmatization
# Stemming and Lemmatization are text normalization techniques.
# They are used to reduce inflected words to their base or root form.
# Stemming is the process of removing suffixes from words.
# Example: 'running' -> 'run'
# Lemmatization is the process of reducing words to their base or root form.
# Example: 'better' -> 'good'
# Lemmatization is more accurate than stemming.
# Stemming is faster than lemmatization.

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

porter_stem = PorterStemmer()
filtered_words = [porter_stem.stem(word) for word in filtered_words]
print(filtered_words)

lemmatizer = WordNetLemmatizer()
filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print(filtered_words)

# Bag of Words
# Bag of Words is a text representation technique in NLP.
# It is used to convert text data into numerical data.
# It is a simple and effective technique.
# It is used for text classification, sentiment analysis, etc.
# Bag of Words is based on the frequency of words in a text.
# It ignores the order of words in a text.
# Bag of Words has two steps:
# 1. Tokenization
# 2. Counting the frequency of words
# Bag of Words can be implemented using the following libraries:
# 1. CountVectorizer (Scikit-learn)
# 2. TfidfVectorizer (Scikit-learn)
# 3. HashingVectorizer (Scikit-learn)
# 4. Bag of Words (NLTK)
# 5. TextBlob

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
text = ["I am doing well. I am working on gardening. I will be watching the movies this weekends!!! I am looking forward to the weekend.",
        "Natural Language Processing (NLP) for representing and processing text, especially in the context of word embeddings. Here’s a comparison of the two approaches",
        "It creates a vocabulary of all unique words in the corpus and represents each document as a vector with dimensions equal to the vocabulary size. Each word occurrence is counted, disregarding grammar and word order",
        "It’s a frequency-based model, so it doesn’t capture context or semantics, only word presence"]
text_df = pd.DataFrame(text, columns=['text'])
Count_vectorizer = CountVectorizer()
text_df['filtered_words'] = text_df['text'].apply(lambda x: " ".join([word for word in word_tokenize(x) if word.lower() not in set(stopwords.words('english'))]))
print(text_df)
X = Count_vectorizer.fit_transform(text_df['filtered_words'])
print(X.toarray())
print(Count_vectorizer.get_feature_names_out())

# TF-IDF
# TF-IDF stands for Term Frequency-Inverse Document Frequency.
# It is a text representation technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the frequency of words in a text.
# It is used for text classification, sentiment analysis, etc.
# TF-IDF is a combination of two terms:
# 1. Term Frequency (TF) - Frequency of a word in a text
# 2. Inverse Document Frequency (IDF) - Importance of a word in a corpus
# TF-IDF has two steps:
# 1. Tokenization
# 2. Calculating the TF-IDF score
# TF-IDF can be implemented using the following libraries:
# 1. TfidfVectorizer (Scikit-learn)
# 2. TfidfTransformer (Scikit-learn)
# 3. TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
tfIdf_vectorizer = TfidfVectorizer()
X = tfIdf_vectorizer.fit_transform(text_df['filtered_words'])

print(X.toarray())
print(tfIdf_vectorizer.get_feature_names_out())
pd.DataFrame(X.toarray(), columns=tfIdf_vectorizer.get_feature_names_out())

# bag of words and tf-idf can be used for text classification, sentiment analysis, etc.
# bag of words from nltk
from nltk.tokenize import word_tokenize
from nltk.text import Text
from nltk.text import TextCollection

text_collection = TextCollection(text_df['filtered_words'])
print(text_collection.tf('word', text_df['filtered_words'][2]))
print(text_collection.idf('word'))


vocab = Text(text_df['filtered_words']).vocab()

# Classification of using IMDB dataset
# Text Classification
# Text Classification is a supervised learning technique in NLP.
# It is used to classify text data into different categories or classes.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')

reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
sentiments = [fileid.split('/')[0] for fileid in movie_reviews.fileids()]
import pandas as pd
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)
accuracy_score(y_test, y_pred)
accuracy_score(y_train, naive_bayes.predict(X_train))

from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
X = count_vectorizer.fit_transform(df['review'])
y = df['sentiment']
y = y.map({'neg': 0, 'pos': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
accuracy_score(y_test, y_pred)
accuracy_score(y_train, logistic_regression.predict(X_train))


from sklearn.linear_model import RidgeClassifier
Ridge_Classifier = RidgeClassifier()

Ridge_Classifier.fit(X_train, y_train)
y_pred = Ridge_Classifier.predict(X_test)
accuracy_score(y_test, y_pred)
accuracy_score(y_train, Ridge_Classifier.predict(X_train))


#TF IDF usage
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdf_vectorizer = TfidfVectorizer()
X = tfIdf_vectorizer.fit_transform(df['review'])
print(X.toarray())
print(tfIdf_vectorizer.get_feature_names_out())

count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(df['review'])
print(X.toarray())


#TF-IDF calculation mathematically
# TF-IDF Calculation
# TF-IDF stands for Term Frequency-Inverse Document Frequency.
# It is a text representation technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the frequency of words in a text.
# It is used for text classification, sentiment analysis, etc.
# TF-IDF is a combination of two terms:
# 1. Term Frequency (TF) - Frequency of a word in a text
# 2. Inverse Document Frequency (IDF) - Importance of a word in a corpus

from sklearn.feature_extraction.text import TfidfVectorizer
documents = ["Machine science is", "Machine learning is powerful"]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())

from nltk.text import TextCollection
text_collection = TextCollection(documents)
text_collection.idf('Machine')*text_collection.tf('machine', 'Machine science is')
text_collection.idf('Machine')


# Word Embedding
# Word Embedding is a text representation technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the semantic meaning of words.
# It is used for text classification, sentiment analysis, etc.
# Word Embedding is a dense representation of words.
# Word Embedding has two steps:
# 1. Tokenization
# 2. Word Embedding
# Word Embedding can be implemented using the following libraries:
# 1. Word2Vec (Gensim)
# 2. GloVe (Stanford NLP)
# 3. FastText (Facebook AI)
# 4. BERT (Google AI)
# 5. ELMo (Allen AI)
# 6. GPT-2 (OpenAI)
# 7. Transformer (Google AI)
# 8. XLNet (Google AI)
# 9. RoBERTa (Facebook AI)
# 10. ALBERT (Google AI)
# 11. T5 (Google AI)
# 12. BART (Facebook AI)
# 13. MarianMT (Facebook AI)
# 14. XLM (Facebook AI)
# 15. mBERT (Google AI)
# 16. DistilBERT (Hugging Face)
# 17. CamemBERT (Facebook AI)
# 18. XLM-RoBERTa (Facebook AI)
# 19. ELECTRA (Google AI)
# 20. Reformer (Google AI)
# 21. Pegasus (Google AI)
# 22. ProphetNet (Microsoft)

# Word2Vec
# Word2Vec is a word embedding technique in NLP.
# It is used to convert text data into numerical data.
import gensim
from gensim.models import Word2Vec
# Sample corpus
sentences = [
"What it is: Word embeddings map words into dense vector spaces where semantically similar words are closer to each other.",
"Why it’s better: Unlike BoW, embeddings capture contextual similarity and relationships between words, so words like 'king'' and 'queen' are closer in the vector space than unrelated words.",
"When to use: For tasks that benefit from semantic similarity and where capturing word relationships is crucial (e.g., sentiment analysis, recommendations)."]

from nltk.tokenize import RegexpTokenizer
regexp_tokenizer =  RegexpTokenizer(r'\w+')
word_tokens = [regexp_tokenizer.tokenize(sentence) for sentence in sentences]
stop_words = set(stopwords.words('english'))
filtered_words = [ [word for word in word_token if word not in stop_words] for word_token in word_tokens]
lemmatizer = WordNetLemmatizer()
filtered_words = [ [lemmatizer.lemmatize(word) for word in word_token] for word_token in filtered_words]

Word2Vec_model = Word2Vec(filtered_words, vector_size=100, window=5)
Word2Vec_model.wv['word']
Word2Vec_model.wv['similarity']
Word2Vec_model.wv.key_to_index
Word2Vec_model.wv.index_to_key

#vector_size: The dimensionality of the word vectors.
#window: The maximum distance between the current and predicted word within a sentence.
#min_count: Ignores all words with a total frequency lower than this.
#sg: The training algorithm. If 1, skip-gram is used; if 0, CBOW is used.

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(words = word_tokenize(sentence), tags = [str(i)]) for i, sentence in enumerate(sentences)]
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
model.dv["0"]
model.dv["1"]
#Similarity between two sentences
model.dv.most_similar("0") # Most similar to sentence 0
model.dv.most_similar("1") # Most similar to sentence 1

#distances between two sentences
model.dv.distances("0", ["1", "2"]) # Distance between sentence 0 and sentence 1
model.dv.distances("1", ["0", "2"]) # Distance between sentence 1 and sentence 2

# FastText
# FastText is a word embedding technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the semantic meaning of words.
#FastText code below
from gensim.models import FastText
FastText_model = FastText([regexp_tokenizer.tokenize(sentence) for sentence in text_df['filtered_words']], min_count=1, vector_size=100, window=5, sg=1)
FastText_model.wv['analysis']
FastText_model.wv.most_similar('analysis')
FastText_model.wv.most_similar('king')
FastText_model.wv.most_similar('apple')
FastText_model.wv.distances('apple', ['watching', 'comparison'])
FastText_model.wv.most_similar(positive=['men', 'king'], negative=['queen'])



from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(FastText_model.wv['palm'].reshape(1,100), FastText_model.wv['coconut'].reshape(1,100))




# BERT (Bidirectional Encoder Representations from Transformers)
# BERT is a word embedding technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the semantic meaning of words.
# BERT is a transformer-based model.
# BERT is pre-trained on a large corpus of text data.
# BERT is fine-tuned on specific NLP tasks.
# BERT has two versions:
# 1. BERT Base (12 layers, 110 million parameters)
# 2. BERT Large (24 layers, 340 million parameters)
# BERT can be implemented using the following libraries:
# 1. Hugging Face Transformers
# 2. TensorFlow Hub
# 3. PyTorch Transformers
# 4. Transformers (Hugging Face)

# BERT code below
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Downloading the pre-trained tokenizer
model = BertModel.from_pretrained('bert-base-uncased')  # Downloading the pre-trained model
tokenizer.bos_token = "<BOS>"
tokenizer.eos_token = "<EOS>"
# text = ["Is Machine learning is is is a subset of artificial intelligence.","How are you doing?",
#         "What is your name?",
#         "What is your age?"]
text = "Machine learning is a subset of artificial intelligence."
# inputs = [tokenizer(te, return_tensors='pt') for te in text] # Tokenizing the text
inputs = tokenizer(text, return_tensors='pt')  # Tokenizing the text
outputs = model(**inputs)  # Passing the input to the model
outputs.last_hidden_state
outputs.last_hidden_state.size()
outputs.pooler_output.size()
# Use of grad_fn
outputs.last_hidden_state.requires_grad
outputs.last_hidden_state.grad_fn


#GPT-2 (Generative Pre-trained Transformer 2)
# GPT-2 is a word embedding technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the semantic meaning of words.
# GPT-2 is a transformer-based model.
# GPT-2 is pre-trained on a large corpus of text data.
# GPT-2 is fine-tuned on specific NLP tasks.
# GPT-2 has different versions:
# 1. GPT-2 Small (117 million parameters)
# 2. GPT-2 Medium (345 million parameters)
# 3. GPT-2 Large (762 million parameters)

# GPT-2 code below
from transformers import GPT2Tokenizer, GPT2Model
gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Downloading the pre-trained tokenizer
gptmodel = GPT2Model.from_pretrained('gpt2')  # Downloading the pre-trained model
gpttokenizer.sep_token = "<SEP>"
gpttokenizer.pad_token = "<PAD>"
gpttokenizer.cls_token = "<CLS>"
gpttokenizer.mask_token = "<MASK>"
text = "Machine learning is a subset of artificial intelligence."
inputs = gpttokenizer(text, return_tensors='pt')  # Tokenizing the text
outputs = gptmodel(**inputs)  # Passing the input to the model
outputs.last_hidden_state
outputs

# Text Summarization
# Text Summarization is a text analysis technique in NLP.
# It is used to summarize a text into a shorter version.
# It is based on the content of the text.
# Text Summarization has two types:
# 1. Extractive Summarization - Selects important sentences from the text
# 2. Abstractive Summarization - Generates new sentences to summarize the text
# Text Summarization can be implemented using the following libraries:
# 1. Sumy
# 2. Gensim
# 3. NLTK
# 4. TextBlob
# 5. BERT
# 6. GPT-2

# Text Summarization code below
from transformers import pipeline
summarizer = pipeline("summarization")
text = "Machine learning is a subset of artificial intelligence. It is the study of computer algorithms that improve automatically through experience. Machine learning algorithms build a mathematical model based on sample data, known as 'training data', to make predictions or decisions without being explicitly programmed to perform the task."
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
print(summary)

# what models are used in above code?
# 1. BERT (Bidirectional Encoder Representations from Transformers)
# 2. GPT-2 (Generative Pre-trained Transformer 2)

#Latent Semantic Analysis
# Latent Semantic Analysis is a text analysis technique in NLP.
# It is used to find the hidden structure in a text.
# It is based on the semantic meaning of words.
# Reduce the dimensionality of a BoW matrix
# identifying latent semantic structures in the data.
# What it is: LSA uses singular value decomposition (SVD) to reduce the dimensionality of a BoW matrix, identifying latent semantic structures in the data.
# Why it’s better: It captures relationships between terms and concepts, reducing the impact of sparsity in BoW and helping with topic modeling.
# When to use: When performing topic modeling or dimensionality reduction on text data.
# Latent Semantic Analysis has two steps:
# 1. Tokenization
# 2. Singular Value Decomposition (SVD)
# Latent Semantic Analysis can be implemented using the following libraries:
# 1. Scikit-learn
# 2. Gensim
# 3. NLTK
# 4. TextBlob

# Latent Semantic Analysis code below
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

documents = ["Machine learning is a subset of artificial intelligence.",
                "It is the study of computer algorithms that improve automatically through experience.",
                "Machine learning algorithms build a mathematical model based on sample data, known as 'training data', to make predictions or decisions without being explicitly programmed to perform the task."]

tfIdf_vectorizer = TfidfVectorizer()
X = tfIdf_vectorizer.fit_transform(documents)
X.toarray().shape

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
X_svd.shape

#usin gensim
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense

print(documents)
documents = [regexp_tokenizer.tokenize(document) for document in documents]
print(documents)
dictionary = Dictionary(documents)
corpus = [dictionary.doc2bow(document) for document in documents]   # Converting the documents to a BoW format
print(corpus)

lsi_model = LsiModel(corpus, num_topics=2)
lsi_model.show_topics()

#ngram
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
X = vectorizer.fit_transform(["Data science is fun", "Machine learning is powerful"])


#ngram using nltk
from nltk.util import ngrams
text = "Data science is fun"
n = 2
ngrams_list = list(ngrams(text.split(), n))

# ngram with 1 2 3 using nltk
ngrams_list = [ngrams(text.split(), n) for n in range(1, 4)]


# using gensim ngram
from gensim.models import Phrases
from gensim.models.phrases import Phraser
sentences = [["data", "science", "is", "fun"], ["machine", "learning", "is", "powerful"]]
bigram = Phrases(sentences, min_count=1, threshold=1)
bigram_phraser = Phraser(bigram)
bigram_phraser[sentences[0]]

# using spacy
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Data science is fun"
doc = nlp(text)
ngrams_list = [doc[i:i+2] for i in range(len(doc)-1)]

# using textblob
from textblob import TextBlob
text = "Data science is fun"
blob = TextBlob(text)
print(blob)
ngrams_list = blob.ngrams(n=1) + blob.ngrams(n=2)  + blob.ngrams(n=3)
print(ngrams_list)

# using nltk
from nltk.util import ngrams
text = "Data science is fun"
n = 2
ngrams_list = list(ngrams(text.split(), n))
print(ngrams_list)



#named entity recognition
from transformers import pipeline
nlp = pipeline("ner")

text = "prakash is good boy and he lives prasanna at nathavaram. he maried on 22/ 03/2024 at 10: 24PM"

# Perform NER
entities  = nlp(text)
print(entities)

# Display the extracted entities
for entity in entities:
    print(f"{entity['word']} - {entity['entity']} - {entity['score']:.3f}")


# Named Entity Recognition
# Named Entity Recognition is a text analysis technique in NLP.
# It is used to identify named entities in a text.
# It is based on the semantic meaning of words.
# Named entities can be of different types:
# 1. Person
# 2. Location
# 3. Organization
# 4. Date
# 5. Time
# 6. Money
# 7. Percent
# Named Entity Recognition can be implemented using the following libraries:
# 1. Spacy
# 2. NLTK
# 3. TextBlob
# 4. Stanford NER

# Named Entity Recognition code below
import spacy
nlp = spacy.load("en_core_web_sm")

obj = nlp("prakash kumar is learning nlp and transformers in python and His village in 123@gmail.com")

for ent in obj.ents:
    print(ent.text, ent.label_)


for token in obj:
    print(token.text, token.is_stop, token.pos_, token.tag_, token.dep_)
    token.is_alpha
    token.is_digit
    token.is_punct
    token.is_space
    token.is_upper
    token.is_lower
    token.is_title
    token.is_currency
    token.like_num
    token.like_email
    token.like_url
    token.like_num

# Gensim
# Gensim is a text analysis library in NLP.
#code
from gensim import corpora
from gensim.models import LsiModel, Word2Vec
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense

documents = ["Machine learning is a Positive of artificial Negative.",
             "BERT has successfully achieved state-of-the-art accuracy on 11 common NLP tasks, outperforming previous top NLP models, and is the first to outperform humans! But, how are these achievements measured"]


from nltk.tokenize import RegexpTokenizer
regexp_tokenizer = RegexpTokenizer(r'\w+')
documents = [regexp_tokenizer.tokenize(document) for document in documents]

vec = Word2Vec(documents, min_count=1, vector_size=100, window=5, sg=1)
vec.wv['achieved']

vec.wv.most_similar('NLP')
vec.wv.distance('NLP', 'BERT')
vec.wv.similarity('NLP', 'BERT')


#Glove
# GloVe is a word embedding technique in NLP.
# It is used to convert text data into numerical data.
# It is based on the semantic meaning of words.
# GloVe is a pre-trained model.
# GloVe is trained on a large corpus of text data.
# GloVe is fine-tuned on specific NLP tasks.
# GloVe has different versions:
# 1. GloVe 50d (50 dimensions)
# 2. GloVe 100d (100 dimensions)
# 3. GloVe 200d (200 dimensions)
# 4. GloVe 300d (300 dimensions)
# GloVe can be implemented using the following libraries:
# 1. Gensim
# 2. SpaCy

# GloVe code below
from gensim.models import KeyedVectors
import kagglehub
path = kagglehub.dataset_download("sawarn69/glove6b100dtxt")
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(r'D:\Downloads\glove.6B.100d.txt\glove.6B.100d.txt', 'glove.6B.100d.word2vec.txt')
glove = KeyedVectors.load_word2vec_format('glove.6B.100d.word2vec.txt', binary=False)
# How can we get glove.6B.100d.txt file?

#FastText

from gensim.models import KeyedVectors
fasttext_model = KeyedVectors.load_word2vec_format("D:\crawl-300d-2M.vec\crawl-300d-2M.vec", binary=False)

import numpy as np
np.dot(fasttext_model["king"], fasttext_model["king"])/ (np.linalg.norm(fasttext_model["king"]) * np.linalg.norm(fasttext_model["king"]))
fasttext_model.most_similar("iphone")


import fasttext
import fasttext.util

# Download and load pre-trained English model
fasttext.util.download_model('en', if_exists='ignore')  # Downloads 'cc.en.300.bin'
model = fasttext.load_model('cc.en.300.bin')

# Access word vectors
print(model.get_word_vector('king'))


import numpy as np

# Path to GloVe file (adjust path as needed)
glove_file = r'D:\Downloads\glove.6B.100d.txt\glove.6B.100d.txt'

# Dictionary to hold word embeddings
glove_embeddings = {}

# Load GloVe embeddings
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        glove_embeddings[word] = vector

print("Loaded GloVe embeddings with {} words.".format(len(glove_embeddings)))

glove_embeddings['men']

from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity between two words
cosine_similarity(glove_embeddings['men'].reshape(1, 100) , glove_embeddings['men'].reshape(1, 100))


from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

word1 = 'king'
word2 = 'queen'

if word1 in glove_embeddings and word2 in glove_embeddings:
    similarity = cosine_similarity(glove_embeddings[word1], glove_embeddings[word2])
    print(f"Cosine similarity between {word1} and {word2}: {similarity}")
else:
    print("One or both words not in GloVe embeddings.")


def find_closest_word(vector, embeddings, top_n=5):
    similarities = {}
    for word, embed_vector in embeddings.items():
        similarity = cosine_similarity(vector, embed_vector)
        similarities[word] = similarity
    # Sort words by similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]

# Perform analogy: king - man + woman ≈ queen
analogy_vector = glove_embeddings['king'] - glove_embeddings['man'] + glove_embeddings['woman']
similar_words = find_closest_word(analogy_vector, glove_embeddings)
print("Most similar words:", similar_words)



# Glove in spacy
import spacy
nlp = spacy.load("en_core_web_md")
nlp.vocab.vectors.shape
doc = nlp("Machine learning is a subset of artificial intelligence.")
doc[8].vector


# Calculate the cosine similarity between two words
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vec.wv['Positive'].reshape(1, len(vec.wv['Positive'])), vec.wv['Negative'].reshape(1, len(vec.wv['Negative'])))
np.dot(vec.wv['Positive'], vec.wv['Negative']) / (np.linalg.norm(vec.wv['Positive']) * np.linalg.norm(vec.wv['Negative']))
#cosine similarity between two words



# Sample corpus
corpus = [
    "Natural Language Processing is amazing.",
    "AI is the future of technology.",
    "Text analysis helps in extracting valuable information."
]

tfidf = TfidfVectorizer()
tfidf_corpus = tfidf.fit_transform(corpus)
tfidf_corpus.shape


from textblob import TextBlob

# Sample text
text = "I love learning about NLP and AI!"

# Create a TextBlob object
blob = TextBlob(text)

# Sentiment analysis
print(f"Sentiment: {blob.sentiment}")

from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

input = "Delhi restaurent is so sos but service is average. TATA car is in New York"

print(sentiment(input, return_all_scores=True))

complete =  pipeline("ner")
print(complete(input))

input = "Write factorial function in Python"
generate = pipeline("text-generation")
print(generate(input)[0]['generated_text'])


#SPACY Vectors
import spacy
nlp = spacy.load("en_core_web_lg")
doc = nlp("I have very big banana")

for token in doc:
    print(token.text, token.has_vector, token.is_oov)

for i in ["apple","banana"]:
    token_i = nlp(i)
    print(token_i.text)


import kagglehub

# Download latest version
path = kagglehub.dataset_download("sawarn69/glove6b100dtxt")

print("Path to dataset files:", path)


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation,TruncatedSVD
from gensim.models.ldamodel import LdaModel
from gensim import corpora
# import pyLDAvis
# import pyLDAvis.gensim_models

fetch_20news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data = fetch_20news.data
# print(data[0])
print(len(data))
data[0][:100]
tfidfvectot = TfidfVectorizer(stop_words='english')
tfidf = tfidfvectot.fit_transform(data)
print(tfidf.shape)
print(tfidf[0])
print(tfidfvectot.get_feature_names_out())

lda = LatentDirichletAllocation(n_components=2, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
lda.fit(tfidf)
print(lda.components_)
print(lda.components_.shape)
print(lda.transform(tfidf))


# LDA using gensim
documents = [text.split() for text in data]
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(document) for document in documents]
lda = LdaModel(corpus, num_topics=2, id2word=dictionary)
lda.show_topics()



glove_dict={}
with open(r"D:\Downloads\glove.6B.100d.txt\glove.6B.100d.txt", 'r', encoding='utf-8') as f:
    for lines in f:
        v = lines.split()
        glove_dict[v[0]] = np.array(v[1:])

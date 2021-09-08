from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np
import os

# Charging the dataset
df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df.head()

# Selecting the headline column
headline = df.iloc[:, 1:2]
print(headline)

# Converting the column of data into a list of documents,
corpus = []
for index, row in headline.iterrows():
    corpus.append(row['headline'])

for i in range(len(corpus)):
    print(corpus[i])

# Stemming
st = SnowballStemmer("english")
corpus_stemmed = []
for sentence in corpus:
    corpus_stemmed.append(" ".join([st.stem(i) for i in sentence.split()]))

for item in corpus_stemmed:
    print(item)

# Removing punctuation


def puncRemove(sentence):
    filtered_sentence = "".join(u for u in sentence if u not in (
        "?", ".", ";", ":", "!", "(", ")", ",", "[", "]"))
    return filtered_sentence


corpus_punc = []
for sentence in corpus_stemmed:
    new_sentence = puncRemove(sentence)
    corpus_punc.append(new_sentence)

for item in corpus_punc:
    print(item)

# Removing stop words
stop_words = set(stopwords.words('english'))


def stopwordsRemove(sentence, stopwords):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = ""
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence = filtered_sentence + w + " "
    return filtered_sentence


corpus_stopWords = []
for sentence in corpus_punc:
    new_sentence = stopwordsRemove(sentence, stopwords)
    corpus_stopWords.append(new_sentence)

for item in corpus_stopWords:
    print(item)

# Count Vectoriser
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus_stopWords)
print(X)

# td-idf transformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape)

# K-means
num_clusters = 30  # Change it according to your data.
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()

# Result
# Creating dict having doc with the corresponding cluster number.
headline = {'headline': corpus, 'Cluster': clusters}
# Converting it into a dataframe.
frame = pd.DataFrame(headline, index=[clusters], columns=[
                     'headline', 'Cluster'])

print("\n")
print(frame)
print("\n")
# Print the counts of doc belonging to each cluster.
print(frame['Cluster'].value_counts())

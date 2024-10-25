from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Create a corpus of text documents
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]

# Step 2: Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Step 3: Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Step 4: Convert the result to an array (optional)
tfidf_array = X.toarray()

# Step 5: Inspect the features and the TF-IDF matrix
print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_array)

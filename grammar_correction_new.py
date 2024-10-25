import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv('E:\\Natural Language Processing\\Machine Learning\\Grammar Correction.csv')

# Clean the data (optional)
data = data.dropna()  # Drop rows with missing values

# Map error types to numerical categories
def errors(x):
    error_dict = {
        'Sentence Structure Errors': 0,
        'Verb Tense Errors': 1,
        'Subject-Verb Agreement': 2,
        'Article Usage': 3,
        'Spelling Mistakes': 4,
        'Preposition Usage': 5
    }
    return error_dict.get(x, -1)

data['Errors'] = data['Error Type'].apply(errors)

# Feature and target
X = data['Ungrammatical Statement']
y = data['Errors']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Create a pipeline with TF-IDF and a classifier (e.g., RandomForest)
pipe = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=3)),
    ('classifier', RandomForestClassifier(class_weight='balanced', n_estimators=100))
])

# Train the model
pipe.fit(x_train, y_train)

# Evaluate the model
y_pred = pipe.predict(x_test)
print(pipe.score(x_test, y_test))  # Accuracy

# Test with a new sentence
print(pipe.predict(['He go to the market yesterday.']))

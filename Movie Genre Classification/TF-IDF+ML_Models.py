import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load and parse the dataset
df = pd.read_csv(
    'Genre Classification Dataset/train_data.txt',
    sep=' ::: ',
    engine='python',
    names=['ID', 'Title', 'Genre', 'Plot']
)

# Step 2: Basic data checks
print("Sample Data:\n", df.head())
print("\nAvailable Genres:\n", df['Genre'].value_counts())

# Step 3: Clean the Plot text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.lower()
    return ""

df['Plot'] = df['Plot'].apply(clean_text)

# Step 4: Encode target labels (Genres)
label_encoder = LabelEncoder()
df['Genre_encoded'] = label_encoder.fit_transform(df['Genre'])

# Step 5: Split dataset
X = df['Plot']
y = df['Genre_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Convert text to numeric features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train and evaluate ML models

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_vec, y_train)
log_pred = log_model.predict(X_test_vec)
print("\n🔹 Logistic Regression Report:\n")
print(classification_report(y_test, log_pred, target_names=label_encoder.classes_))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
print("\n🔹 Naive Bayes Report:\n")
print(classification_report(y_test, nb_pred, target_names=label_encoder.classes_))

# Support Vector Machine
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)
print("\n🔹 Support Vector Machine Report:\n")
print(classification_report(y_test, svm_pred, target_names=label_encoder.classes_))

# Step 8: Accuracy Comparison Summary
print("\n✅ Model Accuracy Summary:")
print(f"Logistic Regression: {accuracy_score(y_test, log_pred):.4f}")
print(f"Naive Bayes:         {accuracy_score(y_test, nb_pred):.4f}")
print(f"SVM:                 {accuracy_score(y_test, svm_pred):.4f}")

import pickle

# Save the model and vectorizer
with open('genre_classifier.pkl', 'wb') as f:
    pickle.dump(log_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

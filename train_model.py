import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
from preprocess import preprocess_text

df = pd.read_csv("augmented_disaster_tweets_doubled.csv")

df['cleaned_text'] = df['text'].apply(preprocess_text)

X = df['cleaned_text']
y = df['label'].apply(lambda x: 1 if x == "Disaster-related" else 0)  # Convert labels to binary (1 or 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  # Use bigrams and more features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Logistic Regression Model...")

logreg_param_grid = {
    'C': [0.1, 1, 10],  
    'solver': ['liblinear', 'lbfgs'], 
    'class_weight': ['balanced'], 
    'max_iter': [100, 200] 
}

grid_search = GridSearchCV(LogisticRegression(), logreg_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_tfidf)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Logistic Regression model and vectorizer saved successfully!")
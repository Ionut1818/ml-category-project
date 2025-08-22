import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Reading the data
url = "https://raw.github.com/Ionut1818/ml-category-project//main/data/products.csv"
df = pd.read_csv(url)

# Data verify
print("First 5 rows")
print(df.head())
print("\nData info")
print(df.info())

# Verify null values
print("\nNull values per column:")
print(df.isnull().sum())

# Drop columns that are not useful for modeling
df = df.drop(columns=['product ID', 'Merchant ID', '_Product Code', 'Number_of_Views', 'Merchant Rating', ' Listing Date  '])

# Eliminate rows with null values
df = df.dropna(subset=['Product Title', ' Category Label'])

# Category distribution analysis
# Analiza distribuției categoriilor
print("\nCategory distribution:")
category_counts = df[' Category Label'].value_counts()
print(category_counts)

# Variables
X = df['Product Title']
y = df[' Category Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"The number of unique categories for training: {y_train.nunique()}")
print(f"The number of unique categories for test: {y_test.nunique()}")


# LinearSVC Pipeline
linear_svc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=2, max_features=10000)),
    ('clf', LinearSVC(random_state=42, max_iter=10000))
])

# Training the model
linear_svc_pipeline.fit(X_train, y_train)

# Trying predictions
y_pred = linear_svc_pipeline.predict(X_test)

# Evaluating the model
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
f1_score = report['weighted avg']['f1-score']

print(f"LinearSVC F1 Score: {f1_score:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))

# Training the final model on all data
print("\nTraining the final LinearSVC model on the entire dataset...")

final_model = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=2, max_features=10000)),
    ('clf', LinearSVC(random_state=42, max_iter=10000))
])

final_model.fit(X, y)

# Saving the model
joblib.dump(final_model, 'model/product_classifier.pkl')
print("Model saved as 'model/product_classifier.pkl'")
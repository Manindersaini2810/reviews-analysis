import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Loading the dataset
data = pd.read_csv('ChatGPT_Reviews.csv')

# Fill missing values
data['Review'] = data['Review'].fillna("")  # Replace missing reviews with empty strings
data['Ratings'] = data['Ratings'].fillna(3)  # Replace missing ratings with neutral (3)

# Define a function to convert ratings to sentiment
def rating_score(rating):
    if rating <= 2:
        return -1  # Negative
    elif rating > 3:
        return 1   # Positive
    else:
        return 0   # Neutral

# Apply the function for sentiment labels
data['Scores'] = data['Ratings'].apply(rating_score)

# Features (X) and labels (Y)
X = data['Review']
Y = data['Scores']

# Vectorize (convert text into numbers)
vectorizer = CountVectorizer(max_features=1000)  # Use the 1000 most common words
X_transformed = vectorizer.fit_transform(X)

#  Logistic Regression model
model = LogisticRegression(max_iter=500)

# Initialize StratifiedKFold for 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform 5-Fold Cross-Validation
accuracy_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(X_transformed, Y)):
    print(f"Fold {fold + 1}")
    
    # Split data into training and validation 
    X_train, X_val = X_transformed[train_index], X_transformed[test_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]
    
    # Training the model
    model.fit(X_train, Y_train)
    
    # Predict on the validation set
    Y_val_pred = model.predict(X_val)
    
    # Calculate and store accuracy
    accuracy = accuracy_score(Y_val, Y_val_pred)
    accuracy_scores.append(accuracy)
    
    # Print metrics for the current fold
    print(f"Accuracy for Fold {fold + 1}: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(Y_val, Y_val_pred))
    print("-" * 50)

# Calculate average accuracy among all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f"Average Accuracy across 5 folds: {average_accuracy:.4f}")

# Predict sentiment for  new user review
new_review = input("Enter your review: ")
new_review_transformed = vectorizer.transform([new_review])
prediction = model.predict(new_review_transformed)

if prediction[0] == 1:
    print("Sentiment: Positive")
elif prediction[0] == 0:
    print("Sentiment: Neutral")
else:
    print("Sentiment: Negative")


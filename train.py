# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load a sample dataset
def load_data():
    # Use sklearn's breast cancer dataset for demonstration
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Train a logistic regression model
def train_model():
    df = load_data()
    X = df.drop('target', axis=1)  # Features
    y = df['target']               # Target variable

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=10000)  # Set a high number of iterations to ensure convergence
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    train_model()

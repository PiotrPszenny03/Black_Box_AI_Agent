import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    print("Downloading German Credit Risk data...")

    # Fetch dataset ID 31 from OpenML (German Credit)
    credit_data = fetch_openml(data_id=31, as_frame=True, parser='auto')
    df = credit_data.frame

    # Convert target variable to 1 (Risk/Default) and 0 (Pays back/Good client)
    df['target'] = df['class'].map({'bad': 1, 'good': 0})
    df = df.drop(columns=['class'])

    print(f"Data downloaded. Size: {df.shape}")

    # Split into X (features) and y (target)
    X = df.drop(columns=['target'])
    y = df['target']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("\nTarget variable distribution (1=Default):")
    print(y_train.value_counts(normalize=True))

    # Save to CSV files so they're ready for modeling
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    print("\nData saved to CSV files. Done!")
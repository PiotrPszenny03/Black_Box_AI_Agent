import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

def train_and_save_model():
    print("Loading data from CSV files...")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')

    # squeeze() converts a single‑column DataFrame into a regular Series (required by models)
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()

    print("Preparing data...")
    # XGBoost needs to know which columns are text (categories) and which are numbers.
    # Convert text columns to 'category' type
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    print("Training the model (building a Black Box)...")
    # Initialize the model. Using built‑in categorical support (enable_categorical=True)
    model = XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        eval_metric='logloss',
        max_depth=4  # Limit depth to prevent overfitting
    )

    # Train the model
    model.fit(X_train, y_train)

    print("\nEvaluating the model (How well does it perform on new data?):")
    y_pred = model.predict(X_test)

    # Display the classification report.
    # We are interested in class 1 (Risk) and class 0 (Pays back)
    print(classification_report(y_test, y_pred))

    print("\nSaving the model...")
    # Save the trained model so we don't have to retrain every time
    joblib.dump(model, 'credit_model.pkl')
    print("Model ready and saved as 'credit_model.pkl'!")

if __name__ == "__main__":
    train_and_save_model()
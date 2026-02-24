import pandas as pd
import joblib
import shap
import numpy as np

def explain_customer():
    print("Loading model and test data...")
    # Load our black box
    model = joblib.load('credit_model.pkl')
    X_test = pd.read_csv('X_test.csv')

    # Restore category types (required by XGBoost)
    for col in X_test.select_dtypes(include=['object']).columns:
        X_test[col] = X_test[col].astype('category')

    # Select one specific customer from our test data (e.g., index 0)
    customer = X_test.iloc[[0]]

    print("\n--- CUSTOMER DATA ---")
    # Display the most important features of this customer
    print(customer[['duration', 'credit_amount', 'age', 'housing', 'employment']].T)

    print("\n--- MODEL DECISION (BLACK BOX) ---")
    # The model returns two probabilities: [Chance of repayment, Chance of default]
    # We are interested in the second one (index 1)
    risk = model.predict_proba(customer)[0][1] * 100
    print(f"Probability of default (Risk): {risk:.1f}%")

    print("\n--- OPENING THE BLACK BOX (SHAP) ---")
    # Initialize the explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(customer)

    # Extract the mathematical impact of each feature
    shap_values_array = shap_values.values[0]

    # Create a readable table
    feature_impact = pd.DataFrame({
        'Feature': customer.columns,
        'Customer_Value': customer.values[0],
        'Impact_on_Risk': shap_values_array
    })

    # Sort from the most important feature (using absolute value for sorting)
    feature_impact['Abs_Impact'] = np.abs(feature_impact['Impact_on_Risk'])
    feature_impact = feature_impact.sort_values(by='Abs_Impact', ascending=False).drop(columns=['Abs_Impact'])

    print("The most important factors that influenced this result:")
    # Show the top 5 factors
    print(feature_impact.head(5))

    # Save this data to a file to pass it to the AI Agent shortly
    feature_impact.head(5).to_csv('shap_reasons.csv', index=False)
    print("\nSaved decision reasons to file 'shap_reasons.csv'.")

if __name__ == "__main__":
    explain_customer()
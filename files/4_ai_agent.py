import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

def generate_report():
    # 1. Load API key from .env file (security!)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: No API key found! Check that you have a .env file and the GEMINI_API_KEY variable.")
        return

    # 2. Configure connection to AI
    genai.configure(api_key=api_key)
    # Use the fast and excellent model
    model = genai.GenerativeModel('gemini-2.5-flash')

    print("Loading data from the Black Box (SHAP)...")
    # 3. Load the results saved in the previous script
    shap_data = pd.read_csv('shap_reasons.csv')

    # Convert the table into text that the LLM can understand
    reasons = ""
    for index, row in shap_data.iterrows():
        feature = row['Feature']
        value = row['Customer_Value']
        impact = row['Impact_on_Risk']

        # If impact is positive, the feature increases risk (bad for the customer)
        direction = "Increases risk" if impact > 0 else "Decreases risk"
        reasons += f"- {feature}: {value} ({direction})\n"

    # 4. Create the prompt (instructions for the AI)
    prompt = f"""
    You are an AI assistant to a credit advisor at a bank.
    You have received the output from a Machine Learning (Black Box) model that assesses a customer's credit risk.

    Here are the 5 most important factors that influenced the algorithm's decision for this specific customer:
    {reasons}

    Your task is to write a short, professional, and analytical report for the bank advisor, so that they can explain the decision to the customer (in line with regulatory requirements for model explainability).

    The report should contain:
    1. The main conclusion (Why did the model assess the customer this way based on the given features?)
    2. An optimisation proposal (What can the advisor suggest to the customer to lower this risk and help them get credit in the future? E.g., shorter loan term, different amount, etc.).

    Write concisely, in professional financial jargon (e.g., creditworthiness, risk exposure). Do not explain what SHAP is.
    """

    print("Sending data to the AI Agent...\n")

    # 5. Generate response via API
    response = model.generate_content(prompt)

    print("==================================================")
    print("        AI REPORT FOR THE BANK ADVISOR            ")
    print("==================================================")
    print(response.text)
    print("==================================================")

if __name__ == "__main__":
    generate_report()
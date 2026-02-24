# Black_Box_AI_Agent
## 🎯 Project Overview
In the highly regulated banking sector, using "Black Box" Machine Learning models for credit scoring is a major compliance risk. Financial institutions must provide clear, actionable reasons for credit rejections (Explainable AI / XAI) to meet regulatory standards (e.g., EBA guidelines).

This project solves the "Black Box" problem by building an end-to-end pipeline: 
1. **Predictive Modeling:** An XGBoost model assesses the Probability of Default (PD).
2. **Explainability (XAI):** SHAP (SHapley Additive exPlanations) values decompose the mathematical decision into feature contributions.
3. **Generative AI:** A Large Language Model (Google Gemini API) acts as an AI Agent, translating complex SHAP data into a professional, human-readable advisory report for the credit officer.

## 🧠 Architecture & Tech Stack

* **Machine Learning Engine:** `Python`, `XGBoost`, `scikit-learn`, `pandas`
* **Explainable AI (XAI):** `SHAP` (TreeExplainer)
* **Generative AI Agent:** `google-generativeai` (Gemini 2.5 Flash), `python-dotenv`
* **Dataset:** German Credit Risk Dataset (OpenML)

## ⚙️ How the Pipeline Works

1.  **`1_data_loader.py`:** Fetches, cleans, and splits the raw credit data into training and testing sets.
2.  **`2_train_model.py`:** Trains a non-linear XGBoost Classifier to predict default risk and saves the model (`.pkl`).
3.  **`3_explain_shap.py`:** Loads the trained model and a test customer. It uses SHAP to calculate exactly *how much* each feature (e.g., age, loan duration, employment length) impacted the final risk score.
4.  **`4_ai_agent.py`:** Feeds the top SHAP values into an LLM via API. The AI generates a business-ready report containing the main reasons for the decision and risk-optimization steps the advisor can offer the client.

## 📊 Example Output (AI Agent)
*When the XGBoost model rejects a client, the AI Agent generates the following insight for the bank advisor:*

> **Main Conclusion:** The model assessed the client with an elevated credit risk exposure. While having savings (>=1000 PLN) is a positive factor, the primary drivers for the risk increase are:
> * **Employment Instability (< 1 year):** A strong predictor of potential income instability.
> * **Existing Bank Payment Plans:** Indicates current financial burden, limiting capacity for new debt.
> 
> **Risk Optimization Proposal for the Client:**
> * **Modify Loan Parameters:** Suggest lowering the requested loan amount or shortening the duration to reduce the bank's total exposure.
> * **Debt Consolidation:** Recommend verifying and potentially consolidating existing bank obligations to lower the monthly burden.

## 🚀 How to Run
1. Clone the repository and set up a virtual environment.
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost shap google-generativeai python-dotenv`
3. Create a `.env` file in the root directory and add your Google Gemini API key: `GEMINI_API_KEY=your_key_here`
4. Run the scripts sequentially from `1_data_loader.py` to `4_ai_agent.py`.

👤 Author Piotr Pszenny Aspiring Risk & Data Analyst

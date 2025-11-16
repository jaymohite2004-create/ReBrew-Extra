# ReBrew – Subscription Intelligence Dashboard

This Streamlit dashboard is built for **ReBrew** and runs easily on Streamlit Cloud.

It:
- Shows the **top 5 marketing insights** as clean, emoji-enhanced charts.
- Trains and compares **Decision Tree, Random Forest and Gradient Boosting** models
  on subscription interest (accuracy, precision, recall, F1, ROC AUC + single ROC plot).
- Includes an **AI Lab** with:
  - Classification comparison
  - Clustering (customer DNA)
  - Association rule mining
  - Linear, Ridge & Lasso regression comparison
- Provides a **Prediction Studio** where you can enter a new customer profile,
  see the predicted subscription label, and **download the full dataset with predictions**.

## How to deploy on Streamlit Cloud

1. Create a new GitHub repo.
2. Upload these four files directly to the root (no folders):
   - `app.py`
   - `ReBrew_Market_Survey_Synthetic_Data_600_responses.xlsx`
   - `requirements.txt`
   - `README.md`
3. On Streamlit Cloud, create a new app and point it to `app.py` on your `main` branch.
4. The dashboard will load automatically using the included dataset.

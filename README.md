# heart-disease-prediction
Heart disease prediction web app built with Python, scikit-learn (KNN) and Streamlit. Created by ROHIT ❤️
# ❤️ Heart Disease Prediction

A machine learning web app that predicts whether a patient is likely to have heart disease from 11 clinical features. Built with Python, scikit-learn and Streamlit.


## Features

- Clean web interface with a simple patient details form
- Instant Low Risk / High Risk prediction with model confidence
- Trained on 918 patient records and compared across 5 models (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM)
- Final model: K-Nearest Neighbors (about 86% test accuracy)

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `HeartDesasepredbyrohit.ipynb` | Data cleaning, EDA, model training and comparison |
| `KNN_heart.pkl` | Trained KNN model |
| `scaler.pkl` | Fitted StandardScaler |
| `columns.pkl` | Feature column order used in training |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit

## Disclaimer

This project is for learning and demonstration only. It is not a medical diagnosis. Always consult a qualified doctor.

---

Created by **ROHIT** with ❤️


# Bank Note Authentication API using Machine Learning and FastAPI

## Project Overview

This project is a simple Machine Learning API built using FastAPI for predicting whether a bank note is authentic or fake.

The model is trained using features extracted from bank note images:

- Variance
- Skewness
- Curtosis
- Entropy

The trained model is saved as a pickle file (`classifier.pkl`) and loaded into a FastAPI application for real-time predictions.

---

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- NumPy
- Pickle

---

## Project Structure

```text
Bank-Note-Authentication/
│
├── app.py
├── BankNotes.py
├── classifier.pkl
├── requirements.txt
└── README.md
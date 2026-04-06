import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

# Create the app object
app = FastAPI()
# Load the model
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

##Index route, opens automatically when we run the API
@app.get('/{name}')
def index(name: str):
    return {"Welcome to the Bank Note Authentication API!" : f'{name}'}

###Reroute for prediction, accepts POST requests with the BankNote model
@app.post('/predict')
def predict_bank_note(data: BankNote):
    # Convert the data to a numpy array
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0]>0.5:
        prediction = "The note is fake"
    else:
        prediction = "The note is authentic"
    # Return the prediction as a JSON response
    return {"prediction": prediction}

 

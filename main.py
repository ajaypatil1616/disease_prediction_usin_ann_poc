# import 
from fastapi import FastAPI, requests, HTTPException, Form, Depends, status
from fastapi_sqlalchemy import DBSessionMiddleware, db 
from sqlalchemy.orm import Session
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os 
from dotenv import load_dotenv
from datetime import datetime, timedelta
from random import randint
import tensorflow as tf
import numpy as np
import pandas as pd

import utility

app = FastAPI()


loaded_model = None

# when server start following automatically runs
@app.on_event("startup")
async def startup_event():
    global loaded_model
    
    if not os.path.exists("diease_prediction_model.keras"):
        utility.run_notebook("final_disease_prediction.ipynb")
    loaded_model = tf.keras.models.load_model("diease_prediction_model.keras")

## when is down following will run
@app.on_event("shutdown")
async def shutdown_event():
    global loaded_model
    loaded_model = None
    

## API for disease prediction
@app.post("/predict-disease")
async def prediction_of_disease(symptoms_list : List[str]):
    try:
        disease = utility.predict_disease(symptoms_list, loaded_model)
        precaution_list = utility.finding_precautions(disease)
        
        return JSONResponse({
            "disease" : disease,
            "precaution_list" : precaution_list
        })
    
    except Exception as e :
        raise HTTPException(status_code= status.HTTP_400_BAD_REQUEST, detail= str(e))


        
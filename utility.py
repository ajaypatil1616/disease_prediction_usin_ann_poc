# import
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pandasql import sqldf
import joblib

#utility functions 
def run_notebook(notebook_path):
    command = f"jupyter nbconvert --to notebook --execute {notebook_path} --output executed_notebook.ipynb"
    subprocess.run(command, shell=True, check= True)
    
def predict_disease(symptom_list, model):

    label_encoder = joblib.load('label_encoder_for_disease_prediction.pkl')
    weights_df = pd.read_csv('./dataset/processed/weights_df.csv')
    
    symptom_weights = []
    for symptom in symptom_list:
        symptom = symptom.strip().replace("_"," ")
        weight = weights_df[weights_df['Symptom'] == symptom]['weight'].values
        if weight.size > 0 :
            symptom_weights.append(weight[0])
        else:
            symptom_weights.append(0)
    
    input_vector = np.zeros(17)        
    for i, weight in enumerate(symptom_weights):
        if i < len(input_vector):
            input_vector[i] = weight
    input_vector = input_vector.reshape(1,-1)
    
    pred_probabilities = model.predict(input_vector)
    predicted_index = np.argmax(pred_probabilities, axis=1)
    predicted_disease = label_encoder.inverse_transform(predicted_index)
    return predicted_disease[0]

def finding_precautions(disease):
    precaution_df = pd.read_csv('./dataset/raw/symptom_precaution.csv')
    query = f""" SELECT precaution_df.precaution_1, precaution_df.precaution_2, precaution_df.precaution_3, precaution_df.precaution_4
    FROM precaution_df
    WHERE Disease = '{disease}' """
    result_df = sqldf(query, locals())
    result_list = result_df.loc[0,:].tolist()
    final_precaution_list = [precaution for precaution in result_list if precaution is not None]
    return final_precaution_list
    
    
    
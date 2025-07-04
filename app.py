from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the saved model
with open('clf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to call the model for prediction

def get_model_prediction(EngineHP, credit_history, Years_Experience, annual_claims, Miles_driven_annually, size_of_family, Gender, Marital_Status, Vehical_type, Age_bucket, State):
    X = [EngineHP, credit_history, Years_Experience, annual_claims, Miles_driven_annually, size_of_family, Gender,Marital_Status, Vehical_type, Age_bucket, State]
    print(f"RAW INPUT : {X}")
    X = np.array(X).reshape(1, -1)
    print(f"PROCESSED INPUT : {X}")
 
    loaded_model = joblib.load('clf_model.pkl')
    predictions = loaded_model.predict(X)
    
        
    prediction_proba = loaded_model.predict_proba(X)
    if predictions[0] == 0:
        predictions = predictions[0]
        prediction_proba = prediction_proba.tolist()[0][0]
    elif predictions[0] == 1:
        predictions = predictions[0]
        prediction_proba = prediction_proba.tolist()[0][1]
        

    return {"Prediction":predictions,"Prediction Probability":prediction_proba}
 

@app.route('/')
def index():
    return render_template('approval.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        EngineHP = float(request.form['EngineHP'])
        credit_history = int(request.form['credit_history'])
        Years_Experience = float(request.form['Years_Experience'])
        annual_claims = int(request.form['annual_claims'])
        Miles_driven_annually = float(request.form['Miles_driven_annually'])
        Gender = int(request.form['Gender'])
        size_of_family = int(request.form['size_of_family'])
        Marital_Status = int(request.form['Marital_Status'])
        Vehical_type = int(request.form['Vehical_type'])
        Age_bucket = int(request.form['Age_bucket'])
        State = int(request.form['State'])


        # Call the prediction function
        inference = get_model_prediction(EngineHP, credit_history, Years_Experience, annual_claims,
                                          Miles_driven_annually, size_of_family, Gender, 
                                          Marital_Status, Vehical_type, Age_bucket, State)
        """
        # If prediction is an ndarray, convert it to a list or scalar
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()  # Convert ndarray to list

        # If the list contains a single value, return that value as scalar
        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]  # Return single prediction as scalar
        """
            
        PredictionProbability=inference["Prediction Probability"]

        if inference["Prediction"] == 0:
            prediction = f"HIGH RISK CLAIM - Process Mannual Review!!!    PREDICTION CONFIDENCE : {'{:.2f}'.format(PredictionProbability*100)} %"
        elif inference["Prediction"] == 1:
            prediction = f"VALID CLAIM - No Deviations observed, please process the claim.    PREDICTION CONFIDENCE : {'{:.2f}'.format(PredictionProbability*100)} %"
        # Return the prediction as JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})

#print statement
@app.route('/print_state')
def print_state():
    State_map = { 
        'AK': 0, 'AL': 1, 'AR': 2, 'AZ': 3, 'CA': 4, 'CO': 5, 'CT': 6, 'DE': 7, 'FL': 8, 'GA': 9, 'HI': 10,
        'IA': 11, 'ID': 12, 'IL': 13, 'IN': 14, 'KS': 15, 'KY': 16, 'LA': 17, 'MA': 18, 'MD': 19, 'ME': 20,
        'MI': 21, 'MN': 22, 'MO': 23, 'MS': 24, 'MT': 25, 'NC': 26, 'ND': 27, 'NE': 28, 'NH': 29, 'NJ': 30,
        'NM': 31, 'NV': 32, 'NY': 33, 'OH': 34, 'OK': 35, 'OR': 36, 'PA': 37, 'RI': 38, 'SC': 39, 'SD': 40,
        'TN': 41, 'TX': 42, 'UT': 43, 'VA': 44, 'VT': 45, 'WA': 46, 'WI': 47, 'WV': 48, 'WY': 49
    }
    
    # Get the value for 'IN' (Indiana)
    in_value = State_map.get('IN')
    
    # Print and return the value for IN
    print(f"State value for IN (Indiana) is: {in_value}")
    
    return f"State value for IN (Indiana) is: {in_value}"

if __name__ == '__main__':
    app.run(debug=True)
    #print(get_model_prediction(120,768,16,1, 7058,1,2, 2, 3,4 , 20))

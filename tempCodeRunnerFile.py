from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# prediction function
def ValuePredictor(to_predict_list):
    loaded_model = pickle.load(open("random_forest_model.pkl", "rb"))
    
    # Ensure that the length of to_predict_list matches the expected number of features
    if len(to_predict_list) == 8:
        # Convert the list to a numpy array
        to_predict = np.array(to_predict_list).reshape(1, -1)
        
        # Perform prediction
        result = loaded_model.predict(to_predict)
        return result[0]
    else:
        # Handle an error or return an appropriate response
        return "Invalid input. Please provide values for all 8 features."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = list(request.form.values())
        to_predict_list = list(map(float, to_predict_list))
        prediction = ValuePredictor(to_predict_list)
        
        if int(prediction) == 1:
            result_text = 'High Risk'
        else:
            result_text = 'Low Risk'
        
        return render_template("result.html", prediction=result_text)

if __name__ == '__main__':
    app.run(debug=True)

import json
from flask import Flask, render_template, request, jsonify
from joblib import Parallel, delayed
import joblib
import time

app = Flask(__name__, template_folder='template')
model = joblib.load('data/Best_model.pkl')
        
# -------------------------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/predict", methods=["POST"])
def predict():
    start_at = time.time()
    input1 = list(request.form.values())[0]
    input2 = list(request.form.values())[1]
    input3 = list(request.form.values())[2]
    inputs = [[input1, input2, input3]]
    print(inputs)
    
    result = round(model.predict(inputs)[0], 2)
    print(result)
    end_at = time.time()
    return render_template('index2.html', result=result, x=input1, y=input2, z=input3 )
    
    
if __name__ == "__main__":
	app.run(debug=True, use_reloader=True)

import numpy as np
from flask import Flask, request, render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('webpage.html')
    #return "Welcome to webpage"

@app.route('/predict',methods=['POST'])
def predict():
    # for rendering result on HTML GUI
    
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
  
    output=round(prediction[0],2)
    return render_template('webpage.html', prediction_text='Employee Salary should be Rs. {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
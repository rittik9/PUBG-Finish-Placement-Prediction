import sklearn
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd



app=Flask(__name__)
##Loading the model
model=pickle.load(open('linearreg.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json
    for i in data.values():
        l =list(i.values())
    print(l)
    
    l=np.array(l)
    output=model.predict(l.reshape(1,-1))
    for i in output:
       x=i[0]

    return jsonify(x)


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=model.predict(final_input)
    for i in output:
       x=i[0]

    x=x*100
    return render_template("home.html",prediction_text="You got {}% chance of winning".format(x))


if __name__=="__main__":
    app.run(debug=False,use_reloader=False,host='0.0.0.0')

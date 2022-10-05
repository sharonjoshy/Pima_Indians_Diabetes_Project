

#import relevant libraries for flask, html rendering and loading the ML model

from flask import Flask,request,url_for,redirect,render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)  #flask app define cheyyunnu


model = joblib.load("model.pk")
scale = joblib.load("scale.pk")




@app.route("/")
def LandingPage():                                       # ith opening pagilek pokan ullathan
    return render_template('index.html')



@app.route("/predict",methods=['post'])
def predict():
    
    Pregnencies = request.form['1']
    Glucose = request.form['2']
    BloodPressure = request.form['3']
    SkinThickness = request.form['4']
    Insulin = request.form['5']
    BMI = request.form['6']
    DPF = request.form['7']
    Age = request.form['8']

    rowDF = pd.DataFrame([pd.Series([Pregnencies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age])])   #row vise aakan aan dataframe
    rowDF_New = pd.DataFrame(scale.transform(rowDF))   # scale pickle file vech input varunna values scale down aakum

    print(rowDF_New)

    # model prediction

    prediction = model.predict_proba(rowDF_New)

    print(f"The predicted value is {prediction[0][1]}")   # 0th listile first element it diabatic allathirikkanulla prediction aan




    if prediction[0][1] >= 0.5:         

        valpred = round(prediction[0][1],3)  #take 3 positions
        print(f"The round value{valpred*100}")

                              # next lineil pred kodukkan karnam reslt.html pageil 28th line il pred aayath  kond
        return render_template('result.html',pred=f'you have a chance of having diabetis. \n Probability of you being diabetic is {valpred*100}% \n Exercise regularly')

    else:
        valpred = round(prediction[0][0],3)  #take 3 positions
        print(f"The round value{valpred*100}")
        
        return render_template('result.html',pred=f'Congratulations you are safe. \n Probability of you being non diabetic is {valpred*100}% \n Exercise regularly')


    



    return render_template('index.html')




if __name__=='__main__':      # run cheyyumbozhan file run akunnath                           
    app.run(debug=True)







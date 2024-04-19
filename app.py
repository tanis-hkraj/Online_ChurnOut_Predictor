from flask import Flask,render_template, request
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    prediction=model.predict([[int(request.form.get('senior')),
                               int(request.form.get('depend')),
                               request.form.get('tenure'),
                               int(request.form.get('OnlineSecurity')),
                               int(request.form.get('OnlineBackup')),
                               int(request.form.get('Protection')),
                               int(request.form.get('TechSupport')),
                               int(request.form.get('Contract')),
                               int(request.form.get('Bill')),
                               request.form.get('Month'),
                               request.form.get('Total')]])
    if prediction[0]==0:
        output="would not"
    else:
        output="would"
    return render_template("index.html",prediction_text=f"With Respect to current Circumstances the customer {output} probably CHURNOUT.")
# if __name__=="__main__":
#     app.run(debug=True)  
    



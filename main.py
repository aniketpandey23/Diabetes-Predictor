from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

file = open('model.pkl', 'rb')
model = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Pregnancies=int(myDict['Number of Pregnancies'])
        Glucose=int(myDict['Glucose Level'])
        BloodPressure=int(myDict['Blood Pressure'])
        SkinThickness=int(myDict['Skin Thickness'])
        Insulin=int(myDict['Insulin Level'])
        BMI=float(myDict['Body Mass Index'])
        DiabetesPedigreeFunction=float(myDict['Diabetes Pedigree Function'])
        Age=int(myDict['Age'])
        inputFeatures = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        predictions = model.predict_proba(inputFeatures)[0][1]
        return render_template('show.html', prob=round((predictions * 100), 2))
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True)
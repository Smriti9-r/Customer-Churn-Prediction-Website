from flask import Flask, request, render_template_string, send_file
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("churn_model.pkl", "rb"))

with open("index.html", "r") as f:
    html_page = f.read()

@app.route("/style.css")
def style():
    return send_file("style.css")

@app.route("/")
def home():
    return render_template_string(html_page.replace("{{result}}",""))

@app.route("/predict", methods=["POST"])
def predict():

    try:

        tenure = float(request.form["tenure"])
        monthly = float(request.form["monthly"])
        total = float(request.form["total"])

        senior = int(request.form["senior"])
        partner = int(request.form["partner"])
        dependents = int(request.form["dependents"])
        paperless = int(request.form["paperless"])
        fiber = int(request.form["fiber"])
        twoyear = int(request.form["twoyear"])
        electronic = int(request.form["electronic"])

        features = np.zeros((1,23))

        features[0][0] = tenure
        features[0][1] = monthly
        features[0][2] = total
        features[0][3] = senior
        features[0][4] = partner
        features[0][5] = dependents
        features[0][6] = paperless
        features[0][7] = fiber
        features[0][8] = twoyear
        features[0][9] = electronic

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            result = f"""
            <div class='result high'>
            ⚠ High Churn Risk <br>
            Probability: {probability:.2%}
            <p>Customer may leave the service. Consider retention offers.</p>
            </div>
            """
        else:
            result = f"""
            <div class='result low'>
            ✅ Low Churn Risk <br>
            Probability: {(1-probability):.2%}
            <p>Customer likely satisfied with service.</p>
            </div>
            """

    except Exception as e:

        result = f"""
        <div class='result error'>
        Error: {str(e)}
        </div>
        """

    return render_template_string(html_page.replace("{{result}}", result))


if __name__ == "__main__":
    app.run(debug=True)
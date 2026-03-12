from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("sales_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    tv = float(request.form["tv"])
    radio = float(request.form["radio"])
    newspaper = float(request.form["newspaper"])

    features = np.array([[tv, radio, newspaper]])

    prediction = model.predict(features)

    # Safely convert model output (any array shape) to a single Python float
    pred_arr = np.asarray(prediction, dtype=float)
    if pred_arr.size == 0:
        result = 0.0
    else:
        result = float(pred_arr.reshape(-1)[0])

    return render_template(
        "index.html",
        prediction_text="Predicted Sales = {}".format(round(result,2))
    )

if __name__ == "__main__":
    app.run(debug=True)

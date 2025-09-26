from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ---------------- Load Model + Scaler ----------------
model = pickle.load(open("irrigation_model (1).pkl", "rb"))
scaler = pickle.load(open("irrigation_scaler.pkl", "rb"))

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])

        # Scale input
        input_data = scaler.transform([[temperature, humidity, moisture]])

        # Prediction
        
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
         # probability of irrigation needed

        if prediction == 0:
            result = f"✅ Irrigation Needed (Confidence: {proba:.2f})"
        else:
            result = f"💧 No Irrigation Needed (Confidence: {1-proba:.2f})"

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"⚠️ Error: {e}")

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)

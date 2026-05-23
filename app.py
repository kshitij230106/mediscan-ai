from flask import Flask, render_template, request, send_file
import requests
import joblib
import numpy as np
import pandas as pd
import sqlite3
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import jsonify

app = Flask(__name__)
# =========================
# LOAD DIABETES MODEL
# =========================
# =========================
# LOAD DIABETES PIPELINE
# =========================

diabetes_model = joblib.load(
    "models/diabetes_pipeline.pkl"
)

# =========================
# LOAD HEART MODEL
# =========================

heart_model = joblib.load(
    "models/heart_model.pkl"
)

heart_scaler = joblib.load(
    "models/heart_scaler.pkl"
)

# =========================
# LOAD KIDNEY MODEL
# =========================

kidney_model = joblib.load(
    "models/kidney_model.pkl"
)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/heart")
def heart_page():
    return render_template("heart.html")


@app.route("/kidney")
def kidney_page():
    return render_template("kidney.html")


@app.route("/history")
def history():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            disease_type,
            prediction,
            confidence,
            risk_level,
            health_score,
            created_at
        FROM prediction_history
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return render_template("history.html", rows=rows)


@app.route("/download_report")
def download_report():
    pdf_file = "health_report.pdf"
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI Healthcare Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT prediction, confidence, risk_level, health_score, created_at
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        elements.append(Paragraph(f"<b>Prediction:</b> {row[0]}", styles["BodyText"]))
        elements.append(Paragraph(f"<b>Confidence:</b> {row[1]}%", styles["BodyText"]))
        elements.append(Paragraph(f"<b>Risk Level:</b> {row[2]}", styles["BodyText"]))
        elements.append(Paragraph(f"<b>Health Score:</b> {row[3]}/100", styles["BodyText"]))
        elements.append(Paragraph(f"<b>Date:</b> {row[4]}", styles["BodyText"]))

    doc.build(elements)

    return send_file(pdf_file, as_attachment=True)

@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form["gender"]

    age = float(request.form["age"])

    hypertension = int(request.form["hypertension"])

    heart_disease = int(request.form["heart_disease"])

    smoking_history = request.form["smoking_history"]

    bmi = float(request.form["bmi"])

    hba1c = float(request.form["hba1c"])

    glucose = float(request.form["glucose"])

    input_df = pd.DataFrame([{

        "gender": gender,

        "age": age,

        "hypertension": hypertension,

        "heart_disease": heart_disease,

        "smoking_history": smoking_history,

        "bmi": bmi,

        "HbA1c_level": hba1c,

        "blood_glucose_level": glucose

    }])

    prediction = diabetes_model.predict(input_df)

    probability = diabetes_model.predict_proba(input_df)

    confidence = round(
        np.max(probability) * 100,
        2
    )

    # IMPORTANT:
    # CHANGE THIS IF YOUR LABELS ARE REVERSED

    if prediction[0] == 1:

        prediction_text = "⚠ High Risk of Diabetes"

        risk_level = "High"

    else:

        prediction_text = "✅ Low Risk of Diabetes"

        risk_level = "Low"

    health_score = round(100 - confidence)

    risk_factors = []

    if bmi > 30:
        risk_factors.append("High BMI detected")

    if glucose > 180:
        risk_factors.append("Very high glucose level")

    if hba1c > 6.5:
        risk_factors.append("Elevated HbA1c level")

    recommendations = [

        "Exercise regularly",

        "Maintain healthy diet",

        "Reduce sugar intake",

        "Monitor glucose levels",

    ]

    return render_template(

        "index.html",

        prediction_text=prediction_text,

        confidence=confidence,

        risk_level=risk_level,

        health_score=health_score,

        risk_factors=risk_factors,

        recommendations=recommendations,
    )

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = float(request.form["fbs"])
    restecg = float(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = float(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["slope"])
    ca = float(request.form["ca"])
    thal = float(request.form["thal"])

    features = [
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    ]

    input_array = np.array(features).reshape(1, -1)

    std_data = heart_scaler.transform(input_array)
    prediction = heart_model.predict(std_data)
    probability = heart_model.predict_proba(std_data)

    confidence = round(
        np.max(probability) * 100,
        2
    )

    # =========================
    # RISK LEVEL
    # =========================

    if prediction[0] == 0:
        if chol > 280 or trestbps > 160 or oldpeak > 3:
            risk_level = "🔴 High Risk"
        else:
            risk_level = "🟠 Moderate Risk"
    else:
        risk_level = "🟢 Low Risk"

    # =========================
    # RISK FACTORS
    # =========================

    risk_factors = []

    if chol > 240:
        risk_factors.append("High Cholesterol")

    if trestbps > 140:
        risk_factors.append("High Blood Pressure")

    if exang == 1:
        risk_factors.append("Exercise Induced Angina")

    if oldpeak > 2:
        risk_factors.append("Abnormal ECG Stress Response")

    if ca > 1:
        risk_factors.append("Blocked Major Vessels")

    # =========================
    # RECOMMENDATIONS
    # =========================

    recommendations = []

    if chol > 240:
        recommendations.append("Reduce oily and high cholesterol foods.")

    if trestbps > 140:
        recommendations.append("Monitor blood pressure regularly.")

    if oldpeak > 2:
        recommendations.append("Consult a cardiologist for further evaluation.")

    if exang == 1:
        recommendations.append("Avoid excessive physical stress and overexertion.")

    recommendations.append("Perform regular cardiovascular exercise.")
    recommendations.append("Maintain a healthy diet and body weight.")

    # =========================
    # PREDICTION RESULT
    # =========================

    if prediction[0] == 0:
        result = (
            f"⚠ Heart Disease Detected "
            f"({confidence}% confidence)"
        )
    else:
        result = (
            f"✅ No Heart Disease Detected "
            f"({confidence}% confidence)"
        )

    # =========================
    # SAVE TO DATABASE
    # =========================

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_history
        (disease_type, prediction, confidence, risk_level, health_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Heart Disease",
            result,
            confidence,
            risk_level,
            50
        ),
    )

    conn.commit()
    conn.close()

    # =========================
    # RENDER TEMPLATE
    # =========================

    return render_template(
        "heart.html",
        prediction_text=result,
        confidence=confidence,
        risk_level=risk_level,
        risk_factors=risk_factors,
        recommendations=recommendations
    )

@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():

    age = float(request.form["age"])
    bp = float(request.form["bp"])
    sg = float(request.form["sg"])
    al = float(request.form["al"])
    su = float(request.form["su"])

    features = [
        age,
        bp,
        sg,
        al,
        su
    ]

    input_array = np.array(features).reshape(1, -1)

    prediction = kidney_model.predict(input_array)
    probability = kidney_model.predict_proba(input_array)

    confidence = round(
        np.max(probability) * 100,
        2
    )

    # =========================
    # RISK LEVEL
    # =========================

    if prediction[0] == 0:
        if bp > 160 or al > 4 or su > 4:
            risk_level = "🔴 High Risk"
        else:
            risk_level = "🟠 Moderate Risk"
    else:
        risk_level = "🟢 Low Risk"

    # =========================
    # RISK FACTORS
    # =========================

    risk_factors = []

    if bp > 140:
        risk_factors.append(
            "High Blood Pressure"
        )

    if sg < 1.015:
        risk_factors.append(
            "Low Specific Gravity"
        )

    if al > 2:
        risk_factors.append(
            "High Albumin Level"
        )

    if su > 2:
        risk_factors.append(
            "High Sugar Level"
        )

    if age > 60:
        risk_factors.append(
            "Older Age Risk"
        )

    # =========================
    # RECOMMENDATIONS
    # =========================

    recommendations = []

    if bp > 140:
        recommendations.append(
            "Reduce salt intake and monitor blood pressure."
        )

    if su > 2:
        recommendations.append(
            "Control blood sugar levels properly."
        )

    if al > 2:
        recommendations.append(
            "Consult a nephrologist for kidney evaluation."
        )

    if sg < 1.015:
        recommendations.append(
            "Maintain proper hydration levels."
        )

    recommendations.append(
        "Drink sufficient water daily."
    )

    recommendations.append(
        "Avoid excessive processed and salty foods."
    )

    # =========================
    # PREDICTION RESULT
    # =========================

    if prediction[0] == 0:
        result = (
            f"⚠ Kidney Disease Detected "
            f"({confidence}% confidence)"
        )
    else:
        result = (
            f"✅ No Kidney Disease Detected "
            f"({confidence}% confidence)"
        )

    # =========================
    # SAVE TO DATABASE
    # =========================

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction_history
        (disease_type, prediction, confidence, risk_level, health_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Kidney Disease",
            result,
            confidence,
            risk_level,
            50
        ),
    )

    conn.commit()

    conn.close()

    # =========================
    # RENDER TEMPLATE
    # =========================

    return render_template(

        "kidney.html",

        prediction_text=result,

        confidence=confidence,

        risk_level=risk_level,

        risk_factors=risk_factors,

        recommendations=recommendations
    )



@app.route("/ask_ai", methods=["POST"])

def ask_ai():

    data = request.get_json()

    question = data["question"]

    prompt = f"""
    You are a healthcare AI assistant.

    Answer this question:

    {question}
    """

    response = requests.post(

        "http://localhost:11434/api/generate",

        json={

            "model": "phi3:mini",

            "prompt": prompt,

            "stream": False

        }

    )

    result = response.json()["response"]

    return jsonify({

        "answer": result

    })

    question = request.form["question"]

    prompt = f"""

    You are a helpful healthcare AI assistant.

    Give short and beginner-friendly
    healthcare advice.

    User question:
    {question}

    """

    try:

        response = requests.post(

            "http://localhost:11434/api/generate",

            json={

                "model": "phi3:mini",

                "prompt": prompt,

                "stream": False

            }

        )

        data = response.json()

        answer = data["response"]

    except Exception as e:

        answer = f"Error: {str(e)}"

    return render_template(

        "health_ai.html",

        answer=answer
    )

@app.route("/health_ai")
def health_ai():

    return render_template(
        "health_ai.html"
    )

if __name__ == "__main__":
    app.run(debug=True)



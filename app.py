import gradio as gr
import joblib
import pandas as pd

model = joblib.load("knn_diabetes_model.pkl")
scaler = joblib.load("knn_diabetes_scaler.pkl")

feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
        columns=feature_names
    )

    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0, 1]

    label = "Berisiko Diabetes" if probability >= 0.5 else "Tidak Berisiko Diabetes"
    return f"{label} (Probabilitas: {probability:.2%})"

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=6),
        gr.Number(label="Glucose", value=148),
        gr.Number(label="BloodPressure", value=72),
        gr.Number(label="SkinThickness", value=35),
        gr.Number(label="Insulin", value=0),
        gr.Number(label="BMI", value=33.6),
        gr.Number(label="DiabetesPedigreeFunction", value=0.627),
        gr.Number(label="Age", value=50),
    ],
    outputs=gr.Textbox(label="Hasil Prediksi"),
    title="Prediksi Diabetes Menggunakan KNN",
    description="Masukkan parameter pasien lalu klik Predict untuk melihat hasil prediksi."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

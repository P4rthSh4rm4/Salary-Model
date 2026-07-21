import os

import gradio as gr
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "salary_model.pkl"))
salary_data = pd.read_csv(os.path.join(BASE_DIR, "Salary.csv"))

GENDERS = sorted(salary_data["Gender"].dropna().unique().tolist())
EDUCATION_LEVELS = sorted(salary_data["Education Level"].dropna().unique().tolist())
JOB_TITLES = sorted(salary_data["Job Title"].dropna().unique().tolist())


def predict_salary(age, gender, education, job_title, experience):
    values = {column: 0 for column in model.feature_names_in_}
    values["Age"] = age
    values["Years of Experience"] = experience

    for column in (
        f"Gender_{gender}",
        f"Education Level_{education}",
        f"Job Title_{job_title}",
    ):
        if column in values:
            values[column] = 1

    features = pd.DataFrame([values], columns=model.feature_names_in_)
    prediction = model.predict(features)[0]
    return f"Estimated salary: {prediction:,.2f}"


interface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="Age", precision=0, value=30),
        gr.Dropdown(GENDERS, label="Gender", value=GENDERS[0]),
        gr.Dropdown(EDUCATION_LEVELS, label="Education level", value=EDUCATION_LEVELS[0]),
        gr.Dropdown(JOB_TITLES, label="Job title", value=JOB_TITLES[0]),
        gr.Number(label="Years of experience", value=5),
    ],
    outputs=gr.Text(label="Prediction"),
    title="Salary Predictor",
    description="Enter career details to estimate annual salary.",
)


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

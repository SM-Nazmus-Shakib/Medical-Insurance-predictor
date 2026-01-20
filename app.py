import pickle
import pandas as pd
import gradio as gr

with open("medical_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_insurance(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(input_df)
    return f"Predicted Insurance Charge: ${prediction[0]:.2f}"

app = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Radio(
            ["southwest", "southeast", "northwest", "northeast"],
            label="Region"
        )
    ],
    outputs="text",
    title="Medical Insurance Cost Predictor",
    description="Predict medical insurance charge"
)

app.launch()

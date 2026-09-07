from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\singh\Desktop\Education\Invoice Intelligence ML Project\models\predict_freight_model.pkl"


def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        return joblib.load(f)


def predict_freight_cost(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Freight"] = model.predict(input_df).round()

    return input_df


if __name__ == "__main__":
    sample_data = {
        "Dollars": [18500, 9000,3000, 200]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)
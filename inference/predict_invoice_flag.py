import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\singh\Desktop\Education\Invoice Intelligence ML Project\models\predict_flag_invoice.pkl"


def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        return joblib.load(f)


def predict_invoice_flag(input_data):
    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df)

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [100, 50, 25, 10],
        "invoice_dollars": [18500, 9000, 3000, 200],
        "Freight": [500, 300, 150, 50],
        "total_items_quantity": [100, 50, 25, 10],
        "total_items_dollars": [18400, 8900, 2950, 200]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)
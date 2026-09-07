import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Invoice Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    r"C:\Users\singh\Desktop\Education\Invoice Intelligence ML Project"
)

MODEL_DIR = BASE_DIR / "models"

INVOICE_MODEL_PATH = MODEL_DIR / "predict_flag_invoice.pkl"
INVOICE_SCALER_PATH = MODEL_DIR / "scaler.pkl"
FREIGHT_MODEL_PATH = MODEL_DIR / "predict_freight_model.pkl"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0, 255, 255, 0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(140, 0, 255, 0.10), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(0, 140, 255, 0.07), transparent 30%),
        #05070d;
    color: #f5f7ff;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #080b14 0%,
        #05070d 100%
    );
    border-right: 1px solid rgba(0, 255, 255, 0.15);
}

h1, h2, h3 {
    color: #ffffff !important;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
    background: linear-gradient(
        90deg,
        #00ffff,
        #ffffff,
        #9d4edd,
        #00ffff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow:
        0 0 15px rgba(0,255,255,0.25),
        0 0 35px rgba(157,78,221,0.20);
}

.subtitle {
    text-align: center;
    color: #9da6bd;
    font-size: 17px;
    margin-bottom: 35px;
}

.model-card {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(
        145deg,
        rgba(15, 22, 38, 0.95),
        rgba(8, 12, 23, 0.95)
    );
    border: 1px solid rgba(0,255,255,0.16);
    box-shadow:
        0 0 20px rgba(0,255,255,0.05),
        inset 0 0 25px rgba(255,255,255,0.015);
    margin-bottom: 25px;
}

.info-box {
    padding: 22px;
    border-radius: 16px;
    background: rgba(10, 17, 30, 0.90);
    border-left: 4px solid #00ffff;
    box-shadow:
        0 0 18px rgba(0,255,255,0.10);
    margin-bottom: 25px;
}

.result-card {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    background:
        linear-gradient(
            145deg,
            rgba(12, 19, 35, 0.98),
            rgba(5, 9, 18, 0.98)
        );
    border: 1px solid rgba(0,255,255,0.25);
    box-shadow:
        0 0 25px rgba(0,255,255,0.10),
        0 0 60px rgba(157,78,221,0.06);
    margin-top: 25px;
    margin-bottom: 25px;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    margin-top: 10px;
}

.result-label {
    color: #9da6bd;
    font-size: 15px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(
        90deg,
        #00bcd4,
        #7b2cbf
    );
    color: white;
    font-weight: 700;
    border: 1px solid rgba(0,255,255,0.35);
    padding: 12px;
    box-shadow:
        0 0 15px rgba(0,255,255,0.15);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 0 25px rgba(0,255,255,0.35),
        0 0 35px rgba(123,44,191,0.25);
}

div[data-baseweb="input"] {
    background-color: #0b1020 !important;
    border-radius: 10px;
}

div[data-baseweb="select"] {
    background-color: #0b1020 !important;
    border-radius: 10px;
}

div[data-testid="stMetric"] {
    background: rgba(12, 18, 32, 0.8);
    border: 1px solid rgba(0,255,255,0.12);
    padding: 15px;
    border-radius: 14px;
    box-shadow:
        0 0 15px rgba(0,255,255,0.05);
}

.footer {
    text-align: center;
    color: #687188;
    margin-top: 50px;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_invoice_model():
    return joblib.load(INVOICE_MODEL_PATH)


@st.cache_resource
def load_invoice_scaler():
    return joblib.load(INVOICE_SCALER_PATH)


@st.cache_resource
def load_freight_model():
    return joblib.load(FREIGHT_MODEL_PATH)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 INVOICE INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered financial intelligence for smarter invoice decisions'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚡ AI MODEL")


    st.markdown("---")

    st.markdown("### 📌 Project")

    st.markdown(
        """
        **Invoice Intelligence**

        This application uses Machine Learning
        to analyze invoice-related financial data
        and generate intelligent predictions.
        """
    )

    st.markdown("---")

    st.caption("Machine Learning • Streamlit • Python")

# =========================================================
# MODEL SELECTION
# =========================================================

st.markdown("### ⚡ Choose AI Model")

model_choice = st.radio(
    "Select the model you want to use:",
    [
        "🚩 Invoice Flagging",
        "🚚 Freight Cost Prediction"
    ],
    horizontal=True
)

# =========================================================
# INVOICE FLAGGING
# =========================================================

if model_choice == "🚩 Invoice Flagging":

    st.markdown(
        '<div class="model-card">'
        '<h2>🚩 Invoice Flagging Model</h2>'
        '<p style="color:#aeb7cc;">'
        'Identify invoices that may require additional review.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # INFORMATION
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="info-box">

        <h3>📖 What does this model do?</h3>

        <p>
        The Invoice Flagging model is a
        <b>classification model</b> that predicts whether
        an invoice should be flagged for further review.
        </p>

        <h4>🎯 Why is it important?</h4>

        <p>
        It can help businesses identify unusual invoice
        patterns, amount mismatches and operational delays
        that may require manual investigation.
        </p>

        <h4>📝 How to use it?</h4>

        <p>
        Enter the invoice and purchase-related information
        below and click <b>Analyze Invoice</b>.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    st.markdown("### 📊 Invoice Information")

    col1, col2 = st.columns(2)

    with col1:

        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        invoice_dollars = st.number_input(
            "Invoice Dollars",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

        freight = st.number_input(
            "Freight",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        total_brands = st.number_input(
            "Total Brands",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    with col2:

        total_items_quantity = st.number_input(
            "Total Item Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        days_po_to_invoice = st.number_input(
            "Days PO → Invoice",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

        total_items_dollars = st.number_input(
            "Total Items Dollars",
            min_value=0.0,
            value=9800.0,
            step=100.0
        )

        avg_receiving_delay = st.number_input(
            "Average Receiving Delay",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    st.markdown("")

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button("🚀 ANALYZE INVOICE"):

        try:

            model = load_invoice_model()
            scaler = load_invoice_scaler()

            # EXACT SAME ORDER AS TRAINING
            input_data = pd.DataFrame({
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_brands": [total_brands],
                "total_items_quantity": [total_items_quantity],
                "days_po_to_invoice": [days_po_to_invoice],
                "total_items_dollars": [total_items_dollars],
                "avg_receiving_delay": [avg_receiving_delay]
            })

            # SCALE INPUT
            input_scaled = scaler.transform(input_data)

            # PREDICT USING SCALED DATA
            prediction = model.predict(input_scaled)[0]

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            if prediction == 1:

                st.markdown(
                    """
                    <div class="result-card">

                    <div style="font-size:55px;">🚨</div>

                    <div class="result-label">
                    Invoice Status
                    </div>

                    <div class="result-value">
                    FLAGGED
                    </div>

                    <p style="color:#aeb7cc;">
                    This invoice may require additional review.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.warning(
                    "⚠️ Please review the invoice for unusual "
                    "financial or operational patterns."
                )

            else:

                st.markdown(
                    """
                    <div class="result-card">

                    <div style="font-size:55px;">✅</div>

                    <div class="result-label">
                    Invoice Status
                    </div>

                    <div class="result-value">
                    NORMAL
                    </div>

                    <p style="color:#aeb7cc;">
                    No significant issue was detected by the model.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.success(
                    "✅ Invoice appears normal according to the model."
                )

            # -------------------------------------------------
            # INPUT SUMMARY
            # -------------------------------------------------

            with st.expander("📋 View Input Summary"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# =========================================================
# FREIGHT COST PREDICTION
# =========================================================

else:

    st.markdown(
        '<div class="model-card">'
        '<h2>🚚 Freight Cost Prediction</h2>'
        '<p style="color:#aeb7cc;">'
        'Estimate the expected freight cost for an invoice.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # INFORMATION
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="info-box">

        <h3>📖 What does this model do?</h3>

        <p>
        The Freight Cost Prediction model is a
        <b>regression model</b> that predicts the expected
        freight cost using invoice quantity and dollar value.
        </p>

        <h4>🎯 Why is it important?</h4>

        <p>
        It can help businesses estimate reasonable freight
        expenses and identify potentially unusual shipping costs.
        </p>

        <h4>📝 How to use it?</h4>

        <p>
        Enter the quantity and dollar value of the invoice,
        then click <b>Predict Freight Cost</b>.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    st.markdown("### 📦 Order Information")

    col1, col2 = st.columns(2)

    with col1:

        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    with col2:

        dollars = st.number_input(
            "Dollars",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

    st.markdown("")

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button("🚀 PREDICT FREIGHT COST"):

        try:

            model = load_freight_model()

            input_data = pd.DataFrame({
            "Quantity": [quantity],
            "Dollars": [dollars]
        })

            # FREIGHT MODEL WAS NOT SCALED DURING TRAINING
            prediction = model.predict(input_data)[0]

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">

                <div style="font-size:55px;">🚚</div>

                <div class="result-label">
                Predicted Freight Cost
                </div>

                <div class="result-value">
                $ {prediction:,.2f}
                </div>

                <p style="color:#aeb7cc;">
                Estimated freight cost based on the
                provided invoice information.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------------------------
            # INPUT SUMMARY
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "📦 Quantity",
                    f"{quantity:,.0f}"
                )

            with col2:

                st.metric(
                    "💰 Invoice Value",
                    f"$ {dollars:,.2f}"
                )

            with st.expander("📋 View Input Summary"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🧠 <b>Invoice Intelligence</b>

    <br>

    Machine Learning powered financial analysis

    <br><br>

    Built with Python • Scikit-learn • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
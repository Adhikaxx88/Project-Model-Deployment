import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"

# ── Constants ─────────────────────────────────────────────────────────────────
MONTHS = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
]
OCCUPATIONS = sorted([
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Media_Manager",
    "Mechanic", "Musician", "Scientist", "Teacher", "Writer", "Unknown",
])
CREDIT_MIX = ["Good", "Standard", "Bad", "Unknown"]
PAYMENT_MIN = ["Yes", "No", "Unknown"]
PAYMENT_BEHAVIOUR = [
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
    "Unknown",
]
LOAN_TYPES = [
    "Student Loan", "Home Equity Loan", "Mortgage Loan",
    "Debt Consolidation Loan", "Credit-Builder Loan",
    "Payday Loan", "Personal Loan", "Auto Loan",
]
SCORE_COLOR = {"Good": "#28a745", "Standard": "#fd7e14", "Poor": "#dc3545"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Section card */
.section-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid #e9ecef;
}
.section-title {
    font-size: 15px;
    font-weight: 700;
    color: #495057;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}
/* Result banner */
.result-banner {
    text-align: center;
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 20px;
}
/* Probability card */
.prob-card {
    text-align: center;
    padding: 18px;
    border-radius: 10px;
    margin: 4px;
}
/* Sidebar label */
.sidebar-label {
    font-size: 12px;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Credit Score Predictor")
    st.markdown("Prediksi credit score nasabah berdasarkan profil keuangan mereka.")
    st.divider()

    with st.expander("ℹ️ Info Model"):
        st.markdown(
            "**Model:** Random Forest  \n"
            "**Target:** Credit Score  \n"
            "**Kelas:** Good | Standard | Poor  \n"
            "**Metrik:** Macro F1 ≈ 0.716"
        )

    st.divider()
    st.markdown('<p class="sidebar-label">Panduan Pengisian</p>', unsafe_allow_html=True)
    st.markdown(
        "1. Isi semua informasi nasabah\n"
        "2. Pilih jenis pinjaman yang dimiliki\n"
        "3. Klik **Predict Credit Score**\n"
        "4. Lihat hasil dan probabilitas tiap kelas"
    )
    st.divider()
    st.caption("Model Deployment Project · 1C")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💳 Credit Score Predictor")
st.markdown("Isi profil keuangan nasabah di bawah untuk memprediksi credit score mereka.")
st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
def gauge_chart(value: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": label, "font": {"size": 15, "color": "#495057"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#dee2e6"},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#fff0f0"},
                {"range": [40, 70], "color": "#fff8e1"},
                {"range": [70, 100], "color": "#f0fff4"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.78,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=60, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def prob_bar_chart(classes: list, probs: list) -> go.Figure:
    colors = [SCORE_COLOR.get(c, "#6c757d") for c in classes]
    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=classes,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="inside",
        insidetextanchor="middle",
        textfont={"color": "white", "size": 13},
    ))
    fig.update_layout(
        title={"text": "Probabilitas per Kelas", "font": {"size": 14, "color": "#495057"}},
        xaxis={"range": [0, 100], "title": "Probabilitas (%)", "showgrid": True, "gridcolor": "#e9ecef"},
        yaxis={"categoryorder": "total ascending"},
        height=220,
        margin=dict(t=45, b=20, l=10, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # Personal info
    st.markdown('<div class="section-title">👤 Informasi Personal</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.selectbox("Month", MONTHS)
        age   = st.number_input("Age", min_value=14, max_value=100, value=30)
    with c2:
        occupation            = st.selectbox("Occupation", OCCUPATIONS)
        monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=3000.0, step=100.0)
    with c3:
        ch_years  = st.number_input("Credit History — Years",  min_value=0, max_value=40, value=5)
        ch_months = st.number_input("Credit History — Months", min_value=0, max_value=11, value=0)
    credit_history_months = int(ch_years) * 12 + int(ch_months)

    st.divider()

    # Account info
    st.markdown('<div class="section-title">🏦 Informasi Rekening</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=10, value=3)
        num_credit_card   = st.number_input("Num Credit Cards",  min_value=0, max_value=10, value=4)
    with c2:
        interest_rate = st.number_input("Interest Rate (%)", min_value=1, max_value=32, value=13)
        num_of_loan   = st.number_input("Num of Loans",      min_value=0, max_value=9,  value=3)
    with c3:
        delay_from_due_date    = st.number_input("Delay from Due Date (days)", min_value=0, max_value=67, value=15)
        num_of_delayed_payment = st.number_input("Num Delayed Payments",       min_value=0, max_value=27, value=10)
    with c4:
        changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, max_value=36.3, value=9.0, step=0.1)
        num_credit_inquiries = st.number_input("Num Credit Inquiries",  min_value=0,   max_value=16,   value=4)

    st.divider()

    # Financial info
    st.markdown('<div class="section-title">💰 Informasi Keuangan</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_mix       = st.selectbox("Credit Mix",    CREDIT_MIX)
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=800.0, step=50.0)
    with c2:
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=28.0, step=0.5)
        total_emi_per_month      = st.number_input("Total EMI per Month",           min_value=0.0, value=120.0, step=10.0)
    with c3:
        amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=300.0, step=50.0)
        payment_of_min_amount   = st.selectbox("Payment of Min Amount", PAYMENT_MIN)

    payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR)

    st.divider()

    # Loan types
    st.markdown('<div class="section-title">🏷️ Jenis Pinjaman</div>', unsafe_allow_html=True)
    selected_loans = st.multiselect(
        "Pilih semua jenis pinjaman yang dimiliki (kosongkan jika tidak ada)",
        LOAN_TYPES,
    )

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Predict Credit Score",
        use_container_width=True,
        type="primary",
    )

# ── Inference & Result ────────────────────────────────────────────────────────
if submitted:
    loan_cols = {
        f"Loan_{loan.replace(' ', '_').replace('-', '_')}": (1 if loan in selected_loans else 0)
        for loan in LOAN_TYPES
    }

    input_data = {
        "Month":                    month,
        "Age":                      age,
        "Occupation":               occupation,
        "Monthly_Inhand_Salary":    monthly_inhand_salary,
        "Num_Bank_Accounts":        num_bank_accounts,
        "Num_Credit_Card":          num_credit_card,
        "Interest_Rate":            interest_rate,
        "Num_of_Loan":              num_of_loan,
        "Delay_from_due_date":      delay_from_due_date,
        "Num_of_Delayed_Payment":   num_of_delayed_payment,
        "Changed_Credit_Limit":     changed_credit_limit,
        "Num_Credit_Inquiries":     num_credit_inquiries,
        "Credit_Mix":               credit_mix,
        "Outstanding_Debt":         outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Payment_of_Min_Amount":    payment_of_min_amount,
        "Total_EMI_per_month":      total_emi_per_month,
        "Amount_invested_monthly":  amount_invested_monthly,
        "Payment_Behaviour":        payment_behaviour,
        "Credit_History_Months":    credit_history_months,
        **loan_cols,
    }

    try:
        response = httpx.post(API_URL, json=input_data, timeout=10.0)
        response.raise_for_status()
    except httpx.RequestError:
        st.error("Tidak dapat terhubung ke API. Pastikan FastAPI server berjalan di `http://localhost:8000`.")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API mengembalikan error: {e.response.status_code} - {e.response.text}")
        st.stop()

    result         = response.json()
    label          = result["label"]
    probabilities  = result["probabilities"]
    classes        = result["classes"]
    pred_proba     = [probabilities[c] for c in classes]

    color    = SCORE_COLOR.get(label, "#6c757d")
    top_prob = max(pred_proba) * 100

    df_input = pd.DataFrame([input_data])

    st.divider()
    st.subheader("Hasil Prediksi")

    # ── Result banner
    icon = {"Good": "✅", "Standard": "⚠️", "Poor": "❌"}.get(label, "")
    desc = {
        "Good":     "Nasabah ini memiliki profil kredit yang **sangat baik**.",
        "Standard": "Nasabah ini memiliki profil kredit yang **cukup baik**.",
        "Poor":     "Nasabah ini memiliki profil kredit yang **perlu diperbaiki**.",
    }.get(label, "")

    st.markdown(
        f"""
        <div class="result-banner" style="background:{color}18; border:2px solid {color};">
            <div style="font-size:48px; margin-bottom:4px;">{icon}</div>
            <h1 style="color:{color}; margin:0; font-size:36px;">Credit Score: {label}</h1>
            <p style="color:#495057; margin-top:8px; font-size:15px;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3-column layout: metrics | gauge | bar chart
    col_metric, col_gauge, col_bar = st.columns([1, 1.4, 1.6])

    with col_metric:
        st.markdown("**Probabilitas per Kelas**")
        for cls, prob in sorted(zip(classes, pred_proba), key=lambda x: -x[1]):
            delta_symbol = "▲" if cls == label else ""
            st.metric(
                label=f"{cls} {delta_symbol}",
                value=f"{prob*100:.1f}%",
            )

    with col_gauge:
        st.plotly_chart(
            gauge_chart(top_prob, f"Confidence: {label}", color),
            use_container_width=True,
        )

    with col_bar:
        st.plotly_chart(
            prob_bar_chart(classes, pred_proba),
            use_container_width=True,
        )

    # ── Input summary
    with st.expander("🔎 Lihat data input yang dikirim ke model"):
        st.dataframe(df_input.T.rename(columns={0: "Value"}), use_container_width=True)

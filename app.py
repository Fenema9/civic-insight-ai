import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os

st.set_page_config(page_title="Civic Insight AI", layout="wide")

st.title("🏙 Civic Insight AI")
st.markdown("""
AI-powered civic intelligence platform that transforms open city data into actionable insights 
for decision-makers and residents.
""")
# Load data
@st.cache_data

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    df["Created Date"] = pd.to_datetime(df["Created Date"], errors="coerce")
    df["Closed Date"] = pd.to_datetime(df["Closed Date"], errors="coerce")
    df["resolution_hours"] = (
        df["Closed Date"] - df["Created Date"]
    ).dt.total_seconds() / 3600
    return df

uploaded_file = st.file_uploader("Upload 311 CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.warning("Please upload a CSV file to start analysis.")
    st.stop()
# Sidebar filters
st.sidebar.header("Filters")

boroughs = df["Borough"].dropna().unique()
selected_borough = st.sidebar.selectbox(
    "Select Borough",
    ["All"] + sorted(boroughs)
)

if selected_borough != "All":
    df = df[df["Borough"] == selected_borough]
# Metrics
col1, col2 = st.columns(2)

col1.metric("Total Requests", len(df))
col2.metric("Avg Resolution Time (hrs)", round(df["resolution_hours"].mean(), 2))

st.divider()

# Top Problems
top_problems = (
    df["Problem (formerly Complaint Type)"]
    .value_counts()
    .head(10)
    .reset_index()
)

fig1 = px.bar(
    top_problems,
    x="count",
    y="Problem (formerly Complaint Type)",
    orientation="h",
    title="Top 10 Service Request Types"
)

st.plotly_chart(fig1, use_container_width=True)

# Resolution by Borough
avg_resolution = (
    df.groupby("Borough")["resolution_hours"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    avg_resolution,
    x="resolution_hours",
    y="Borough",
    orientation="h",
    title="Average Resolution Time by Borough (Hours)"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# AI Section
st.header("🤖 AI Civic Insights")

if st.button("Generate AI Insights"):

    summary_data = f"""
    Total requests: {len(df)}

    Average resolution time (hours): {df["resolution_hours"].mean():.2f}

    Top problem types:
    {df["Problem (formerly Complaint Type)"].value_counts().head(5).to_string()}

    Requests by borough:
    {df["Borough"].value_counts().to_string()}

    Average resolution time by borough:
    {df.groupby("Borough")["resolution_hours"].mean().round(2).to_string()}
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a civic data analyst."},
            {"role": "user", "content": f"""
            Based on the following NYC 311 data summary:

            {summary_data}

            Provide:
            1. Executive summary
            2. Key patterns
            3. Potential operational risks
            4. Actionable recommendations for city management

            Keep it concise and professional.
            """}
        ],
        max_tokens=400
    )

    st.write(response.choices[0].message.content)
    st.divider()
st.subheader("📈 Requests Over Time")

requests_per_day = (
    df.groupby(df["Created Date"].dt.date)
    .size()
    .reset_index(name="count")
)

fig3 = px.line(
    requests_per_day,
    x="Created Date",
    y="count",
    title="Daily Service Request Volume"
)

st.plotly_chart(fig3, use_container_width=True)
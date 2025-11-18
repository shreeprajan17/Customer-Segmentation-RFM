import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Segmentation | RFM Analysis",
    page_icon="🛒",
    layout="wide"
)

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_model():
    try:
        # Load the file we saved in the previous step
        artifacts = joblib.load("rfm_kmeans_model.joblib")
        return artifacts
    except FileNotFoundError:
        st.error("File 'rfm_kmeans_model.joblib' not found. Please run the extraction code first.")
        return None

data_artifacts = load_model()

if data_artifacts:
    scaler = data_artifacts["scaler"]
    kmeans_model = data_artifacts["kmeans_model"]
    df = data_artifacts["final_data"]
    cluster_descriptions = data_artifacts["cluster_descriptions"]

    # --- SIDEBAR ---
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to:", ["Dashboard Overview", "Cluster Analysis", "Predict New Customer"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Project Info")
    st.sidebar.info(
        """
        **Algorithm:** K-Means Clustering
        **Data:** Online Retail II (UCI)
        **Technique:** RFM (Recency, Frequency, Monetary)
        """
    )
    # Placeholder for your links
    st.sidebar.markdown("[GitHub Repo Link](#)") 
    st.sidebar.markdown("[LinkedIn Profile](#)")

    # --- TAB 1: DASHBOARD OVERVIEW ---
    if options == "Dashboard Overview":
        st.title("📊 Customer Segmentation Dashboard")
        st.markdown("### Strategic insights based on purchasing behavior")

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Avg Monetary Value", f"£{df['MonetaryValue'].mean():.2f}")
        col3.metric("Avg Frequency", f"{df['Frequency'].mean():.1f} purchases")

        st.markdown("---")

        # 3D Scatter Plot (Interactive)
        st.subheader("3D Cluster Visualization")
        st.markdown("Rotate the graph to explore how customers are grouped based on RFM.")
        
        fig_3d = px.scatter_3d(
            df, 
            x='MonetaryValue', 
            y='Frequency', 
            z='Recency',
            color='ClusterLabel',
            hover_data=['Customer ID'],
            opacity=0.7,
            height=700,
            title="Customer Segments (3D View)"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAB 2: CLUSTER ANALYSIS ---
    elif options == "Cluster Analysis":
        st.title("🧐 Detailed Cluster Analysis")
        
        # Cluster Distribution
        st.subheader("Customer Count by Segment")
        count_data = df['ClusterLabel'].value_counts().reset_index()
        count_data.columns = ['Segment', 'Count']
        
        fig_bar = px.bar(count_data, x='Segment', y='Count', color='Segment', text='Count')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Segment Characteristics")
        # Show statistical summary of clusters
        summary = df.groupby('ClusterLabel')[['Recency', 'Frequency', 'MonetaryValue']].mean().reset_index()
        st.dataframe(
            summary.style.format({
                'Recency': '{:.2f}', 
                'Frequency': '{:.2f}', 
                'MonetaryValue': '{:.2f}'
            }), 
            use_container_width=True
        )

        st.markdown("### Strategy Guide")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**🟦 RETAIN (Cluster 0):** Regular buyers. Keep them engaged with loyalty programs.")
            st.warning("**🟧 RE-ENGAGE (Cluster 1):** Infrequent, low spenders. Use discounts to win them back.")
            st.success("**🟩 NURTURE (Cluster 2):** New or recent buyers. Provide excellent support.")
        
        with col2:
            st.error("**🟥 REWARD (Cluster 3):** Top loyalists. Give exclusive VIP access.")
            st.markdown("**🟣 Outliers (Pamper/Upsell/Delight):** Extreme high values requiring manual attention.")

    # --- TAB 3: PREDICTION ---
    elif options == "Predict New Customer":
        st.title("🔮 Predict Customer Segment")
        st.markdown("Enter the RFM values for a new or existing customer to classify them.")

        # Input Form
        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            
            with c1:
                monetary_input = st.number_input("Monetary Value (Total Spend)", min_value=0.0, value=1000.0)
            with c2:
                frequency_input = st.number_input("Frequency (Number of Transactions)", min_value=1, value=5)
            with c3:
                recency_input = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)

            submit_button = st.form_submit_button("Predict Segment")

        if submit_button:
            # 1. Preprocess input (Standard Scaling)
            # Note: The model expects [[Monetary, Frequency, Recency]] based on your notebook In[36]
            input_data = [[monetary_input, frequency_input, recency_input]]
            scaled_input = scaler.transform(input_data)

            # 2. Predict
            predicted_cluster_id = kmeans_model.predict(scaled_input)[0]

            # 3. Map ID to Label
            # Note: The KMeans model only predicts 0, 1, 2, 3. 
            # Outliers (-1, -2, -3) were manual rules in your notebook.
            # Here we map the core clusters.
            
            labels_map = {
                0: "RETAIN",
                1: "RE-ENGAGE",
                2: "NURTURE",
                3: "REWARD"
            }
            
            result_label = labels_map.get(predicted_cluster_id, "Unknown")
            
            # Display Result
            st.markdown("---")
            st.metric(label="Predicted Customer Segment", value=result_label)
            
            if result_label == "REWARD":
                st.balloons()
                st.success("This is a high-value loyal customer!")
            elif result_label == "RE-ENGAGE":
                st.warning("Action needed: Send a reactivation campaign.")
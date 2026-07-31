import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Wine Quality PCA Explorer", 
    page_icon="🍷", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Data & Models ---
@st.cache_data(show_spinner=False)
def load_data():
    # Updated to match the exact file name on your server
    df = pd.read_csv("PCA.csv")
    X = df.drop('quality', axis=1)
    y = df['quality']
    return df, X, y

@st.cache_resource(show_spinner=False)
def load_models():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    return scaler, pca

try:
    df, X_train, y_train = load_data()
    scaler, pca = load_models()
except FileNotFoundError as e:
    st.error(f"🚨 **Missing File:** {e}. Please ensure 'PCA.csv', 'scaler.pkl', and 'pca_model.pkl' are in your GitHub repository.")
    st.stop()
except EOFError:
    st.error("🚨 **Corrupted Model:** One of your `.pkl` files is empty (0 bytes). Please re-upload healthy model files.")
    st.stop()

# Calculate the background PCA
X_train_scaled = scaler.transform(X_train)
X_train_pca = pca.transform(X_train_scaled)

# --- 3. Sidebar GUI (User Inputs) ---
with st.sidebar:
    st.header("🧪 Formulate Your Wine")
    st.write("Adjust the chemical properties below to map a new custom wine.")
    st.divider()
    
    # Create a dictionary to hold user inputs dynamically
    user_inputs = {}
    for col in X_train.columns:
        min_val = float(X_train[col].min())
        max_val = float(X_train[col].max())
        mean_val = float(X_train[col].mean())
        
        user_inputs[col] = st.slider(
            label=col, 
            min_value=min_val, 
            max_value=max_val, 
            value=mean_val
        )
        
    st.divider()
    st.caption("Principal Component Analysis reduces dimensionality by projecting data onto eigenvectors:")
    st.latex(r"Z = X W")

# Convert user inputs into a DataFrame
user_df = pd.DataFrame([user_inputs])

# --- 4. Process Custom Wine ---
user_scaled = scaler.transform(user_df)
user_pca = pca.transform(user_scaled)

# --- 5. Main Dashboard UI ---
st.title("🍷 Wine Quality PCA Explorer")
st.markdown("This dashboard uses **Principal Component Analysis (PCA)** to compress 11 complex chemical features down to 2 dimensions. See where your custom wine formulation lands compared to the historical dataset!")
st.divider()

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    st.subheader("Dimensionality Reduction Projection")
    
    # Create a sleek, modern plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0E1117') # Streamlit dark mode background
    ax.set_facecolor('#0E1117')

    # Plot the historical dataset
    scatter = ax.scatter(
        X_train_pca[:, 0], 
        X_train_pca[:, 1], 
        c=y_train, 
        cmap='viridis', 
        alpha=0.6,
        s=40,
        label='Historical Data'
    )

    # Plot the new custom wine on top
    ax.scatter(
        user_pca[:, 0], 
        user_pca[:, 1], 
        c='#FF4B4B', # Streamlit red 
        marker='*', 
        s=600, 
        edgecolor='white',
        linewidth=1.5,
        label='Your Custom Wine'
    )

    # Formatting
    cbar = plt.colorbar(scatter)
    cbar.set_label('Wine Quality', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xlabel("Principal Component 1 (PC1)", color='white')
    ax.set_ylabel("Principal Component 2 (PC2)", color='white')
    ax.tick_params(colors='white')
    
    # Style the legend and grid
    legend = ax.legend(facecolor='#262730', edgecolor='none')
    for text in legend.get_texts():
        text.set_color('white')
        
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    
    # Hide top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    # Display the plot
    st.pyplot(fig)

with col2:
    st.subheader("Live Results")
    st.info("The red star on the chart represents your custom wine.")
    
    st.metric(label="PC1 Coordinate", value=f"{user_pca[0][0]:.3f}")
    st.metric(label="PC2 Coordinate", value=f"{user_pca[0][1]:.3f}")
    
    st.divider()
    st.markdown("**What does this mean?**")
    st.markdown("Wines that cluster close together share similar chemical profiles. If your star lands in a dense cluster of high-quality wines (yellow/green), your formulation is likely highly rated!")

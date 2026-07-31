import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Wine Quality PCA Explorer", layout="wide")

st.title("🍷 Wine Quality PCA Explorer")
st.write("Adjust the chemical properties in the sidebar to see where your custom wine lands on the Principal Component Analysis (PCA) plot!")

# -------------------------------------------------------------------------
# 1. LOAD DATA & MODELS
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PCA.csv")
    X = df.drop('quality', axis=1)
    y = df['quality']
    return df, X, y

@st.cache_resource
def load_models():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca_model.pkl', 'rb') as f:
        pca = pickle.load(f)
    return scaler, pca

df, X_train, y_train = load_data()
scaler, pca = load_models()

# Calculate the PCA for the background scatter plot
X_train_scaled = scaler.transform(X_train)
X_train_pca = pca.transform(X_train_scaled)

# -------------------------------------------------------------------------
# 2. SIDEBAR GUI (User Inputs)
# -------------------------------------------------------------------------
st.sidebar.header("Custom Wine Features")

# Create a dictionary to hold user inputs, using the min/max from the dataset for the slider ranges
user_inputs = {}
for col in X_train.columns:
    min_val = float(X_train[col].min())
    max_val = float(X_train[col].max())
    mean_val = float(X_train[col].mean())
    
    user_inputs[col] = st.sidebar.slider(
        label=col, 
        min_value=min_val, 
        max_value=max_val, 
        value=mean_val
    )

# Convert user inputs into a DataFrame
user_df = pd.DataFrame([user_inputs])

# -------------------------------------------------------------------------
# 3. PROCESS CUSTOM WINE & PLOT
# -------------------------------------------------------------------------
# Scale and apply PCA to the custom wine
user_scaled = scaler.transform(user_df)
user_pca = pca.transform(user_scaled)

st.subheader("PCA Projection")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the original dataset
scatter = ax.scatter(
    X_train_pca[:, 0], 
    X_train_pca[:, 1], 
    c=y_train, 
    cmap='viridis', 
    alpha=0.5,
    label='Historical Data'
)

# Plot the new custom wine on top
ax.scatter(
    user_pca[:, 0], 
    user_pca[:, 1], 
    c='red', 
    marker='*', 
    s=500, 
    edgecolor='black',
    label='Your Custom Wine'
)

# Formatting
plt.colorbar(scatter, label='Wine Quality')
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("PCA: Red Wine Analysis")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Display the plot in Streamlit
st.pyplot(fig)

# Show the exact PCA coordinates for the custom wine
st.write(f"**Your Custom Wine PCA Coordinates:** PC1 = `{user_pca[0][0]:.2f}`, PC2 = `{user_pca[0][1]:.2f}`")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set the background color using Streamlit's markdown
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef); /* Light gradient background */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #343a40; /* Darker text color for better readability */
    }
    .stButton>button {
        background-color: #007bff; /* Primary button color */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker button color on hover */
    }
    .stTextInput>div>input {
        border-radius: 4px;
        border: 1px solid #ced4da;
        padding: 10px;
        font-size: 16px;
    }
    .stSelectbox>div>select {
        border-radius: 4px;
        border: 1px solid #ced4da;
        padding: 10px;
        font-size: 16px;
    }
    .stDataFrame {
        margin-top: 20px;
        border-radius: 4px;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Load the dataset
data = pd.read_csv('Student_Performance.csv')

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Extracurricular Activities' to numerical values
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('Performance Index', axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split the data
X = scaled_df
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

def plot_feature_vs_target(df, target):
    features = df.columns[:-1]  # Exclude the target column
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[feature], df[target], alpha=0.5)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f'{feature} vs {target}')
        st.pyplot(fig)  # Display the figure in Streamlit
        plt.close(fig)  # Close the figure to avoid memory issues

def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel('Actual Performance Index')
    ax.set_ylabel('Predicted Performance Index')
    ax.set_title('Actual vs Predicted Performance Index')
    st.pyplot(fig)  # Display the figure in Streamlit
    plt.close(fig)  # Close the figure to avoid memory issues

# Streamlit App
st.title('Performance Index Prediction')

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Visualizations", "Model"])

with tab1:
    # Input fields
    hours_studied = st.number_input('Hours Studied', min_value=0)
    previous_scores = st.number_input('Previous Scores', min_value=0)
    extracurricular_activities = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    sleep_hours = st.number_input('Sleep Hours', min_value=0)
    sample_question_papers_practiced = st.number_input('Sample Question Papers Practiced', min_value=0)

    # Check if sleep_hours + hours_studied is less than or equal to 24
    if (sleep_hours + hours_studied) > 24:
        st.error("The total of Sleep Hours and Hours Studied should not exceed 24 hours.")

    # Encode 'Extracurricular Activities' as it was done during training
    extracurricular_activities_encoded = 1 if extracurricular_activities == 'Yes' else 0

    # Create a DataFrame for the input with the correct column names
    input_data = pd.DataFrame({
        'Hours Studied': [hours_studied],
        'Previous Scores': [previous_scores],
        'Extracurricular Activities': [extracurricular_activities_encoded],
        'Sleep Hours': [sleep_hours],
        'Sample Question Papers Practiced': [sample_question_papers_practiced]
    })

    input_data = input_data[X.columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    if st.button('Predict'):
        prediction = model.predict(input_data_scaled)
        # Check if prediction exceeds 100
        if prediction[0] > 100:
            st.warning("The predicted Performance Index exceeds the maximum value of 100.")
        else:
            st.write(f'Predicted Performance Index: {prediction[0]:.2f}')

with tab2:
    st.header('Model Visualizations')
    st.write("Actual Vs Predicted Performance Index")
    plot_actual_vs_predicted(y_test, y_pred)  # Plot actual vs predicted values
    st.write("Parameters Vs Predicted Performance Index")
    plot_feature_vs_target(df, 'Performance Index')  # Plot each feature against the target

with tab3:
    st.header('Model Details')

    # Display metrics
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"R² Score: {r2}")

    # Display the dataset
    st.header('Dataset')
    st.dataframe(df)

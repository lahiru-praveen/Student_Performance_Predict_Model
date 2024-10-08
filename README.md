# Performance Index Prediction

This project aims to predict the performance index of students based on various factors such as hours studied, previous scores, extracurricular activities, sleep hours, and the number of sample question papers practiced.

## Project Overview

This project uses a multiple linear regression model to predict the performance index of students. The model is built using Python and several machine learning libraries. The project is hosted as a web application using Streamlit, allowing users to input various parameters and obtain predictions instantly.

### Features
- Predict student performance index based on multiple parameters.
- Visualizations to show relationships between the parameters and the predicted performance index.
- Model evaluation metrics such as Mean Squared Error, Root Mean Squared Error, and R² Score.

## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/). It includes various features that may influence a student's performance index, such as:

- Hours Studied
- Previous Scores
- Extracurricular Activities (Yes/No)
- Sleep Hours
- Sample Question Papers Practiced

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/lahiru-praveen/Student_Performance_Predict_Model.git
   cd Student_Performance_Predict_Model

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit app:**

   ```bash
   streamlit run app.py

## Usage

Once the app is running, you can:

- Input your data into the form provided in the "Prediction" tab.
- View visualizations in the "Visualizations" tab.
- Review model details and metrics in the "Model" tab.

## Requirements

This project requires the following Python packages, which are listed in the `requirements.txt` file:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

## Acknowledgements

- **Dataset**: The dataset used in this project is obtained from Kaggle. Full credit goes to the original creator of the dataset.
- **Libraries**: This project utilizes several open-source libraries, including Scikit-learn, Matplotlib, Seaborn, and Streamlit.

      


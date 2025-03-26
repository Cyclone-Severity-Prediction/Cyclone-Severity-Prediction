Cyclone Severity Prediction
Project Overview
Introduction
Cyclones are powerful atmospheric phenomena that can cause widespread destruction, loss of life, and significant economic damage. Accurate predictions of cyclone intensity are crucial for:

Warning populations in advance, reducing casualties and property damage.

Optimizing emergency response measures, such as evacuations and infrastructure protection.

Supporting insurance companies in risk assessment and disaster planning.

Assisting governments and organizations in making strategic decisions.

This project uses machine learning to predict the severity of tropical cyclones based on historical data. By leveraging the International Best Track Archive for Climate Stewardship (IBTrACS) dataset, we aim to forecast the cyclone intensity and help improve preparedness efforts.

Key Objectives
Identify key factors that influence cyclone strength:

Analyze historical cyclone data and identify factors like geographical location (latitude and longitude), wind speed, and pressure.

Develop a model to predict cyclone intensity levels:

Create a machine learning model that predicts the severity of a cyclone based on available features such as wind speed and pressure.

Output: The model will predict the severity of a cyclone using the TD9636_STAGE column, which includes categories for cyclone intensity.

The IBTrACS dataset is a global collection of historical tropical cyclone data, compiled by various meteorological agencies. The dataset includes key information such as:

Cyclone location (latitude and longitude)

Cyclone intensity (wind speed and central pressure)

Storm size and structure (radius of maximum winds)

Technologies Used
This project uses the following technologies:

Python: Main programming language for data analysis and machine learning.

Jupyter Notebook: For running and visualizing data analysis, model training, and evaluation.

Flask or Streamlit: To build the web application and serve the trained model.

Scikit-learn: For implementing machine learning algorithms.

XGBoost: A powerful library for gradient boosting used in model training.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations and handling arrays.

Matplotlib/Seaborn: For data visualization (e.g., plots, graphs).

Installation Guide
Clone the Repository
Start by cloning this repository to your local machine:
git clone https://github.com/yourusername/Cyclone-Severity-Prediction.git
cd Cyclone-Severity-Prediction

Set Up the Virtual Environment
It is recommended to use a virtual environment to manage dependencies. Create a virtual environment using the following command:
python -m venv venv

Activate the virtual environment:

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

Install the Dependencies
Once the virtual environment is active, install the required dependencies by running:
pip install -r requirements.txt

Dataset Information
The dataset used in this project is sourced from the International Best Track Archive for Climate Stewardship (IBTrACS). This dataset includes historical tropical cyclone data from various meteorological agencies around the world.

Key features of the dataset:

SID: Storm identifier

SEASON: Year the cyclone occurred

NUMBER: The cyclone number for that year

LAT, LON: The geographical location (latitude, longitude) of the cyclone

WMO_WIND: Maximum sustained wind speed (knots)

WMO_PRES: Minimum central pressure (millibars)

For more information on the dataset, you can refer to the IBTrACS Technical Documentation.

Steps to Run the Project
1. Data Preprocessing
Before training the machine learning model, we need to clean and preprocess the data:

Missing values: Handle missing values by either filling or removing rows.

Feature engineering: Transform the data by extracting meaningful features (e.g., converting time to morning/afternoon/evening).

Normalization: Normalize numerical features like wind speed and pressure.

2. Exploratory Data Analysis (EDA)
Perform initial exploratory data analysis to understand the data:

Visualize the distribution of cyclones over time.

Explore correlations between cyclone features and severity.

Identify outliers and trends.

3. Model Selection
We evaluate several machine learning models to predict cyclone severity:

Logistic Regression: For binary classification tasks.

Decision Tree: A simple model for classification.

Random Forest: An ensemble method that improves performance by averaging the results of multiple decision trees.

XGBoost: A more advanced gradient boosting method known for its high performance.

4. Model Evaluation
After training the models, evaluate them based on:

Accuracy

Precision

Recall

F1-score

5. Web Application
We have built a simple web application to showcase the model predictions:

Flask or Streamlit serves as the backend for running the prediction model.

Users can input cyclone features (e.g., latitude, longitude, wind speed) through the web interface and receive a severity prediction.

6. Deployment
You can run the web application locally with the following commands:

Flask:
python app.py

Streamlit:
streamlit run app.py

The application will be available at:

Flask: http://localhost:5000
Streamlit: http://localhost:8501

How to Contribute
Contributions are welcome! Feel free to fork the repository, create a new branch, and submit a pull request.

Steps to Contribute:
Fork the repository to your GitHub account.

Create a new branch for your feature:
git checkout -b feature-name

Commit your changes:
git commit -am "Description of the feature"

Push to your forked repository:
git push origin feature-name

Open a pull request in the main repository.

Acknowledgments
IBTrACS for providing the tropical cyclone dataset.

Flask and Streamlit for web app development.

Scikit-learn and XGBoost for machine learning.

Matplotlib and Seaborn for data visualization.



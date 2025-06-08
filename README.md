
# Solar Energy Potential Predictor

## Overview
The *Solar Energy Potential Predictor* is a machine learning-powered web application built with Streamlit to forecast solar energy potential using real-time and historical weather data. It supports *UN Sustainable Development Goal 7 (SDG 7)*: Affordable and Clean Energy by optimizing solar energy utilization, reducing reliance on fossil fuels, and promoting equitable energy access across regions. The project uses historical weather data from weather.csv and real-time data from the OpenWeatherMap API to train and compare three machine learning models (Linear Regression, Random Forest, and XGBoost) for predicting solar potential.

The application provides:
- Real-time weather data retrieval for user-specified locations.
- Model performance metrics (MAE, MSE, R², F1-score, ROC-AUC, confusion matrix).
- Real-time solar potential predictions with estimated CO2 reduction.
- Visualizations of predicted vs. actual solar potential.
- Insights into sustainability impacts, aligning with SDG 7.

A companion Jupyter notebook (climate_change.ipynb) provides exploratory data analysis (EDA), model training, and ethical reflections on biases and fairness.

## Features
- *Real-Time Weather Data*: Fetches current weather conditions (cloud coverage, air temperature, wind speed) via the OpenWeatherMap API for locations like New York, London, Tokyo, or custom coordinates.
- *Machine Learning Models*: Compares Linear Regression, Random Forest, and XGBoost for predicting solar potential, calculated as 1 - (cloudCoverage / 100).
- *Performance Metrics*: Evaluates models using regression (MAE, MSE, R²) and classification metrics (F1-score, ROC-AUC, confusion matrix) for binary solar potential classification.
- *Visualization*: Displays scatter plots of predicted vs. actual solar potential using Matplotlib.
- *Sustainability Impact*: Estimates annual CO2 reduction based on predicted solar potential, supporting SDG 7 goals.
- *Ethical Considerations*: Addresses potential biases (geographical, temporal, data collection, proxy target) to ensure fair and equitable predictions.

## Installation

### Prerequisites
- *Python 3.8+*
- *Dependencies*: Listed in requirements.txt (see below).
- *OpenWeatherMap API Key*: Obtain a free API key from [OpenWeatherMap](https://openweathermap.org/api).

### Setup
1. *Clone the Repository*:
   bash
   git clone https://github.com/your-repo/solar-energy-predictor.git
   cd solar-energy-predictor
   

2. *Create a Virtual Environment*:
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

3. *Install Dependencies*:
   Create a requirements.txt file with the following:
   
   pandas==2.3.0
   numpy==2.3.0
   scikit-learn==1.5.1
   xgboost==2.0.2
   streamlit==1.39.0
   requests==2.32.3
   matplotlib==3.9.2
   seaborn==0.13.2
   
   Then install:
   bash
   pip install -r requirements.txt
   

4. *Configure API Key*:
   - Replace "YOUR_VALID_OPENWEATHERMAP_API_KEY" in solar_energy_app_updated.py with your OpenWeatherMap API key.

5. *Prepare Dataset*:
   - Ensure weather.csv is in the project root directory. The dataset should contain columns: timestamp, site_id, airTemperature, cloudCoverage, dewTemperature, precipDepth1HR, precipDepth6HR, seaLvlPressure, windDirection, windSpeed.

## Usage

### Running the Streamlit App
1. Start the Streamlit server:
   bash
   streamlit run solar_energy_app.py
   
2. Open the provided URL (e.g., http://localhost:8501) in your browser.
3. *Interact with the App*:
   - Select a predefined location (New York, London, Tokyo) or enter custom latitude/longitude.
   - Click *"Fetch Real-Time Data"* to retrieve current weather data.
   - View model performance metrics, real-time predictions, CO2 reduction estimates, and a scatter plot of predicted vs. actual solar potential.
   - Explore sustainability impacts under the "Sustainability Impact (SDG 7)" section.

### Running the Jupyter Notebook
1. Launch Jupyter Notebook:
   bash
   jupyter notebook climate_change.ipynb
   
2. Run the cells to perform EDA, train models, and visualize predictions.
3. Review the ethical reflection section for insights on bias and fairness.

## Dataset
- *Historical Data* (weather.csv):
  - *Source*: User-provided dataset (not publicly available).
  - *Size*: 331,166 rows, 10 columns.
  - *Columns*:
    - timestamp: Date and time (e.g., 2016-01-01 00:00:00).
    - site_id: Location identifier (e.g., Panther).
    - airTemperature: Temperature in °C.
    - cloudCoverage: Cloud cover percentage (0-100, significant missing values).
    - dewTemperature: Dew point temperature in °C.
    - precipDepth1HR: Hourly precipitation depth in mm.
    - precipDepth6HR: 6-hour precipitation depth in mm.
    - seaLvlPressure: Sea-level pressure in hPa.
    - windDirection: Wind direction in degrees.
    - windSpeed: Wind speed in m/s.
  - *Target*: solar_potential (derived as 1 - (cloudCoverage / 100)).
  - *Note*: cloudCoverage has ~51% missing values, handled via mean imputation in the app.

- *Real-Time Data*:
  - Fetched via OpenWeatherMap API.
  - Features: cloudCoverage, airTemperature, windSpeed, timestamp.

## Model Details
- *Features Used*: hour (derived from timestamp), cloudCoverage, airTemperature, windSpeed.
- *Models*:
  - *Linear Regression*: Simple baseline model.
  - *Random Forest*: Ensemble model with 100 estimators.
  - *XGBoost*: Gradient boosting model with 100 estimators.
- *Training*:
  - 80/20 train-test split.
  - Features scaled using StandardScaler.
- *Metrics*:
  - Regression: MAE, MSE, R².
  - Classification: F1-score, ROC-AUC, confusion matrix (based on median threshold for solar potential).
- *CO2 Reduction Estimate*: Assumes 0.5 kg CO2/day per unit of solar potential (hypothetical).

## Visualization
- *Plot*: Scatter plot of predicted vs. actual solar potential for each model, with a diagonal reference line.
- *Library*: Matplotlib (backend set to Agg for Streamlit compatibility).
- *Issue Resolution*: Ensure weather.csv is present, API key is valid, and features align (airTemperature vs. temperature).

## Ethical Considerations
Based on the climate_change.ipynb reflection:
- *Potential Biases*:
  - *Geographical Bias*: Overrepresentation of certain site_id values may skew predictions, affecting regions with less data.
  - *Temporal Bias*: Limited temporal coverage may miss seasonal variations, leading to unreliable forecasts.
  - *Data Collection Bias*: Missing values (e.g., 51% of cloudCoverage) or sensor errors could distort predictions.
  - *Proxy Target Bias*: Simplified solar_potential formula may not capture complex weather interactions, favoring clear-sky scenarios.
- *Fairness*:
  - Includes diverse locations via user input.
  - Mitigates bias through mean imputation and validation.
  - Transparent reporting of metrics and limitations.
- *Sustainability*:
  - Optimizes solar energy use, reducing fossil fuel dependency.
  - Enhances resource efficiency through accurate predictions.
  - Supports scalable, equitable energy access.

## Limitations
- *Data Dependency*: Requires weather.csv and a valid OpenWeatherMap API key.
- *Missing Values*: High proportion of missing cloudCoverage data may affect accuracy.
- *Simplified Target*: solar_potential formula is a proxy and may not reflect true solar radiation.
- *API Reliability*: Real-time predictions depend on API availability and response quality.

## Future Improvements
- Use advanced imputation methods (e.g., interpolation) for missing data.
- Incorporate additional features (e.g., dewTemperature, seaLvlPressure) for better predictions.
- Validate solar_potential formula against actual solar radiation data.
- Add interactive visualizations using Chart.js for enhanced user experience.
- Expand dataset to include more regions and time periods to reduce bias.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (git checkout -b feature/your-feature).
3. Commit changes (git commit -m 'Add your feature').
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.


## Acknowledgments
- *OpenWeatherMap*: For providing real-time weather data.
- *UN SDG 7*: Inspiration for promoting affordable and clean energy.
- *Streamlit*: For enabling rapid web app development.
- *Scikit-learn & XGBoost*: For robust machine learning tools.

## Contact
For issues or inquiries, please open an issue on the GitHub repository or contact lathithavena3@gmail.com.

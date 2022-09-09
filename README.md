# Project 4: Machine Learning Models in Cardiovascular Disease Prediction
<p align = "center">
    <img src="Resources/img1_heartbody.jpeg", width="1000"/>
</p>

** Group 4 Members:** Alex Riiska, Laurane Gerber, Nikki Dao, Victoria Sandoval, Yuntian Xue

### Project Goal: 

By implementing multiple maching learning models, we aim to determine whether patients' heart disease can be predicted by clinical cardiovascular diseases data. Using ETL method to clean data and EDA approach to analyze the data, as well as summarize the data characteristic to discover what hidden story lies within the data, prior to applying various machine learning models. 

## Dataset: 

**Data Description:**  
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.  

Data can be found in the following links: [Kaggle link](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), csv file: [heart.csv](Resources/heart.csv)  

**Columns:**  
| Column Name | Description |
| --- | --- |
| `Age` | Patient Age |
| `Sex` | Patient Sex; F - Female, M - Male |
| `ChestPainType` | ASY – Asymptomatic, ATA – Atypical Angina, NAP – Non-Anginal Pain, TA – Typical Angina |
| `RestingBP` | Patient blood pressure at rest |
| `Cholesterol` | Patient cholesterol level |
| `FastingBS` | Patient blood sugar when fasting |
| `RestingECG` | Patient’s electrocardiogram reading |
| `MAXHR` | Patient’s highest heart rate reading |
| `ExcersiseAngina` | Determines whether Patient experiences angina from exercise; Y - Yes, N - No |
| `Oldpeak` | Patients with an oldpeak = ST; ST reading determines abnormality on an ECG. |
| `ST_Slope` | Determines incremental heart-rate shifts, usually excercise-induced; Down – downsloping, Flat – no slope, Up – upsloping |
| `Heartdisease` | Determines whether patient has heart disease; 1 – patient has heart disease, 0 – patient does not have heart disease |


OUTLINE DATE IMAGE

## Data Preparation and Modeling
Exploring data allows us to better understand the relationship of the predictor and its key factors. Here we generated the descriptive statistic summary for Patient Age, Blood Pressure, Cholesterol, Blood Sugar, Heart Rate, Old Peak (Numeric value measured in depression), and Heart Disease.

Pre-procression Step: 
-   ETL – Removing null values, renaming columns name, transform objects from string to integer, identify outliers, load from Postgres database
-   Scaling –  

Tools Used:
-	SQLAlchemy : connecting to Postgres database
-	Python SKLearn libraries
-	Plotly.express/ matplotlib/ seaborn : graphs and visualizations

Model 1: Decision Tree (notebook)
-	About: why we chose, the %
-	Image Screen Shot
Model 2: Hyperlink
-	Image Screen shot
Model 3: Hyperlink
-Image
Model 4: 
-	Image
Model 5:
-	Image

Data
Images Folder
Scripts (All Model notebook)

**Results:**
| Model Name | Accuracy |
| --- | --- |
| Logistic Regression | 0. |
| Random Forest | 0.875 |
| Gaussian Naive Bayes | 0.883   |
| KNN | 0. |
| Decision Tree | 0.8384 |
| Neural Network | 0. |


# Project 4: Machine Learning Models in Cardiovascular Disease Prediction
<p align = "center">
    <img src="Resources/img1_heartbody.jpeg", width="1000"/>
</p>

**Group 4 Members:** Alex Riiska, Laurane Gerber, Nikki Dao, Victoria Sandoval, Yuntian Xue

## Project Goal: 

By implementing multiple maching learning models, we aim to determine whether patients' heart disease can be predicted by clinical cardiovascular diseases data. Using ETL method to clean data and EDA approach to analyze the data, as well as summarize the data characteristic to discover what hidden story lies within the data, prior to applying various machine learning models. 

## Data Descriptions: 

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease. This dataset was created from 5 other dataset from Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog, with a total of 918 observations of Heart Disease related cases. 

Data can be found in [Kaggle link](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), csv file: [heart.csv](Resources/heart.csv)  

**Columns Details:**  
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

## Data Preparation:
Exploring data allows us to better understand the relationship of the predictor and its key factors. Here we generated the descriptive statistic summary for Patient Age, Blood Pressure, Cholesterol, Blood Sugar, Heart Rate, Old Peak (Numeric value measured in depression), and Heart Disease. This step helped us determine which Machine Learning Models are best to implement.

**Pre-procression Step:**
-   ETL – Removing null values, renaming columns name, transform objects from string to integer, identify outliers, load from Postgres database
-   EDA – Scatter Matrix Plot, Bar Plot, Line Plot, and Heatmap to demonstrate data distribution and correlation between importance features

**Tools Used:**
-	SQLAlchemy : connecting to Postgres database
-	Python SKLearn libraries
-	Plotly.express/ matplotlib/ seaborn : graphs and visualizations

## Machine Learning Models:
Since our response variable is binary categorical variable, we will be using classification algorithms such as: Logistic Regression, Random Forest Classifier, Decision Tree Classifier, K-Nearest Neighbor, and Gaussian Naive Bayes. We believe these models would be well fitted as our categorical variable classified as 0: Non-Carriers and 1: Heart Disease Carriers. We did considered the typical 80:20 ratio of 80% train and 20% test data split and adequate sample size in our testing. 

#### Model 1: Decision Tree Classifier (notebook hyperlink)
-	About: why we chose, the %
-	Image Screen Shot
##### Model 2: Logistic Regression (notebook hyperlink)
-	Image Screen shot
##### Model 3: Random Forest Classifier (notebook hyperlink)
-Image
##### Model 4: K-Nearest Neighbor(notebook hyperlink)
-	Image
##### Model 5: Gaussian Naive Bayes (notebook hyperlink)
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

## Conclusion:
Supervised learning a type of machine learning model were a variable or variables is to represent a response. The goal of supervised learning is to make inference or make predictions . Algorithms are used to fit and select models for classification or prediction. In our case, we are using classification.

As previously found, the model that scores the highest in accuracy is ____. 

For future We would also consider hypertune and set hyperparameters in future approaches. 

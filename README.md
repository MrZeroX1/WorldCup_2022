# FIFA World Cup Prediction Model

This repository contains Python code for building a predictive model for FIFA World Cup outcomes using machine learning. The model employs three different classifiers: Random Forest, Decision Tree, and K-Nearest Neighbors (KNN). The code includes data loading, exploration, preprocessing, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [File Descriptions](#file-descriptions)
- [Getting Started](#getting-started)
- [Exploring the Data](#exploring-the-data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to predict the outcomes of FIFA World Cup matches using historical data. Three different classifiers are employed: Random Forest, Decision Tree, and K-Nearest Neighbors. The performance of each model is evaluated using various metrics such as accuracy, precision, recall, and a confusion matrix.

## Dataset

The dataset used in this project is named `worldCup_dataset.csv`. It contains information about FIFA World Cup matches, including features such as date, teams, city, country, and match outcome.

## File Descriptions

- `worldCup_dataset.csv`: The dataset containing FIFA World Cup match information.
- `model.ipynb`: The notbook that containing the code for data loading, exploration, preprocessing, model training, and evaluation.
- `fifa_world_cup_clean.csv`: The cleaned version of the dataset after preprocessing.

## Getting Started

To run the code, you need to have Python installed on your machine along with the required libraries, such as NumPy, Pandas, Seaborn, Matplotlib, and Scikit-Learn. You can install these dependencies using the following command:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

After installing the dependencies, you can run the `model.ipynb` script to execute the entire pipeline.

## Exploring the Data

The data exploration section includes loading the dataset, displaying the first and last 10 rows, checking for missing values, and creating visualizations such as heatmaps to visualize missing data and correlation matrices.

## Data Preprocessing

Data preprocessing involves handling missing values, converting categorical variables into numerical format using label encoding, and scaling the data using StandardScaler. The preprocessed data is saved as `fifa_world_cup_clean.csv`.

## Model Training and Evaluation

Three classifiers are trained and evaluated: Random Forest, Decision Tree, and K-Nearest Neighbors. The evaluation includes metrics such as accuracy, precision, recall, and a confusion matrix.

## Results

The results section presents the performance metrics of each model, including accuracy scores, classification reports, and confusion matrices.

## Conclusion

In conclusion, this project demonstrates the process of building and evaluating machine learning models for predicting FIFA World Cup match outcomes. Each model's strengths and weaknesses are discussed based on the evaluation metrics.

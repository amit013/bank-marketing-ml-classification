# Bank Marketing Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit (`yes` or `no`).

The project implements multiple classification algorithms on the same Bank Marketing dataset and compares their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to allow users to upload test data, select a machine learning model, and view prediction results and evaluation metrics.

---

## 2. Dataset Description

### Dataset

**Bank Marketing Dataset**

The dataset contains customer demographic information, account-related information, and details about previous marketing campaigns.

### Target Variable

The target variable is:

* `yes` – Customer subscribed to a term deposit
* `no` – Customer did not subscribe to a term deposit

### Dataset Characteristics

* Total instances: **45,211**
* Original features: **16**
* Target classes: **2**
* Training samples: **36,168**
* Testing samples: **9,043**
* Transformed features after preprocessing: **51**

### Numerical Features

* `age`
* `balance`
* `day_of_week`
* `duration`
* `campaign`
* `pdays`
* `previous`

### Categorical Features

* `job`
* `marital`
* `education`
* `default`
* `housing`
* `loan`
* `contact`
* `month`
* `poutcome`

### Class Distribution

The target variable is imbalanced:

| Class |  Count | Percentage |
| ----- | -----: | ---------: |
| No    | 39,922 |      88.3% |
| Yes   |  5,289 |      11.7% |

The training and testing datasets maintain approximately the same class distribution.

### Preprocessing

Numerical features were processed using:

* Median imputation
* StandardScaler

Categorical features were processed using:

* Missing-value replacement with `Unknown`
* One-hot encoding
* Unknown categories ignored during transformation

The preprocessing pipeline transformed the original 16 features into **51 features**.

---

## 3. GitHub Repository

GitHub Repository:

https://github.com/amit013/bank-marketing-ml-classification

The repository contains the source code, saved machine learning models, requirements file, test data, README, and Streamlit application.

---

## 4. Machine Learning Models

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest Ensemble

The models were evaluated using:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 5. Model Performance Comparison

| ML Model Name            |   Accuracy |        AUC |  Precision |     Recall |         F1 |        MCC |
| ------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression      |     0.8457 |     0.9079 |     0.4182 |     0.8147 |     0.5527 |     0.5092 |
| Decision Tree            |     0.8287 |     0.8766 |     0.3929 |     0.8516 |     0.5377 |     0.5004 |
| kNN                      | **0.9015** |     0.8842 | **0.6701** |     0.3110 |     0.4248 |     0.4123 |
| Naive Bayes              |     0.8548 |     0.8101 |     0.4059 |     0.5198 |     0.4559 |     0.3774 |
| Random Forest (Ensemble) |     0.8696 | **0.9275** |     0.4676 | **0.8251** | **0.5969** | **0.5564** |

---

## 6. Observations on Model Performance

| ML Model Name                       | Observation about Model Performance                                                                                                                                                                                                                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**             | Achieved 84.57% accuracy and an AUC of 0.9079. It achieved good recall (81.47%), indicating that it was effective at identifying customers who subscribed to the term deposit.                                                                                                                             |
| **Decision Tree**                   | Achieved 82.87% accuracy. It obtained the highest recall (85.16%) among the models, but its precision was relatively low (39.29%), resulting in more false-positive predictions.                                                                                                                           |
| **kNN**                             | Achieved the highest accuracy (90.15%) and highest precision (67.01%). However, its recall was only 31.10%, meaning it missed a considerable number of positive customers.                                                                                                                                 |
| **Naive Bayes**                     | Achieved 85.48% accuracy. Its AUC (0.8101), F1 score (0.4559), and MCC (0.3774) were comparatively lower than the other models, indicating weaker overall classification performance.                                                                                                                      |
| **Random Forest (Ensemble)**        | Achieved the highest AUC (0.9275), F1 score (0.5969), and MCC (0.5564). It also achieved high recall (82.51%), providing a good balance between identifying positive customers and limiting false predictions.                                                                                             |
| **Overall Winner for your dataset** | **Random Forest (Ensemble)** was selected as the overall winner because it achieved the best AUC, F1 score, and MCC, while maintaining high recall. Although kNN had the highest accuracy, its low recall (31.10%) makes Random Forest a better overall choice for this imbalanced classification problem. |

---

## 8. Feature Importance

Feature importance was analyzed using the trained Random Forest model.

The most important original features were:

| Rank | Feature     | Importance |
| ---: | ----------- | ---------: |
|    1 | duration    |   0.429563 |
|    2 | month       |   0.104431 |
|    3 | poutcome    |   0.077895 |
|    4 | contact     |   0.061865 |
|    5 | housing     |   0.049000 |
|    6 | age         |   0.046733 |
|    7 | balance     |   0.043596 |
|    8 | day_of_week |   0.038578 |
|    9 | pdays       |   0.034053 |
|   10 | job         |   0.029354 |

The feature importance values were aggregated from the one-hot encoded features back to their original feature names.

The total importance sums to 1.0.

---

## 9. Streamlit Application

An interactive Streamlit application was developed for demonstrating the trained classification models.

The application provides:

* Test-data CSV upload
* Model selection
* Customer prediction
* Prediction probability
* Evaluation metrics
* Confusion matrix / classification results
* Comparison of model predictions

The application uses the saved trained model pipelines from the `model/` directory.

---

## 10. Project Structure

```text
bank-marketing-ml-classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

## 11. Requirements

The project uses the following Python packages:

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Plotly

Dependencies are listed in `requirements.txt`.

---

## 12. Running the Application Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 13. Deployment

The Streamlit application is intended to be deployed using Streamlit Community Cloud.

The deployed application provides an interactive frontend for testing the trained classification models.

---

## 14. Conclusion

Five classification models were implemented and evaluated on the Bank Marketing dataset.

KNN achieved the highest accuracy and precision, while Decision Tree achieved the highest recall.

Random Forest achieved the highest AUC, F1 score, and MCC score and provided the best overall balance for the imbalanced classification problem.

Therefore, **Random Forest was selected as the final model** for this project.

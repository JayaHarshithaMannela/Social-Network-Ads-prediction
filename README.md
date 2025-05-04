# Social Network Ads - KNN Classification

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Features](#project-features)
- [How to Set Up the Project](#how-to-set-up-the-project)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Building and Training the Model](#building-and-training-the-model)
- [Model Evaluation](#model-evaluation)
- [Tuning K Parameter](#tuning-k-parameter)
- [Usage](#usage)
- [Contributing](#contributing)

---

## Project Overview
This project uses the K-Nearest Neighbors (KNN) classification algorithm to predict whether a user will purchase a product based on their age, gender, and estimated salary.

---

## Dataset
The dataset used is **Social_Network_Ads.csv**, which contains features like `User ID`, `Gender`, `Age`, `EstimatedSalary`, and `Purchased`.

Place the `Social_Network_Ads.csv` file in the project directory.

---

## Project Features
- **Data Preprocessing**: Label encoding and standardization.
- **Visualization**: Confusion matrix and accuracy vs. K-value plot.
- **Model Training**: K-Nearest Neighbors classifier.
- **Model Evaluation**: Accuracy, confusion matrix, classification report.

---

## How to Set Up the Project

### Prerequisites
Make sure you have **Python 3.x** installed (recommended: 3.7+).

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Exploratory Data Analysis (EDA)

You can visualize data distributions or correlation patterns if needed using `seaborn` or `matplotlib`.

---

## Building and Training the Model

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

---

## Model Evaluation

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## Tuning K Parameter

```python
k_values = range(1, 21)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, preds))
```

Visualization:

```python
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.grid(True)
plt.show()
```

---

## Usage
- Run the script to train and evaluate the KNN model.
- Modify the number of neighbors for optimization.
- Use the confusion matrix and classification report to assess performance.

---

## Contributing
Feel free to fork this project, make enhancements, and submit a pull request. All contributions are appreciated!

## Dataset

The dataset used in this project is `Social_Network_Ads.csv`, which contains demographic data and user behavior on a social network.

> **Source:** This dataset is commonly used in open-source machine learning tutorials and educational material.  
> **License:** Used strictly for educational and non-commercial purposes. No proprietary or sensitive information is included.
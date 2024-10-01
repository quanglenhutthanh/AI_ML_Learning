# Introduction

This project demonstrate demonstrates the process of using various machine learning algorithms on a given dataset.

The project use scikit-learn, one of the most popular machine learning libraries in Python, to implement and evaluate the model.

# Prerequisites
## scikit-learn
scikit-learn is a popular open source library for machine learning in Python. It provides simple and efficient tools for data mining and data analysis, it alse offer variety of machine learning algorithms for both supervised and unsupervised learning. In this project, scikit-learn is used for:
- Date preprocessing task like scaling and splitting the data.
- Implemeting machine learning algorithms such as Support Vector Machine (SVM), Random Forest.
- Evaluation model performance based on metrics such as accuracy, precision, recall, F1 score.

## pandas
pandas is an open-source library for data manipulation and data analysis in Python. It provides powerful, flexible and easy-to-use data structures, such as DataFrame. With pandas, use can easily load data from various format (such as CSV, excel, json,...), clean and transform datasets, perform data analysis and integrate with others libraries for machine learning and visualization.

## ydata-profiling
ydata-profiling is a Python library that generates a data analysis report for a given dataset. It helps in quickly understanding the data, identifying missing values, detecting correlations, and generating visualizations, all with just a few lines of code.

# Dataset

## Diabetes Dataset Overview
This dataset contains medical records of patients, aimed at predicting whether or not they have diabetes based on certain health measurements. Below is a brief explanation of each feature:

- Pregnancies: Number of times the patient has been pregnant.

- Glucose: Blood sugar levels after a glucose tolerance test.

- BloodPressure: Diastolic blood pressure (the lower number in a blood pressure reading).

- SkinThickness: Thickness of skin on the triceps, used to estimate body fat.

- Insulin: Insulin levels in the blood two hours after a test.

- BMI: Body Mass Index, calculated from height and weight, indicating body fat levels.

- DiabetesPedigreeFunction: A score that indicates the patient’s likelihood of diabetes based on their family history.

- Age: The patient’s age.

- Outcome: The result of whether the patient has diabetes (1) or not (0).

## Stroke dataset overview

# Implamentation
## Data Preparation
The dataset is loaded using pandas.

read_csv
```python
data = pd.read_csv("diabetes.csv")
print(data.head())
```

read_excel
```python
data = pd.read_excel("Classification/data/stroke_classification.xlsx",dtype=str)
```

Use ydata-profiling to gain insights in to the dataset.

```python
# profiling
profile = ProfileReport(data,title="Dataset Report", explorative=True)
profile.to_file("dataset_report.html")

```
## Data Preporcessing

In this project, we will use some of essential functions provided by scikit-learn:

**SimpleImputer** is used to handle missing data by filling in with the specified strategy, such as mean, median, or most frequent value.

**LabelEncoder** converts caterical variebles into numerical values.



using **train_test_split** function to device dataset into 2 subsets, one for training model and one for testing it.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
```
**StandardScaler** 
Data scaler
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```
## Model Selection
```python
model = SVC()
```
training model and make prediction

```python
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(y_predict)
```
## Model Evaluation

```python
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy}")
```

## Model Optimization

```python
params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}
clf = GridSearchCV(
    estimator=SVC(random_state=100),
    param_grid=params,
    scoring="f1",
    cv=6,
    verbose=1,
    n_jobs=6
)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(clf.best_score_)
print(clf.best_params_)
```


/Library/Developer/CommandLineTools/usr/bin/python3.9
/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9
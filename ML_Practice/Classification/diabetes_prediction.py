import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv("diabetes.csv")
print(data.head())

# profiling
# profile = ProfileReport(data,title="Dataset Report", explorative=True)
# profile.to_file("dataset_report.html")


target = "Outcome"
x = data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model = SVC()
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
# print(y_predict)

# accuracy = accuracy_score(y_test, y_predict)
# print(f"Accuracy: {accuracy}")


# params = {
#     "n_estimator": [50, 100, 200],
#     "criterion" : ["gini", "entropy", "log_loss"]
# }
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












import pandas as pd
from run_models import run_classification_models

data = pd.read_csv("Classification/data/diabetes.csv")

run_classification_models(data,"Outcome")



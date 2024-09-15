import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_excel("NLP/final_project.ods", engine="odf", dtype=str)
#print(data.head())
#print(data["career_level"].value_counts())

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
# result = vectorizer.fit_transform(x_train["title"])
# print(vectorizer.vocabulary_)
# print(result.shape)

encoder = OneHotEncoder()
result = encoder.fit_transform(x_train[["location"]])
print(result.shape)


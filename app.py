import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
df = pd.read_csv("Names_dataset.csv")
x_df = df.name
y_df = df.gender
corpus = x_df.values.astype('U')
cv = CountVectorizer()
X = cv.fit_transform(corpus)

# Load the ML Model
clf_1 = joblib.load("naivebayes.pkl")

# Streamlit UI
st.title("Gender Predictor")

# Text input for name
name_query = st.text_input("Enter a name:")

# Prediction logic
if st.button("Predict"):
    if name_query:
        data = [name_query]
        vct = cv.transform(data).toarray()
        my_prediction = clf_1.predict(vct)
        st.write(f"'{name_query}' is: {'Female' if my_prediction == 0 else 'Male'}")
    else:
        st.error("Please enter a name.")

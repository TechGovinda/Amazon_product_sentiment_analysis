import streamlit as st
import pandas as pd
import pandas as pd
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords

# Select the predictor to be loaded from Models folder
predictor = pickle.load(open(r"model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"scaler.pkl", "rb"))
cv = pickle.load(open(r"countVectorizer (1).pkl", "rb"))
# def single_prediction(predictor, scaler, cv, text_input):
#     corpus = []
#     stemmer = PorterStemmer()
#     review = re.sub("[^a-zA-Z]", " ", text_input)
#     review = review.lower().split()
#     review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
#     review = " ".join(review)
#     corpus.append(review)
#     X_prediction = cv.transform(corpus).toarray()
#     X_prediction_scl = scaler.transform(X_prediction)
#     y_predictions = predictor.predict_proba(X_prediction_scl)
#     y_predictions = y_predictions.argmax(axis=1)[0]
#
#     return "Positive" if y_predictions == 1 else "Negative"

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_prediction = predictor.predict(X_prediction_scl)[0]
    return y_prediction
def main():
    st.title('Text Sentiment Predictor')
    st.header('Hello Guys,')
    st.subheader('Welcome to Sentiment analysis websites')
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
        with st.form(key='nlpForm'):
            raw_text = st.text_area('Enter Text Here')
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info('Sentiment')
                prediction = single_prediction(predictor, scaler, cv, raw_text)
                if prediction == 1:
                    st.write("Positive")
                else:
                    st.write('The prediction sentiment is'+' '+'negative')

            with col2:
                st.info('Sentiment Expression')
                if prediction == 1:
                    st.write('😍')
                else:
                    st.write('😭')
    else:
        st.subheader('About')


if __name__ == '__main__':
    main()

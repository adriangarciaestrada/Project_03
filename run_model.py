from joblib import dump, load
import pandas as pd
import string

vectorizer = load("./ML_code/Preprocessing/vectorizer.joblib")
idf_transformer = load("./ML_code/Preprocessing/idf_transformer.joblib")
model = load('./ML_code/Model/model.joblib')

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def run_model_text(text):
    new_comment = f'{text}'
    new_df = pd.DataFrame([new_comment])
    new_df[0] = new_df[0].apply(remove_punctuations)
    new_df = new_df.rename(columns={0:"text"})
    X_train_unigram = vectorizer.transform(new_df["text"].values)
    X_train_unigram_tf_idf = idf_transformer.transform(X_train_unigram)
    new_prediction = model.predict(X_train_unigram_tf_idf)
    if new_prediction[0] == 1:
        text_prediction = "Positive"
    else:
        text_prediction = "Negative"
    return text_prediction
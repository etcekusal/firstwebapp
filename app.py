import streamlit as st
st.title("Movie Comment Sentiment Classifier Using Sequential Neural Network")
st.subheader("Try it ! ")

import numpy as np 
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Dropout,Dense,Flatten,Embedding,LSTM
 
def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]',' ',sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+",' ',sentence)
    sentence = re.sub(r'\s+',' ',sentence)
    return sentence
 
TAG_RE = re.compile(r'<[^>]+>')
 
def remove_tags(text):
    return TAG_RE.sub('',text)

import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

num_tokens = len(tokenizer.word_index) + 2
embedding_dim = 100
embedding_layer = Embedding(num_tokens,embedding_dim,trainable=False)
inputs = tf.keras.Input(shape=(None,))
X = embedding_layer(inputs)
X = tf.keras.layers.LSTM(200,return_sequences=True)(X)
X = tf.keras.layers.LSTM(200)(X)
X = Dense(1,activation = 'sigmoid')(X)
model = Model(inputs=inputs,outputs=X)
model.load_weights("model/weights")

def predict(tokenizer,model,comment):
    text = comment
    text = preprocess_text(text)
    text = [text.split(" ")]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text)
    prediction=model.predict(text)
    value = prediction[0][0]
    if value<0.3:
        return "Negative"
    elif value<0.7:
        return "Neutral"
    else:
        return "Positive"
    
def main():
    comment = st.text_input("Enter your comment","Type here...")
    prediction=""
    if st.button("Submit"):
        prediction = predict(tokenizer,model,comment)
        st.success("Your Comment Sentiment is : "+prediction)
    st.subheader("Created By Kusal Bhattacharyya , Department Of ETCE , Jadavpur University")
if __name__ == "__main__":
    main()
    
    
    
    
    
    
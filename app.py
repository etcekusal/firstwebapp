import streamlit as st
import time
st.title("Text Sentiment Classifier Using Sequential Neural Network")
st.subheader("Try it ! It's updated version and Text to speech is added to have more better experience..")
st.subheader("Just type any comment and the model will classify it.\n\n")
import numpy as np 
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model,Sequential
from tensorflow.keras.layers import Dropout,Dense,Flatten,Embedding,LSTM
from gtts import gTTS
from IPython.display import Audio
 
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

#comments_store = open("comments.txt","a")
#feedback_store = open("feedbacks.txt","a")

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
    comment = st.text_area("Enter your comment",height=30)
    #comments_store.write(comment+"\n")
    prediction=""
    col1, col2, col3 , col4, col5 = st.columns(5)
    check = False
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3 :
        check = st.button("Submit")
    if check == True:
            st.text("Almost done ..... ")
            bar = st.progress(0)
            for i in range(100):
                bar.progress(i + 1)
                time.sleep(0.01)
            prediction = predict(tokenizer,model,comment)
            
    if (check==True):
            #st.success("Your Comment Sentiment is : "+prediction+ " , ............ Play the audio to have more fun ... " )
            text = "you have written : " + comment +" : and Your Comment Sentiment is : "+prediction + " : : Please don't foget to give us feedback and after giving it see what happens and run it. Hope you will have more fun"
            tts = gTTS(text,lang='en', tld='ca')
            tts.save("check.mp3")
            #audio = "check.mp3"
    #Audio(audio)
            audio_file = open("check.mp3", 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg',start_time=0)
    feedback = st.text_area("Enter your feedback..",height=20)
    #feedback_store.write(feedback+"\n")
    check = st.button("Submit Feedback")
    if check==True:
          prediction = predict(tokenizer,model,feedback)
          if prediction=="Positive":
                tts = gTTS("Thank you very much for experiencing fun with this.. : Bye ",lang='en', tld='ca')
                tts.save("check.mp3")
                audio_file = open("check.mp3", 'rb')
                audio_bytes = audio_file.read()
                st.text("Play it .... ")
                st.audio(audio_bytes, format='audio/ogg',start_time=0)
          elif prediction=="Neutral":
                tts = gTTS("Thank you very much for your valuable suggestion. : Bye ",lang='en', tld='ca')
                tts.save("check.mp3")
                audio_file = open("check.mp3", 'rb')
                audio_bytes = audio_file.read()
                st.text("Play it .... ")
                st.audio(audio_bytes, format='audio/ogg',start_time=0)
          else:
                tts = gTTS("Sorry to hear that you are not comfortable with us. We are improving for better user experience. : Bye ",lang='en', tld='ca')
                tts.save("check.mp3")
                audio_file = open("check.mp3", 'rb')
                audio_bytes = audio_file.read()
                st.text("Play it .... ")
                st.audio(audio_bytes, format='audio/ogg',start_time=0)
    st.subheader("Created by Kusal Bhattacharyya , Department Of ETCE , Jadavpur University")
if __name__ == "__main__":
    main()
    
    
    
    
    
    

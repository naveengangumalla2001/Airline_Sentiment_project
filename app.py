import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

st.set_page_config(page_title="AI Airline Sentiment Analyzer", page_icon="✈️", layout="centered")

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#EAF6FF,#FFFFFF);}
div.stButton>button{
background:linear-gradient(90deg,#0EA5E9,#2563EB);
color:white;border-radius:10px;width:100%;font-size:18px;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_model.keras")

@st.cache_resource
def load_scaler():
    with open("scaler.pkl","rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl","rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_embeddings():
    with open("embedding_lookup.pkl","rb") as f:
        return pickle.load(f)

model=load_model()
scaler=load_scaler()
encoder=load_encoder()
embedding_lookup=load_embeddings()

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+"," ",text)
    text=re.sub(r"@\w+"," @ ",text)
    text=re.sub(r"[^a-z@ ]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text.split()

def sentence_vector(sentence):
    words=clean_text(sentence)
    vectors=[embedding_lookup[w] for w in words if w in embedding_lookup]
    return np.mean(vectors,axis=0) if vectors else np.zeros(200)

st.markdown("""
<div style='text-align:center'>
<div style='font-size:60px;'>✈️</div>
<h1 style='color:#0E4D92;'>AI Airline Sentiment Analyzer</h1>
<p style='color:gray;'>Analyze airline tweets using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

tweet=st.text_area("📝 Enter Airline Tweet",height=160)

if st.button("✈️ Analyze Sentiment"):
    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:
        with st.spinner("🛫 Aircraft is flying through your tweet..."):
            vec=sentence_vector(tweet)
            vec=scaler.transform(vec.reshape(1,-1))
            pred=model.predict(vec,verbose=0)
        idx=np.argmax(pred)
        sentiment=encoder.inverse_transform([idx])[0]
        conf=pred[0][idx]
        if sentiment.lower()=="positive":
            st.success("😊 Positive")
            st.balloons()
        elif sentiment.lower()=="neutral":
            st.info("😐 Neutral")
        else:
            st.error("😠 Negative")
        st.metric("Confidence",f"{conf*100:.2f}%")
        st.subheader("Prediction Probabilities")
        for lbl,p in zip(encoder.classes_,pred[0]):
            st.write(lbl.capitalize())
            st.progress(float(p))
            st.write(f"{p*100:.2f}%")

st.markdown("---")
st.markdown("<center><small>❤️ TensorFlow • Streamlit • GloVe</small></center>", unsafe_allow_html=True)

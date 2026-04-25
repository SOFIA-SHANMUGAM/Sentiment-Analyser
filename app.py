import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import hashlib

st.set_page_config(page_title='Sentiment App', page_icon='🔐', layout='centered')
MODEL_NAME='Sofia0331/sentiment_model'

@st.cache_resource
def load_model():
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tok, mdl

tokenizer, model = load_model()

if 'users' not in st.session_state:
    st.session_state.users = {'admin': hashlib.sha256('1234'.encode()).hexdigest()}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'history' not in st.session_state:
    st.session_state.history = []

labels={0:'Negative 😡',1:'Neutral 😐',2:'Positive 😊'}

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def predict(text):
    inputs=tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out=model(**inputs)
        pred=torch.argmax(out.logits, dim=1).item()
    return labels[pred]

menu = st.sidebar.selectbox('Menu', ['Login','Register'] if not st.session_state.logged_in else ['Analyzer','History','Logout'])

if menu=='Register':
    st.title('Create Account')
    u=st.text_input('Username')
    p=st.text_input('Password', type='password')
    if st.button('Register'):
        if u in st.session_state.users:
            st.error('User already exists')
        else:
            st.session_state.users[u]=hash_pw(p)
            st.success('Registered successfully')
elif menu=='Login':
    st.title('Login')
    u=st.text_input('Username')
    p=st.text_input('Password', type='password')
    if st.button('Login'):
        if u in st.session_state.users and st.session_state.users[u]==hash_pw(p):
            st.session_state.logged_in=True
            st.session_state.username=u
            st.success('Login successful')
            st.rerun()
        else:
            st.error('Invalid credentials')
elif menu=='Analyzer':
    st.title('Product Review Sentiment Analyser')
    st.write(f"Welcome, {st.session_state.username}")
    txt=st.text_area('Enter Review')
    if st.button('Predict'):
        if txt.strip():
            res=predict(txt)
            st.success(res)
            st.session_state.history.append((txt,res))
        else:
            st.warning('Please enter text')
elif menu=='History':
    st.title('Prediction History')
    for i,(t,r) in enumerate(reversed(st.session_state.history),1):
        st.write(f'{i}. {t} -> {r}')
else:
    st.session_state.logged_in=False
    st.success('Logged out')
    st.rerun()

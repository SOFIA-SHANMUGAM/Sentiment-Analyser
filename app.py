import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import hashlib

st.set_page_config(page_title='Sentiment App', page_icon='📊', layout='centered')

st.markdown('''<style>
.main {background-color:#f8fafc;}
.block-container {padding-top:2rem; max-width:700px;}
.card {background:white; padding:2rem; border-radius:18px; box-shadow:0 10px 25px rgba(0,0,0,0.08);}
.bigtitle {font-size:2.2rem; font-weight:700; color:#0f172a; text-align:center;}
.subtitle {text-align:center; color:#475569; margin-bottom:1rem;}
.stButton>button {width:100%; border-radius:10px; font-weight:600;}
</style>''', unsafe_allow_html=True)

MODEL_NAME='Sofia0331/sentiment_model'
@st.cache_resource
def load_model():
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tok, mdl

tokenizer, model = load_model()

if 'users' not in st.session_state:
    st.session_state.users={'admin': hashlib.sha256('1234'.encode()).hexdigest()}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in=False
if 'history' not in st.session_state:
    st.session_state.history=[]

labels={0:'Negative 😡',1:'Neutral 😐',2:'Positive 😊'}
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()
def predict(text):
    inputs=tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out=model(**inputs)
        pred=torch.argmax(out.logits, dim=1).item()
    return labels[pred]

menu = st.sidebar.selectbox('Navigation', ['Login','Register'] if not st.session_state.logged_in else ['Analyzer','History','Logout'])

if menu=='Login':
    st.markdown("<div class='card'><div class='bigtitle'>Welcome Back</div><div class='subtitle'>Login to continue</div></div>", unsafe_allow_html=True)
    u=st.text_input('Username')
    p=st.text_input('Password', type='password')
    if st.button('Login'):
        if u in st.session_state.users and st.session_state.users[u]==hash_pw(p):
            st.session_state.logged_in=True; st.session_state.username=u; st.rerun()
        else: st.error('Invalid credentials')
elif menu=='Register':
    st.markdown("<div class='card'><div class='bigtitle'>Create Account</div><div class='subtitle'>Join the platform</div></div>", unsafe_allow_html=True)
    u=st.text_input('Choose Username')
    p=st.text_input('Choose Password', type='password')
    if st.button('Register'):
        if u in st.session_state.users: st.error('User already exists')
        else:
            st.session_state.users[u]=hash_pw(p); st.success('Registered successfully')
elif menu=='Analyzer':
    st.markdown(f"<div class='card'><div class='bigtitle'>Product Review Sentiment Analyser</div><div class='subtitle'>Hello, {st.session_state.username}</div></div>", unsafe_allow_html=True)
    txt=st.text_area('Enter Review')
    if st.button('Predict'):
        if txt.strip():
            res=predict(txt); st.success(f'Prediction: {res}'); st.session_state.history.append((txt,res))
        else: st.warning('Please enter text')
elif menu=='History':
    st.markdown("<div class='bigtitle'>Prediction History</div>", unsafe_allow_html=True)
    for i,(t,r) in enumerate(reversed(st.session_state.history),1): st.write(f'{i}. {t} → {r}')
else:
    st.session_state.logged_in=False; st.success('Logged out'); st.rerun()

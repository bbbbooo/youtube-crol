#-*- coding: utf-8 -*-
from email import header
from operator import truediv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import SCORERS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from konlpy.tag import Okt
import re
from pykospacing import Spacing
import pandas as pd
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import streamlit as st
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from konlpy.tag import Okt
from keras.models import load_model
import speech_recognition as sr
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from konlpy.tag import Kkma


def Analysis():
    tokenizer = Tokenizer()
    okt = Okt()
    max_len = 30

    # 불용어
    stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '줄', '를', '을', '에', '에게', '께', '한테', '더러', '에서', '에게서',
                 '한테서', '로', '으로', '와', '과', '도', '부터', '도', '만', '이나', '나', '라도', '의', '거의', '겨우', '결국', '그런데', '즉', '참', '챗', '할때', '할뿐', '함께', '해야한다', '휴']

    PATH = '/Users/82102/Desktop/project/yt_cr/backup/'
    # PATH = './model/'
    PATH2 = '/Users/82102/Desktop/project/yt_cr/backup/'
    # PATH2 = './model/'

    #모델 및 토큰 불러오기
    model = load_model(PATH + 'best_model.h5')
    with open(PATH2+'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # 감정 예측
    def sentiment_predict(new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', str(new_sentence))
        new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
        score = float(model.predict(pad_new))  # 예측

        if(score > 0.6):
            st.markdown("<h2 style='text-align: center; '>분석 결과</h2>", unsafe_allow_html=True)
            streamlit_positive_1()
            st.success("전체 문장의 분석 결과, 감정 점수 {:.2f}점 으로 강한 긍정의 문장입니다.\n".format(score * 100))
            st.balloons()
            detail()
        elif(0.6> score > 0.5):
            st.markdown("<h2 style='text-align: center; '>분석 결과</h2>", unsafe_allow_html=True)
            streamlit_positive_2()
            st.success("전체 문장의 분석 결과, 감정 점수 {:.2f}점 으로 긍정적인 문장입니다.\n".format(score * 100))
            st.balloons()
            detail()
        elif(0.5> score > 0.4):
            st.markdown("<h2 style='text-align: center; '>분석 결과</h2>", unsafe_allow_html=True)
            streamlit_neutrality()
            st.warning("전체 문장의 분석 결과, 감정 점수 {:.2f}점 으로 중립적인 문장입니다.\n".format(score * 100))
            st.balloons()
            detail()
        elif(0.4> score > 0.3):
            st.markdown("<h2 style='text-align: center; '>분석 결과</h2>", unsafe_allow_html=True)
            streamlit_negative_1()
            st.error("전체 문장의 분석 결과, 감정 점수 {:.2f}점 으로 부정적인 문장입니다.\n".format(score * 100))
            st.snow()
            detail()
        else:
            st.markdown("<h2 style='text-align: center; '>분석 결과</h2>", unsafe_allow_html=True)
            streamlit_negative_2()
            st.error("전체 문장의 분석 결과, 감정 점수 {:.2f}점 으로 강한 부정의 문장입니다.\n".format(score * 100))
            st.snow()
            detail()

    sentiment_predict(user_input)


 #리스트 문자열로 변환
def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()


def detail():

    text = user_input

    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', text)
    new_sent = text.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기

    spacing = Spacing()
    kospacing_text = spacing(new_sent)

    

    ###################################################################형태소 분석

    # okt = Okt()

    # example = kospacing_text

    
    # # stop_words = set(stop_words.split(' '))
    # word_tokens = okt.morphs(example)

    # st.info(word_tokens)
    
    ######################################################################형태소 + 불용어
    kkma = Kkma()
    # len(ex_nouns)
    ex_pos = kkma.morphs(kospacing_text) 

    # text_data = [] 
    # for (text, tclass) in ex_pos : # ('형태소', 'NNG') 
    #     if tclass == 'NNG' or tclass == 'NNP' or tclass == 'NP' : 
    #         text_data.append(text)

    good_text = " ".join(ex_pos)
    st.info(good_text)

    ###################################################################

    texxt = open("./data/fear.txt","r",encoding='UTF-8')
    lists = texxt.readlines()
    result_fear = listToString(lists)
    texxt.close()
    #공포
    sentence = (result_fear,good_text)

    texxt2 = open("./data/pleasure.txt","r",encoding='UTF-8')
    lists2 = texxt2.readlines()
    result_pleasure = listToString(lists2)
    texxt2.close()
    #기쁨
    sentence2 =(result_pleasure,good_text)  

    texxt3 = open("./data/anger.txt","r",encoding='UTF-8')
    lists3 = texxt3.readlines()
    result_anger = listToString(lists3)
    texxt3.close()
    #분노
    sentence3=(result_anger,good_text)
        
    texxt4 = open("./data/love.txt","r",encoding='UTF-8')
    lists4 = texxt4.readlines()
    result_love = listToString(lists4)
    texxt4.close()
    #사랑
    sentence4=(result_love,good_text)

    texxt6 = open("./data/sad.txt","r",encoding='UTF-8')
    lists6 = texxt6.readlines()
    result_sad = listToString(lists6)
    texxt6.close()
    #슬픔
    sentence6=(result_sad,good_text)

    # 객체 생성
    tfidf_vectorizer = TfidfVectorizer()
    # 문장 벡터화 진행
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
    tfidf_matrix2 = tfidf_vectorizer.fit_transform(sentence2)
    tfidf_matrix3 = tfidf_vectorizer.fit_transform(sentence3)
    tfidf_matrix4 = tfidf_vectorizer.fit_transform(sentence4)
    tfidf_matrix6 = tfidf_vectorizer.fit_transform(sentence6)

    # 각 단어
    text = tfidf_vectorizer.get_feature_names()
    # 각 단어의 벡터 값
    idf = tfidf_vectorizer.idf_

    manhattan_distances(tfidf_matrix[0:1], tfidf_matrix[1:2])
    manhattan_distances(tfidf_matrix2[0:1], tfidf_matrix2[1:2])
    manhattan_distances(tfidf_matrix3[0:1], tfidf_matrix3[1:2])
    manhattan_distances(tfidf_matrix4[0:1], tfidf_matrix4[1:2])
    manhattan_distances(tfidf_matrix6[0:1], tfidf_matrix6[1:2])

    tokenized_doc1 = set(sentence[0].split(' '))
    tokenized_doc2 = set(sentence[1].split(' '))
    tokenized_doc5 = set(sentence2[0].split(' '))
    tokenized_doc6 = set(sentence2[1].split(' '))
    tokenized_doc7 = set(sentence3[0].split(' '))
    tokenized_doc8 = set(sentence3[1].split(' '))
    tokenized_doc9 = set(sentence4[0].split(' '))
    tokenized_doc10 = set(sentence4[1].split(' '))
    tokenized_doc13 = set(sentence6[0].split(' '))
    tokenized_doc14 = set(sentence6[1].split(' '))


    union = set(tokenized_doc1).union(set(tokenized_doc2))
    union2 = set(tokenized_doc5).union(set(tokenized_doc6))
    union3 = set(tokenized_doc7).union(set(tokenized_doc8))
    union4 = set(tokenized_doc9).union(set(tokenized_doc10))
    union6 = set(tokenized_doc13).union(set(tokenized_doc14))


    intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
    intersection2 = set(tokenized_doc5).intersection(set(tokenized_doc6))
    intersection3 = set(tokenized_doc7).intersection(set(tokenized_doc8))
    intersection4 = set(tokenized_doc9).intersection(set(tokenized_doc10))
    intersection6 = set(tokenized_doc13).intersection(set(tokenized_doc14))


    

    Score = len(intersection)/len(union)*1000
    Score2 = len(intersection2)/len(union2)*1000
    Score3 = len(intersection3)/len(union3)*1000
    Score4 = len(intersection4)/len(union4)*1000
    Score6 = len(intersection6)/len(union6)*1000

    print(Score)
    print(Score2)
    print(Score3)
    print(Score4)
    print(Score6)


    def ORDER():
        dict_test = {
            '감정': ['공포', '기쁨', '분노', '사랑', '슬픔'],
            '유사도': [Score, Score2, Score3, Score4, Score6],
        }
        dict2_test = {
            '순위': [1,2,3,4,5]
        }

        # 감정 및 유사도
        df_test = pd.DataFrame(dict_test)
        df_test = df_test[df_test['유사도'] > 0]
        df_test = df_test.sort_values(by=['유사도'], ascending=False)
        
        # 순위
        df2_test = pd.DataFrame(dict2_test)


        df = df_test['감정']
        df_list = []


        for df_cell in df:
            df_list.append(df_cell)    

        df_finish =  pd.DataFrame({'감정' : df_list})

        result = pd.concat([df2_test, df_finish], axis=1)
        result = result[result['감정'].notna()]
        result.set_index('순위', inplace=True)


        st.markdown("<h2 style='text-align: center; '>세부 감정 분석 결과</h2>", unsafe_allow_html=True)
        
        
        st.table(result)

        if Score > 0:
            st.warning("공포와 관련된 단어")
            st.info(intersection)
        if Score2 > 0:
            st.success("기쁨과 관련된 단어")
            st.info(intersection2)
        if Score3 > 0:
            st.error("분노와 관련된 단어")
            st.info(intersection3)
        if Score4 > 0:
            st.success("사랑과 관련된 단어")
            st.info(intersection4)
        if Score6 > 0:
            st.warning("슬픔과 관련된 단어")
            st.info(intersection6)


    ORDER()
    

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def streamlit_title():
    lottie_url_title = "https://assets5.lottiefiles.com/private_files/lf30_XPbZB7.json"
    lottie_title = load_lottieurl(lottie_url_title)
    st_lottie(lottie_title, key="title", height=300)

def streamlit_analysis():
    lottie_url_title = "https://assets5.lottiefiles.com/private_files/lf30_XPbZB7.json"
    lottie_title = load_lottieurl(lottie_url_title)
    st_lottie(lottie_title, key="title", height=300)

def streamlit_positive_1():
    lottie_url_positive_1 = "https://assets3.lottiefiles.com/packages/lf20_wml3ec.json"
    lottie_positive_1 = load_lottieurl(lottie_url_positive_1)
    st_lottie(lottie_positive_1, key="analysis", height=300)

def streamlit_positive_2():
    lottie_url_positive_2 = "https://assets3.lottiefiles.com/packages/lf20_lopDQz.json"
    lottie_positive_2 = load_lottieurl(lottie_url_positive_2)
    st_lottie(lottie_positive_2, key="analysis", height=300)

def streamlit_neutrality():
    lottie_url_neutrality = "https://assets3.lottiefiles.com/packages/lf20_VR4DyP.json"
    lottie_neutrality = load_lottieurl(lottie_url_neutrality)
    st_lottie(lottie_neutrality, key="analysis", height=300)

def streamlit_negative_1():
    lottie_url_negative_1 = "https://assets3.lottiefiles.com/packages/lf20_7DnubQ.json"
    lottie_negative_1 = load_lottieurl(lottie_url_negative_1)
    st_lottie(lottie_negative_1, key="analysis", height=300)

def streamlit_negative_2():
    lottie_url_negative_2 = "https://assets3.lottiefiles.com/packages/lf20_CZ5mts.json"
    lottie_negative_2 = load_lottieurl(lottie_url_negative_2)
    st_lottie(lottie_negative_2, key="analysis", height=300)






##############################################################STT
def Audio():
    # microphone에서 auido source를 생성합니다

    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.success("음성녹음 시작")
        audio = r.listen(source)

        st.error("음성녹음 종료")

    global user_input
    user_input = r.recognize_google(audio, language='ko')

def start():
    if st.form_submit_button("마이크 ON"):
        #음성녹음 시작
        Audio()

        #감정 시작
        if st.form_submit_button and user_input:
            st.markdown("<h1 style='text-align: center; '>작성 내용</h1>", unsafe_allow_html=True)
            st.info(user_input)
            Analysis()
##############################################################



st.markdown("<h1 style='text-align: center; '>테스트 심리 분석</h1>", unsafe_allow_html=True)

streamlit_title()

with st.form('form', clear_on_submit=True):
    st.success("아래에 내용을 입력해주세요")
    user_input = st.text_input('')
    st.form_submit_button('전송')

if st.form_submit_button and user_input:
    st.info(user_input)
    Analysis()

# with st.form('stt', clear_on_submit=True):
#     st.success("아래에 마이크를 입력해주세요")
#     start()
from genericpath import exists
from logging import exception
import time
import pandas as pd
from googleapiclient.discovery import build
import os
import re
from sqlalchemy import true
import streamlit as st
import pafy as pa
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
from PIL import Image
import pickle

okt = Okt()

# 웹 제목
st.title("Youtube-CR")



# 주소 입력
input_url = st.text_input(label="URL", value="")
url=input_url
my_str = url.replace("https://www.youtube.com/watch?v=","")


# 주소 뒤 시간이 있다면..
def num_re():
    # range(int) -> int는 시간값
    for i in range(10000000):
        # &t= 코드는 고정
        if my_str.find("&t="):
            # 찾았다면 repplace
            temp ="&t=%ss" %i
            str = my_str.replace(temp, "")
            
            # find 결과가 false면 -1 리턴. 변환 후엔 당연히 false가 반환
            if str.find("&t=")==-1:
                return str
            else:
                # 쓰레기 값. if문을 쓰기 위함
                a=1
        else:
            # 시간 안적혀 있으면 그대로 리턴
            return str 
    return str
  
# 주소 뒤 시간 제거
my_str2 = num_re()

#썸네일 출력
def get_thumbnail(url):
    id = url
    img = 'https://img.youtube.com/vi/{}/0.jpg'.format(id)
    return img    

# 제목 가져와서 변환 후 변환 값 return
def title_get():
    videoinfo = pa.new(url)
    video_title = videoinfo.title

    #제목 특수기호 있으면 공백으로 치환
    rp_video_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', video_title)
    return rp_video_title

# 댓글 크롤링하여 xlsx 형태로 video_xlsx 폴더에 저장
def Crawling():
    #api키 입력
    api_key = 'AIzaSyBwKI6s7TsyJ1yNNvRcJ50SuhiLyNqHdSs'
    comments = list()
    api_obj = build('youtube', 'v3', developerKey=api_key)
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str2, maxResults=100).execute()
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str2, pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break
    df = pd.DataFrame(comments)
    df = df.astype(str)
    df.to_excel('./video_xlsx/%s.xlsx' % (title_get()), header=['comment', 'author', 'date', 'num_likes'], index=None)
    path = './video_xlxs/%s.xlsx' % (title_get())
    while os.path.exists(path) :
        df.to_excel('./video_xlsx/%s.xlsx' % (title_get()), header=['comment', 'author', 'date', 'num_likes'], index=None)
        break
       
contain = []            #긍정 cell
contain_number =[]      #긍정 확률
contain2 = []           #부정 cell
contain2_number = []    #부정 확률
contain3 = []           #중립
contain3_number = []    #중립 확률

def Analysis():
    tokenizer = Tokenizer()
    okt = Okt()
    max_len = 30

    # 불용어
    stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한','줄','를','을','에','에게','께','한테','더러','에서','에게서','한테서','로','으로','와','과','도','부터','도','만','이나','나','라도','의'
                 , '거의', '겨우', '결국', '그런데', '즉', '참', '챗', '할때', '할뿐', '함께', '해야한다', '휴']
    
    #PATH = '/Users/82102/Desktop/project/yt_cr/backup/'
    PATH = '/Users/82102/Desktop/project/yt_cr/model_test/'
    #PATH2 = '/Users/82102/Desktop/project/yt_cr/backup/'
    PATH2 = '/Users/82102/Desktop/project/yt_cr/model_test/'
    
    #모델 및 토큰 불러오기
    model = load_model(PATH + 'best_model.h5')
    with open(PATH2+'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


    # 감정 예측
    def sentiment_predict(new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(new_sentence))
        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(model.predict(pad_new)) # 예측
        
        # 긍정적이라면 contain 리스트에 추가
        if(score > 0.7):
            contain.append(list)
            contain_number.append(score * 100)
        # 중립이라면 contain3 리스트에 추가
        elif 0.5 < score < 0.7:
            contain3.append(list)
            contain3_number.append(score*100)
        # 부정적이라면 contain2 리스트에 추가
        else:
            contain2.append(list)
            contain2_number.append( (1 - score) * 100)


    # 다른 함수에서도 쓰기 위해 global(전역변수) 선언
    global filename, sheet, date, likes
    filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlsx/%s.xlsx' % title_get())
    sheet = filename['comment']

    # comment 칼럼의 각각의 데이터를 읽기
    for cell in sheet:
        list = []
        output_sentence = str(cell)
        
        # 댓글에 html 코드가 존재한다면...
        if "</a>" in output_sentence:
            split = output_sentence.split('</a>')

            # 단순 시간만 적혀있었다면 split은 공백값이 남음. split의 1번째 index에 '' <<(이름 모를 공백값, NULL값 아님)이 있다면 출력에서 제외
            if split[1] == '':
                continue
            # 없으면 list에 저장
            else:
                for c in split:
                    split2 = re.sub('(<([^>]+)>)','', c)
                list.append(split2)
            
        # 아무것도 해당 안되면 바로 list에 추가
        else:
            split = re.sub('(<([^>]+)>)','', output_sentence)
            list.append(split)

        # 감정 예측
        sentiment_predict(list)

# 리스트를 문자열 형태로 변환(코드 이해용)
def list_to_str(list):
    # 결과를 담을 공백 리스트 생성
    data_list = []
    # 인자로 받아온 리스트를 for문으로 돌린 후 data_list에 저장
    for i in list:
        data_list.append(i)
    # 저장된 data_list를 str(문자열) 형태로 변환 후 return
    return str(data_list)

# 긍정 워드 클라우드
def Create_pword():
    # 리스트를 문자열 형태로 변환
    # 1. '(요소 사이에 넣을 문자)'.join(str(변수) for 변수 in 리스트)
    # 2. 리스트 내의 요소를 변수에 대입하고 변수를 str(문자열) 형태로 변환 
    pos = ''.join([str(n) for n in contain])
    
    # 190줄을 풀어서 설명한 코드
    # pos = list_to_str(contain)
    
    # 데이터 전처리
    pn = okt.nouns(pos)
    # 문장의 길이가 1은 제외
    pw = [n for n in pn if len(n) > 1]
    
    pc = Counter(pw)
    pwc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250)
    
    pg = pwc.generate_from_frequencies(pc)
    pfig = plt.figure()
    
    plt.imshow(pg, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('./result_wc/%s_positive.png' % title_get())
    plt.show()

    st.markdown('긍정')
    st.pyplot(pfig)
    
# 부정 워드 클라우드
def Create_nword():
    neg = ''.join([str(n) for n in contain2])
    # neg = list_to_str(contain2)
    
    nn = okt.nouns(neg)
    nw = [n for n in nn if len(n) > 1]
    nc = Counter(nw)
    nwc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250)
    
    ng = nwc.generate_from_frequencies(nc)
    nfig = plt.figure()
    
    plt.imshow(ng, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('./result_wc/%s_negative.png' % title_get())
    plt.show()
    
    st.markdown('부정')
    st.pyplot(nfig)
   
# 중립 워드 클라우드
def Create_aword():
    neu = ''.join([str(n) for n in contain3])
    # neg = list_to_str(contain2)
    
    nn = okt.nouns(neu)
    nw = [n for n in nn if len(n) > 1]
    nc = Counter(nw)
    nwc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250)
    
    ng = nwc.generate_from_frequencies(nc)
    nfig = plt.figure()
    
    plt.imshow(ng, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('./result_wc/%s_neutral.png' % title_get())
    plt.show()
    
    st.markdown('중립')
    st.pyplot(nfig)   

# 원형 차트 생성
# 1. 저장된 엑셀 파일의 comments 길이를 계산
# 2. 긍정, 부정으로 저장된 comments의 길이를 계산, 이를 활용해 비율을 계산
# 3. 이를 토대로 원형 차트 출력 
def Create_plot():
  global allen, poslen, neglen, neulen
  allen = len(sheet)
  poslen = len(pd_contain)
  neglen = len(pd_contain2)
  neulen = len(pd_contain3)
  
  pos_ratio = (poslen/allen) * 100
  neg_ratio = (neglen/allen) * 100
  neu_ratio = (neulen/allen) * 100
  
  labels = ['Positive', 'Negative', 'Neutral']
  ratio = pos_ratio, neg_ratio, neu_ratio
  
  fig, ax = plt.subplots()
  ax.pie(ratio, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
  ax.axis('equal')
  plt.savefig('./result_image/%s_chart.png' % title_get())
  
  st.pyplot(fig)




if st.sidebar.button('기록 1'):
    path = './result_image/'
    path1 = './result_wc/'
    
    if os.path.isfile(path + '%s_chart.png' % title_get()):
        def load_chart():
            chart = Image.open(path + '%s_chart.png' % title_get())
            return chart
        def load_pwc():
            pwc = Image.open(path1 + '%s_positive.png' % title_get())
            return pwc
        def load_nwc():
            nwc = Image.open(path1 + '%s_negative.png' % title_get())
            return nwc
        def load_awc():
            nwc = Image.open(path1 + '%s_neutral.png' % title_get())
            return nwc
        
        rep = pd.read_excel('./result_video/%s_positive.xlsx' % title_get())
        ren = pd.read_excel('./result_video/%s_negative.xlsx' % title_get())
        rea = pd.read_excel('./result_video/%s_neutral.xlsx' % title_get())
        
        st.write(rep)
        st.write(ren)
        st.write(rea)
        
        chart = load_chart()
        pwc = load_pwc()
        nwc = load_nwc()
        awc = load_awc()
        
        st.image(chart)
        st.image(pwc)
        st.image(nwc)
        st.image(awc)
    else:
        st.write("저장된 기록이 없습니다.")
            



# 댓글 분석 눌렀을때...
def Youtube_Comments_Analysis():
    # Search 버튼 클릭 시....
    if st.button("검색"):
        con = st.container()
        with st.spinner("검색중...."):
            time.sleep(2)
        st.success("검색을 완료됐습니다. 댓글 개수이 많아질수록 분석 시간도 증가합니다.")

        # 결과(입력 주소) 출력
        con.caption("검색 결과")
        con.write("입력하신 주소는 %s 입니다." % input_url)
        
        # 썸네일 출력
        st.header('썸네일')
        st.image(get_thumbnail(my_str2))

        # 댓글 크롤링
        Crawling()
            
        # 데이터 분석
        Analysis()
        
        # 긍정 댓글, 확률
        global pd_contain, pd_contain2, pd_contain3, pos_result, neg_result, neu_result
        pd_contain = pd.DataFrame({'긍정 댓글' : contain})
        pd_contain_number = pd.DataFrame({'확률': contain_number})
        pos_result = pd.concat([pd_contain, pd_contain_number], axis=1)

        
        
        # 부정 댓글, 확률
        pd_contain2 = pd.DataFrame({'부정 댓글' : contain2})
        pd_contain_number2 = pd.DataFrame({'확률': contain2_number})
        neg_result = pd.concat([pd_contain2, pd_contain_number2], axis=1)
        
        # 중립 댓글, 확률
        pd_contain3 = pd.DataFrame({'중립 댓글' : contain3})
        pd_contain_number3 = pd.DataFrame({'확률': contain3_number})
        neu_result = pd.concat([pd_contain3, pd_contain_number3], axis=1)
        
        
        
        # 결과 저장
        pos_result.to_excel('./result_video/%s_positive.xlsx' % title_get(), header=['comments', 'Probability'])
        neg_result.to_excel('./result_video/%s_negative.xlsx' % title_get(), header=['comments', 'Probability'])
        neu_result.to_excel('./result_video/%s_neutral.xlsx' % title_get(), header=['comments', 'Probability'])


        # 원형 차트 출력
        st.header('원형 차트')
        Create_plot()
        
        
        # 데이터 프레임
        st.header("긍정(개수 : %s)" % poslen)
        st.dataframe(pos_result)
        
        st.header("부정(개수 : %s)" % neglen)
        st.dataframe(neg_result)
        
        st.header("중립(개수 : %s)" % neulen)
        st.dataframe(neu_result)
        
        
        # 워드 클라우드 출력
        st.header('워드 클라우드')
        Create_pword()
        Create_nword()
        Create_aword()
    



Youtube_Comments_Analysis()
from selenium import webdriver
import time
from openpyxl import Workbook
import pandas as pd
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from sqlalchemy import null
from googleapiclient.discovery import build
import os
import re
import streamlit as st


from selenium import webdriver
import time
from openpyxl import Workbook
import pandas as pd
import pafy as pa
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

try:
    wb = Workbook(write_only=True)
    ws = wb.create_sheet()


    st.title("Youtube-CR")

    input_url = st.text_input(label="URL", value="")
        


    if st.button("Search"):
        con = st.container()
        con.caption("Result")
        con.write(f"The entered video address is {str(input_url)}")

    url= ""
    url=input_url
    my_str = url.replace("https://www.youtube.com/watch?v=","")

    #제목 가져오기
    videoinfo = pa.new(url)
    video_title = videoinfo.title

    #제목 특수기호 있으면 공백으로 치환
    rp_video_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', video_title)


    comments = list()
    api_obj = build('youtube', 'v3', developerKey='AIzaSyDCLqtKIMyBZ82hWpUj1QcTg_glkAlk1kk')
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str, maxResults=100).execute()

    

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
        
            # 대댓글 불러오기
            # if item['snippet']['totalReplyCount'] > 0:
            #     for reply_item in item['replies']['comments']:
            #         reply = reply_item['snippet']
            #         comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
 
        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(part='snippet,replies', videoId='sWC-pp6CXpA', pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break

    df = pd.DataFrame(comments)
    df.to_excel('./video_xlxs/%s.xlsx' % (rp_video_title), header=['comment', 'author', 'date', 'num_likes'], index=None)

    path = './video_xlxs/%s.xlsx' % (rp_video_title)
    while os.path.exists(path) :
        df.to_excel('./video_xlxs/%s.xlsx' % (rp_video_title), header=['comment', 'author', 'date', 'num_likes'], index=None)
        break
    

except:
    st.error("Please Write URL")
    st.stop()
    
st.success("The search was successful. If you want to exit, press Ctrl + C")

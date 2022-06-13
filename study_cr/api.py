from ast import pattern
import os
import pandas as pd
from googleapiclient.discovery import build
import warnings
import pafy as pa
import re
import warnings # 경고창 무시
warnings.filterwarnings('ignore')


url = ''
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
 
        # if item['snippet']['totalReplyCount'] > 0:
        #     for reply_item in item['replies']['comments']:
        #         reply = reply_item['snippet']
        #         comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
 
    if 'nextPageToken' in response:
        response = api_obj.commentThreads().list(part='snippet,replies', videoId='sWC-pp6CXpA', pageToken=response['nextPageToken'], maxResults=100).execute()
    else:
        break

df = pd.DataFrame(comments)
df.to_excel('%s.xlsx' % (rp_video_title), header=['comment', 'author', 'date', 'num_likes'], index=None)

path = '%s.xlsx' % (rp_video_title)
while os.path.exists(path) :
      df.to_excel('%s.xlsx' % (rp_video_title), header=['comment', 'author', 'date', 'num_likes'], index=None)
      break

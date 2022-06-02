import pandas as pd
from googleapiclient.discovery import build
import warnings

warnings.filterwarnings('ignore')

url = 'YEceGmGAQ3I'

comments = list()
api_obj = build('youtube', 'v3', developerKey='AIzaSyCk8DRWHLfZ4HRTKoJonavXs98ldzuBWJ0')
response = api_obj.commentThreads().list(part='snippet,replies', videoId=url, maxResults=100).execute()




while response:
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
 
        if item['snippet']['totalReplyCount'] > 0:
            for reply_item in item['replies']['comments']:
                reply = reply_item['snippet']
                comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
 
    if 'nextPageToken' in response:
        response = api_obj.commentThreads().list(part='snippet,replies', videoId='sWC-pp6CXpA', pageToken=response['nextPageToken'], maxResults=100).execute()
    else:
        break

df = pd.DataFrame(comments)
df.to_excel('%s.xlsx' % (url), header=['comment', 'author', 'date', 'num_likes'], index=None)

path = '%s.xlsx' % (url)
while os.path.exists(path) :
      df.to_excel('%s.xlsx' % (url), header=['comment', 'author', 'date', 'num_likes'], index=None)
      break

exit()
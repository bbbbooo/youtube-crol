from matplotlib.pyplot import title
from selenium import webdriver
from openpyxl import Workbook
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from sqlalchemy import null
import os
import time
import pandas as pd
import pafy as pa

wb = Workbook(write_only=True)
ws = wb.create_sheet()


url= "https://www.youtube.com/watch?v=JYV9TZJ1beE"
my_str = url.replace("https://www.youtube.com/watch?v=","")

# 제목 가져오기
videoinfo = pa.new(url)
video_title = videoinfo.title

# 크롬 드라이버 실행..
driver = webdriver.Chrome(executable_path="chromedriver.exe")
driver.get(url)
driver.implicitly_wait(3)

time.sleep(1.5)

driver.execute_script("window.scrollTo(0, 800)")
time.sleep(3)

# 페이지 끝까지 스크롤
last_height = driver.execute_script("return document.documentElement.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(1.5)

    new_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

time.sleep(1.5)

# 팝업 닫기
try:
    driver.find_element_by_css_selector("#dismiss-button > a").click()
except:
    pass

# 대댓글 모두 열기
# buttons = driver.find_elements_by_css_selector("#more-replies > a")

# time.sleep(1.5)

# for button in buttons:
#     button.send_keys(Keys.ENTER)
#     time.sleep(1.5)
#     button.click()

# time.sleep(1.5)

# 정보 추출하기
html_source = driver.page_source
soup = BeautifulSoup(html_source, 'html.parser')

id_list = soup.select("div#header-author > h3 > #author-text > span")
comment_list = soup.select("yt-formatted-string#content-text")

id_final = []
comment_final = []

for i in range(len(comment_list)):
    temp_id = id_list[i].text
    temp_id = temp_id.replace('\n', '')
    temp_id = temp_id.replace('\t', '')
    temp_id = temp_id.replace('    ', '')
    id_final.append(temp_id)

    temp_comment = comment_list[i].text
    temp_comment = temp_comment.replace('\n', '')
    temp_comment = temp_comment.replace('\t', '')
    temp_comment = temp_comment.replace('    ', '')
    comment_final.append(temp_comment)

pd_data = {"아이디" : id_final , "댓글 내용" : comment_final}
youtube_pd = pd.DataFrame(pd_data)

# 유튜브 제목으로 저장
youtube_pd.to_excel(video_title + '.xlsx')

# 중복 처리
path = '%s.xlsx' % (video_title)
if os.path.exists(path):
    youtube_pd.to_excel('%s.xlsx' % (video_title))

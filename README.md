# youtube-crol

유튜브 댓글 정보를 가져와 댓글을 분석, positive 와 negative로 분류.
해당 영상의 positive, negative 비율을 출력

model은 모델 만드는 폴더
model 폴더에서 dataset은 rating_data, 생성된 모델은 save_model, 토큰값은 token에 저장된다.
model_test는 다른 사람이 생성한 모델과 토큰값을 저장하는 폴더


메인코드는 main.py
테스트 코드는 test.py (url, 크롤링 부분 제외)
실행코드는 straemlit run main.py
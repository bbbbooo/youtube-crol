import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import os
import openpyxl
from konlpy.tag import Okt
from tqdm import tqdm
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
 

# 데이터셋 다운로드. 완료하였다면 주석처리 할 것.
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('/Users/82102/Desktop/project/yt_cr/model/rating_data/ratings_train.txt')
test_data = pd.read_table('/Users/82102/Desktop/project/yt_cr/model/rating_data/ratings_test.txt')

# document 열과 label 열의 중복을 제외한 값의 개수
train_data['document'].nunique(), train_data['label'].nunique()

# document 열의 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True)
train_data['label'].value_counts().plot(kind = 'bar')

#널값을 가진 샘플이 어디 인덱스에 위치했는지..
train_data.loc[train_data.document.isnull()]
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거

# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]","")
# train_data[:5]
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how = 'any')

# test도 똑같이데이터 전처리
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]","") # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거

#불용어 처리
stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한','줄','를','을','에','에게','께','한테','더러','에서','에게서','한테서','로','으로','와','과','도','부터','도','만','이나','나','라도','의']

okt = Okt()

X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)
    

X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)
    
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    
    
max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


path = '/Users/82102/Desktop/project/yt_cr/model/save_model/best_model.h5'
PATH = '/Users/82102/Desktop/project/yt_cr/model/save_model/'
if os.path.isfile(path):
    print("File exists")
else:
    embedding_dim = 256 # 임베딩 벡터의 차원
    dropout_ratio = 0.3 # 드롭아웃 비율
    num_filters = 256 # 커널의 수
    kernel_size = 3 # 커널의 크기
    hidden_units = 128 # 뉴런의 수
    
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Dropout(dropout_ratio))
    model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(1, activation='sigmoid'))
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(PATH + 'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min', min_lr= 1e-6)
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train, epochs=15, callbacks=[es,mc,rp], batch_size=256, validation_split=0.2)

    
    
    PATH2 = '/Users/82102/Desktop/project/yt_cr/model/token/'
    with open(PATH2 + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)


loaded_model = load_model(PATH+'best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
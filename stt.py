import speech_recognition as sr
import os
import sys
import streamlit as st
from pydub import AudioSegment

#-----------------------------------------------------------------------------------




# 업로드
def upload():
    uploaded_file = st.file_uploader("Choose a file", type =(["mp3", "wav"]))
    if uploaded_file is not None:
        #bytes_data = 오디오 파일
        global bytes_data
        bytes_data = uploaded_file.name
        st.success('파일을 업로드 했습니다. : {} '.format(bytes_data))
        
    # elif not os.path.exists(path):
    #     os.mkdir(path)
    # with open(os.path.join(path, bytes_data), 'wb') as f:
    #     f.write(uploaded_file.getbuffer())
    #     st.success('파일을 업로드 했습니다. : {} '.format(bytes_data))
        
    
# 음성 파일 불러와서 텍스트로 전환
def STT():
    os.chdir(path)
    # 파일명과 확장자 분리
    name, ext = os.path.splitext(filepath)
    
    # wav
    if ext == ".wav":
        harvard_audio = sr.AudioFile(filepath)
        with harvard_audio as source:
            audio = r.record(source)
    # mp3
    elif ext == '.mp3':
        mp3_sound = AudioSegment.from_mp3(filepath)
        wav_sound = mp3_sound.export("{0}.wav".format(name), format="wav")
        harvard_audio = sr.AudioFile(wav_sound)
        with harvard_audio as source:
            audio = r.record(source, duration=150)
    # 나머지..
    else:
        st.write("wav 와 mp3 형식만 호환됩니다.")


    # 텍스트 파일로 전환
    try:
        sys.stdout = open('%s.txt' % name, 'w', encoding='UTF-8') 
        text = r.recognize_google(audio, language='ko-KR')
        print(text)
        sys.stdout.close()
    except Exception as e:
        print(e)
       
       
       
       
# def Anal
#---------------------------------------------------------------------------

st.header('Text To Speech')
r = sr.Recognizer()
path = './audio/'
upload()
filepath= bytes_data
STT()
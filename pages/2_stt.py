import speech_recognition as sr
import os
import sys
import streamlit as st
from pydub import AudioSegment

r = sr.Recognizer()

#-----------------------------------------------------------------------------------

# 업로드
def upload():
    uploaded_file = st.file_uploader("Choose a file", type =(["mp3", "wav"]))
    if uploaded_file is not None:
        #bytes_data = 오디오 파일
        global bytes_data
        bytes_data = uploaded_file.name
        st.success('파일을 업로드 했습니다. : {} '.format(bytes_data))


#음성 파일 불러와서 텍스트로 전환
def STT():
    os.chdir(path)
    name, ext = os.path.splitext(filepath)
    if ext == ".wav":
        harvard_audio = sr.AudioFile(filepath)
        with harvard_audio as source:
            audio = r.record(source)
    elif ext == '.mp3':
        mp3_sound = AudioSegment.from_mp3(filepath)
        wav_sound = mp3_sound.export("{0}.wav".format(name), format="wav")
        harvard_audio = sr.AudioFile(wav_sound)
        with harvard_audio as source:
            audio = r.record(source, duration=150)       
    else:
        st.write("wav 와 mp3 형식만 호환됩니다.")


    try:
        sys.stdout = open('%s.txt' % name, 'w', encoding='UTF-8') 
        text = r.recognize_google(audio, language='ko-KR')
        print(text)
        sys.stdout.close()
    except Exception as e:
        print(e)
       
#---------------------------------------------------------------------------

st.header('Text To Speech')

upload()

filepath= bytes_data
path = './audio/'

STT()
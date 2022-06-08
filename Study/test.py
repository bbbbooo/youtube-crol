import streamlit as st
import plotly.express as px
from api import crurl

st.title("유튜브 크롤링")

input_user_name = st.text_input(label="Youtube-URL", value="URL을 입력해주세요")


if st.button("검색"):
    con = st.container()
    con.caption("결과")
    con.write(f"User Name is {str(input_user_name)}")
    
    
def create_plot(data):
    data = data['sentiment category'].value_counts()
    labels = list(data.index)
    values = list(data)
    fig = px.pie(labels, values=values, 
    names= labels, color = labels, 
    color_discrete_map = {
        'Positive': '#1f77b4',
        'Negative': '#d62728',
        'Neutral': '#7f7f7f'
    })
    return fig
    

def make_dataframe(data, options = []):
    if len(options) == 0 or options[0] == 'ALL':
        st.header('Youtube comment data')
        st.dataframe(data[['comments','sentiment score']].style.bar(subset=['sentiment score'], color=['#d65f5f','#5fba7d']))
        return data
    else:
        data = data[data['sentiment category'].isin(options)]
        st.header('Youtube comment data')
        st.dataframe(data[['comments','sentiment score']].style.bar(subset=['sentiment score'], color=['#d65f5f','#5fba7d']))
        return data
    
def get_data(video_url):
    data = crurl(video_url)
    data = sentiment(data)
    return data

    
# sidebar ----------------------------
sidebar = st.sidebar

# header 
sidebar.header('Input Options')
video_url=sidebar.text_input('youtube url')
# multiselect
options = sidebar.multiselect('What sentiment category do you want to see?',
    ['ALL','Positive', 'Negative', 'Neutral'])


col1, col2 = st.columns(2)



if sidebar.button('search'):

    
    # dataframe
    data = get_data(video_url)
    data = make_dataframe(data,options=options)
    

    # pie plot
    fig = create_plot(data)
    st.header('Pie plot')
    st.plotly_chart(fig, use_container_width = True)

    # multiselect
    
    
    
# def get_thumbnail(url):
#     id = url_to_id(url)
#     img = 'https://img.youtube.com/vi/{}/0.jpg'.format(id)
#     return img 
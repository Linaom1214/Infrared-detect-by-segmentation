import streamlit as st
from PIL import Image
import os 
from main import App

if __name__ == '__main__':
    # st.image(['./images/test_image_0.png', './images/train_image_0.png'])
    st.write('左侧上传文件')

    option = 'unet'
    option = st.selectbox('选择一个模型', ['unet', 'fcn'])
    application = App(option)
    st.write('Load Model', option)

    source = None
    uploaded_file = st.sidebar.file_uploader("上传图片", type=['png', 'jpeg', 'jpg', 'bmp'])

    if uploaded_file is not None:
        is_valid = True

        with st.spinner(text='资源加载中...'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = picture.save(f'images/{uploaded_file.name}')
            source = f'images/{uploaded_file.name}'
    else:
        is_valid = False


    if is_valid:
        image = application.single(csource)
        with st.spinner(text='Soure Images'):
            st.image(source)
        with st.spinner(text='Preparing Images'):
            st.image('images/pred.png')


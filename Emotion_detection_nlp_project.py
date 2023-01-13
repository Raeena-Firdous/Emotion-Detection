import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

import joblib

pipe_lr=joblib.load(open("emotion_classifier_pipe_lr.pkl","rb"))

def predict_emotions(docx):
    results=pipe_lr.predict([docx])
    return results

def get_prediction_proba(docx):
    results=pipe_lr.predict_proba([docx])
    return results

#emotions_emoji_dict={"Anger":"😡","disgust": "🤮","fear":"😱","joy" : "😂","neutral":"😑","sadness":"🤗","shame":"","surprise":""}


def main():
    menu=["Home","Monitor","About"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key="emotion_clf_form"):
            raw_text=st.text_area("Type Here")
            submit_text=st.form_submit_button(label="submit")

        if submit_text:
            col1,col2 = st.columns(2)

            prediction= predict_emotions(raw_text)
            probability=get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("prediction")
                if prediction=="joy":
                    global output
                    l=["happy","grateful","cheerful","glad","pleasure"]
                    output=np.random.choice(l)
                    st.write(output)
                elif prediction=="anger":
                    t=["offended","annoyed","irritated","displeased"]
                    output1=np.random.choice(t)
                    st.write(output1)
                elif prediction=="fear":
                    p=["horror","panic","scared","doubt","jitter"]
                    output2=np.random.choice(p)
                    st.write(output2)
                elif prediction=="disgust":
                    q=["dislike","hatred","repulsion","objection","revolt"]
                    output3=np.random.choice(q)
                    st.write(output3)
                elif prediction=="neutral":
                    r=["disinterested","inactive","undecided"]
                    output4=np.random.choice(r)
                    st.write(output4)
                elif prediction=="sadness":
                    s=["sorrow","grief","dejection","distress","anguish"]
                    output5=np.random.choice(s)
                    st.write(output5)
                elif prediction=="shame":
                    a=["humiliation","dishonor","irritation","discredict","abashment"]
                    output6=np.random.choice(a)
                    st.write(output6)
                elif prediction=="surprise":
                    b=["shock","amazement","miracle","wonderment"]
                    output7=np.random.choice(b)
                    st.write(output7)

                #emoji_icon=emotions_emoji_dict[prediction]
                #st.write(output1)
                st.write("Max Probability Class:{}".format(np.max(probability)))
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                st.write(proba_df.T)

            with col2:
                st.success("Prediction probability")
                st.write(probability)
                proba_df_clean=proba_df.T.reset_index()
                proba_df_clean.columns=["emotions","probability"]

                fig=alt.Chart(proba_df_clean).mark_bar().encode(x="emotions",y="probability")
                st.altair_chart(fig,use_container_width=True)


    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.header("About")


if __name__=="__main__":
    main()
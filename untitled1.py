import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('titanic_model.pkl','rb'))


def predict_forest(Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked):
    input=np.array([[Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked]])
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("")
    html_temp = """
    <style>
         body {
            background-image: url("https://images.unsplash.com/photo-1554034483-04fda0d3507b?ixlib=rb-1.2.1&w=1000&q=80");background-size :cover;
         }
    </style>
    <div><img src="https://miro.medium.com/max/875/1*VqAjw0QcBo13t9ISpRjGMw.png" id="bg" alt="" width="700" height="300" > </img>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    from PIL import Image
    image = Image.open('24786.png')
    st.image(image,use_column_width = True)
    Pclass = st.radio("Passenger Class (1 - 1st; 2=2nd; 3=3rd)",("1","2","3"))
    Sex = st.radio("Sex (0-Male;1-Female)",("0","1"))
    Age = st.slider("Age Of Passenger",0,80)
    SibSp = st.radio("SibSp (Number of Siblings/Spouses Aboard)",("0", "1", "2", "3", "4", "5", "8"))
    Parch = st.radio("Parch (Number of Parents/Children Aboard)",("0","1","2","3","4","5","6"))
    Fare = st.slider("Passenger Fare",0,520)
    Cabin = st.slider("Cabin",0,146)
    Embarked = st.radio("Port of Embarkation (0 = Southampton; 1 = Cherbourg; 2 = Queenstown)",("0","1","2"))
    safe_html=""" 
      <img src="data:image/png;base64,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" id="bg" alt="" width="600" height="200" > </img>
    """
    danger_html="""  
      <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRunEqyvn9xPiWfkBgH3BQcFVu9EYx5lbg--A&usqp=CAU" id="bg" alt="" width="600" height="200" > </img>
    """

    if st.button("Predict"):
        output=predict_forest(Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked)
        st.info('The probability of Dying is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
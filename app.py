import streamlit as st
import pickle
import numpy as np


def deep_learning_char():
    import SessionState
    session = SessionState.get(run_id=0)
    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
    from streamlit_drawable_canvas import st_canvas
    st.title("Handwritten Character Recognition using Neural Nerwork")
    if st.button("Reset"):
        session.run_id += 1
    st.markdown("Please Click on Reset Button after predicting each image")
    image_data = st_canvas(brush_width=30, brush_color='white', background_color='#000703', key=session.run_id)
    if image_data is not None:
        print(image_data)
        print(image_data.shape)
        image_data = image_data.astype('float32')
        image_data = image_data / 255
        img = plt.imsave('hum.png', image_data)
        if st.button("Predict Saved Image", key=session.run_id):
            pro = []
            img_input = cv2.imread('hum.png', cv2.IMREAD_GRAYSCALE)
            pro.append(cv2.resize(img_input, (28, 28)))
            pro_arr = np.array(pro)
            pro_arr = pro_arr.astype('float32')
            pro_arr = pro_arr / 255
            pro_arr = pro_arr.reshape(1, 28, 28, 1)
            alphabets_mapper_aplha = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                                  9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
                                  18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
            alphabets_mapper_aplha = dict(alphabets_mapper_aplha)
            loaded_model = load_model("char99.h5")
            output = loaded_model.predict_classes(pro_arr)
            l = list(output)
            ln = l[0]
            print(ln)
            alp = alphabets_mapper_aplha.get(ln)
            print(alp)
            print(str(output))
            st.info(alp)

def deep_learning_digit():
    import SessionState
    session = SessionState.get(run_id=0)
    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model
    import matplotlib.pyplot as plt
    from streamlit_drawable_canvas import st_canvas
    st.title("Handwritten Digit Recognition using Neural Nerwork")
    if st.button("Reset"):
        session.run_id += 1
    st.markdown("Please Click on Reset Button after predicting each image")
    image_data = st_canvas(brush_width=30, brush_color='white', background_color='#000703', key=session.run_id)
    if image_data is not None:
        print(image_data)
        print(image_data.shape)
        image_data = image_data.astype('float32')
        image_data = image_data / 255
        img = plt.imsave('hum1.png', image_data)
        if st.button("Predict Saved Image", key=session.run_id):
            pro = []
            img_input = cv2.imread('hum1.png', cv2.IMREAD_GRAYSCALE)
            pro.append(cv2.resize(img_input, (28, 28)))
            pro_arr = np.array(pro)
            pro_arr = pro_arr.astype('float32')
            pro_arr = pro_arr / 255
            pro_arr = pro_arr.reshape(1, 28, 28, 1)
            loaded_model = load_model("digit99.h5")
            output = loaded_model.predict_classes(pro_arr)
            l = list(output)
            st.info(l[0])

def titanic():
    html_temp = """
        <div><img src="https://miro.medium.com/max/875/1*VqAjw0QcBo13t9ISpRjGMw.png" id="bg" alt="" width="700" height="300" > </img>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    model = pickle.load(open('titanic_model.pkl', 'rb'))
    def predict_forest(Pclass,Sex,Age,SibSp,Parch,Fare,Cabin,Embarked):
        input = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked]])
        prediction = model.predict_proba(input)
        pred = '{0:.{1}f}'.format(prediction[0][0], 2)
        return float(pred)
    #from PIL import Image
    Pclass = st.radio("Passenger Class (1 - 1st; 2=2nd; 3=3rd)", ("1", "2", "3"))
    Sex = st.radio("Sex (0-Male;1-Female)", ("0", "1"))
    Age = st.slider("Age Of Passenger", 0, 80)
    SibSp = st.radio("SibSp (Number of Siblings/Spouses Aboard)", ("0", "1", "2", "3", "4", "5", "8"))
    Parch = st.radio("Parch (Number of Parents/Children Aboard)", ("0", "1", "2", "3", "4", "5", "6"))
    Fare = st.slider("Passenger Fare", 0, 520)
    Cabin = st.slider("Cabin", 0, 146)
    Embarked = st.radio("Port of Embarkation (0 = Southampton; 1 = Cherbourg; 2 = Queenstown)", ("0", "1", "2"))
    safe_html = """ 
                <img src="data:image/png;base64,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" id="bg" alt="" width="600" height="200" > </img>
                """
    danger_html = """  
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRunEqyvn9xPiWfkBgH3BQcFVu9EYx5lbg--A&usqp=CAU" id="bg" alt="" width="600" height="200" > </img>
                """

    if st.button("Predict"):
        output = predict_forest(Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked)

        st.success('The probability of Dying is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html, unsafe_allow_html=True)
        else:
            st.markdown(safe_html, unsafe_allow_html=True)

def Cab_booking():
    st.title("Predicting The Cab Bookings using Machine Learning")
    model = pickle.load(open('cab_booking.pkl', 'rb'))
    def predict_booking(holiday, workingday, temp, atemp, humidity, windspeed,Month, Hour, Weekday_No, Year):
        input = np.array([[holiday,workingday,temp,atemp,humidity,windspeed,Month,Hour,Weekday_No,Year]])
        pred = model.predict(input)
        return float(pred)
    holiday = int(st.radio("Holiday or Not (0 - Not a Holiday; 1 = Holiday)", ("0", "1")))
    #workingday = int(st.radio("Working Day or Not ((0 - Not a workingday; 1 = workingday)",("0","1")))
    if holiday == 0:
        workingday=1
    if holiday ==1:
        workingday=0
    temp = int(st.slider("Temparature (you Feel)", 0,41))
    atemp = int(st.slider("Accurate Temparature", 0, 50))
    humidity = int(st.slider("Humidity",0,100))
    windspeed = int(st.slider("WindSpeed",0,60))
    Month = int(st.radio("Month",("1","2","3","4","5","6","7","8","9","10","11","12")))
    Hour = int(st.slider("Hour (24 Hrs ) ", 0, 24))
    Weekday_No = int(st.radio("Weekday No: (0 = Monday; 1 = Tuesday ... 6: Sunday)", ("0", "1", "2","3","4","5","6")))
    Year = int(st.slider("Year",2012,2020))
    print(holiday, workingday, temp, atemp, humidity, windspeed,Month, Hour, Weekday_No, Year)
    if st.button("Predict"):
        output = predict_booking(holiday, workingday, temp, atemp, humidity, windspeed,Month, Hour, Weekday_No, Year)
        print(output)
        st.success('The predicted Bookings for the day {}'.format(output))


def main():
    st.title("")
    html_temp = """
    <style>
        body {
            background-image: url("https://images.unsplash.com/photo-1554034483-04fda0d3507b?ixlib=rb-1.2.1&w=1000&q=80");background-size :cover;
        }
    </style>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    model_type = st.sidebar.radio("Select Model :", ("Classification", "Regression","DL - Character","DL - Digit"))
    if model_type == "Classification":
        titanic()
    print(model_type)
    print(type(model_type))
    if model_type == "Regression":
        Cab_booking()
    if model_type == "DL - Character":
        deep_learning_char()
    if model_type == "DL - Digit":
        deep_learning_digit()

if __name__=='__main__':
    main()
def digit_pred(k):
    import numpy as np
    import cv2
    import pickle
    pickle_in = open("model_trained.p", "rb")
    model = pickle.load(pickle_in)
    img = cv2.imread(str(k),cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(28,28))
    img = np.array(img)
    img = img.astype('float32')
    img = img.reshape(1,28,28,1)
    y_pred = model.predict_classes(img)
    model = None
    print(model)
    return y_pred

import numpy as np
import keras
from keras.preprocessing import image


model = keras.models.load_model('covid_detector_model.h5')

# classify the input image
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    if prediction[0] < 0.5:
        return 'Negative COVID-19 case'
    else:
        return 'Positive COVID-19 case'


image_path = input("Enter the path of the image: ")
classification_result = classify_image(image_path)
print("Classification Result: ", classification_result)

# Covid-19 detection using chest X-Rays
This deep learning project detects COVID-19 using Chest X-ray images. A Convolutional Neural Network (CNN) based model in keras tensorflow is used to identify positive cases of the virus. 


### Technology Stack:
- Python
- Keras 
- TensorFlow
- NumPy

## Dataset used
Datasets from a GitHub and Kaggle source were used (linked in the sources).
144 Positive X-Rays and 144 Negative X-Rays were used. These sets were split into two sets training and validation purposes.  

<img src = "https://github.com/anvita-g/Covid-19-Detector/assets/75912590/388ef151-411e-4800-a389-811c29f5505e" width ="200" /> 
<img src = "https://github.com/anvita-g/Covid-19-Detector/assets/75912590/50f3a5e4-982b-419e-912f-01d87929e3cb" width ="200" /> 
<img src = "https://github.com/anvita-g/Covid-19-Detector/assets/75912590/8675d66a-c359-4508-8967-8e81f1e5f2dc" width ="200" /> 
<img src = "https://github.com/anvita-g/Covid-19-Detector/assets/75912590/b1b2962c-b46e-48c5-9c5e-1b7f5a1f2f84" width ="200" /> 

### Model Architecture
- Built using a CNN in Keras, consisting of convolutional layers, max pooling layers, dropout layers, and fully connected layers.
- The output is passed through two connected layers with the sigmoid activation function to predict the binary classification of the image.

## Results
It achieved a training accuracy of 92% and a validation accuracy of 98%. This means that the model was able to classify COVID-19 cases from the X-ray images with a high degree of accuracy.
![image](https://github.com/anvita-g/Covid-19-Detector/assets/75912590/131eb577-cd7e-461d-8929-14082f2150d7)


## Sources
- Negative cases data set : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Positive cases data set: https://github.com/ieee8023/covid-chestxray-dataset
- Coding Blocks tutorial on youtube (Detecting COVID-19 from X-Ray| Training a Convolutional Neural Network | Deep Learning) was used as a guide

'''

Features of the CNN:
    - Convolutional Layers: The model starts with a series of convolutional layers to extract features from the input X-ray images.
    - Max Pooling Layers: Max pooling layers are used to reduce the spatial dimensions of the feature maps, aiding in capturing essential information.
    - Dropout Layers: Dropout layers are use to prevent overfitting by randomly disabling a fraction of the neurons during training.
    - Flatten Layer: This layer flattens the output from the previous layers into a 1-D vector.
    - Fully Connected Layers: The flattened features are passed through fully connected layers to learn complex patterns and make predictions.
    - Output Layer: The final layer uses the sigmoid activation function to produce a binary classification output, indicating COVID-19 positive or negative.


Sources
    - Negative cases data set : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    - Positive cases data set: https://github.com/ieee8023/covid-chestxray-dataset
    - Coding Blocks tutorial on youtube (Detecting COVID-19 from X-Ray| Training a Convolutional Neural Network | Deep Learning) was used as a guide



'''
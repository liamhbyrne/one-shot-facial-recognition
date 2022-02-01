# one-shot-facial-recognition
This project aimed to provide a facial recognition system to the Raspberry Pi which only required 1 photo of the subject. 
I had previously experimented with Convolutional Neural Networks for facial recognition and found out that in order to make meaningful predicitons: 
the training set needs a multitude of photos. I came across one shot learning, which only required 1 image of a person to recognise them in the future. 

A common tool to achieve this is Siamese neural networks, which involves using a pretrained neural network to convert an image into an embedding,
we can then find the distance between the embedding of the new face and known faces. 

## Implementation
I used the Haar-cascade classifier in OpenCV to locate and crop faces in the Rasperry Pi camera video feed. 
I trained a sequential Tensorflow model on the [Olivetti Dataset of faces](https://www.kaggle.com/serkanpeldek/face-recognition-on-olivetti-dataset).
I converted this model to TensorFlow Lite so it could be run on the Raspberry Pi (and Android) efficiently.

## Conclusion
I successfully deployed the model onto the Raspberry Pi. The Raspberry Pi 3 could run the system for extended periods of time without slowing. Through testing I found that it
could consistently recognise different people shortly after they register their face. Although, small changes to lighting would heavily decrease its confidence in a prediction.
I tried to mitigate this through registering multiple faces of the same person in different lighting, a more robust solution would be to use image augmentation.

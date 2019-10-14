# TF-Siamese-Face-Recognition
<B>The code is implemented using Tensorflow(version 1.14) </B>
This is a simple Face recognition model that will predict wether the two images belong to same person or not.
This model is trained to recognize face of 10 different people including my own but it can be extended to more people also.
Loss Function used in this model is <B>Binary Cross entropy</B> but in code there's <B>Contrastive loss</B> also available to use.
Dataset used for training is [AT&T face dataset](https://github.com/maheshreddykukunooru/Face_recognition/tree/master/att_faces)
The Model is trained for 100 epochs and after testing it on unseen test images the ROC score is <B>0.83</B>



<B>Contrastive Loss:</B><n>
![alt text](https://hackernoon.com/hn-images/1*tzGB6D97tHWR_-NJ8FKknw.jpeg)
  
<B>Siamese Network Architecture</B><n>
![alt text](https://miro.medium.com/max/2524/1*8Nsq1BYQCuj9giAwltDubQ.png)


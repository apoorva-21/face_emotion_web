# face_emotion_web

Web Application for Facial Emotion Recognition

This is a simple web application that reads in a snapshot of the face of the client through their webcam and sends it to be processed using a dense neural network at the server. The output of the network is the predicted emotion that the person must be feeling when the picture was taken, based on the facial expression.

Makes use of HTML, CSS, JavaScript, jQuery, Flask, dlib and Keras.

Emotion recognition is one of the most critical tasks that must be achieved robustly in order to enhance man-machine interactions to the next level. By interpreting the emotion that a person is feeling, a machine/ software can achieve a number of tasks that are expected from sentient beings. Be it recording the behavior of the person over a time interval, keeping track of the pulse of the person or even reading in an image of the person's face to predict the emotion he is feeling, this knowledge can help in the following ways :
-Psychoanalysis of the person
-Automation of effective media playback (audio, video and imagery suitable to the person's mood)
-Medical Research in psychology
-Medical Research pertaining to the effect of drugs, infections and disorders on the behavior of a person
-Product recommendations can also be made effective

Dataset Used: JAFFE, a concise dataset for facial expressions corresponding to emotions : http://www.kasrl.org/jaffe.html

Tip : Run the fwdPropOnRcvdImg.py script if using Theano backend, since a TF backend gives errors.

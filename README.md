# Deep-Learning-Models-without-Frameworks

Implementation of deep learning models without AI frameworks (tensorflow, pytorch, etc.)<br>
Refer to ADT of pytorch models

## DNN
### Task
Handwritten Digit Recognition
### Data
MNIST
### Result (Confusion Matrix)
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/DNN/figure/lrelu_cf.png">
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/DNN/figure/relu_cf.png">

## CNN
### Task
Handwritten Digit Recognition
### Data
MNIST
### Result (Confusion Matrix)
CNN2: Convolution layer &rarr; Convolution layer &rarr; Flatten layer &rarr; Linear layer <br>
CNN3: Convolution layer &rarr; Convolution layer &rarr; Convolution layer &rarr; Relu &rarr; Maxpool &rarr; Flatten layer &rarr; Linear layer
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/CNN/CNN_results.png">

## RNN / LSTM
### Task
Sentence Emotion Analysis
### Emotion Type
0: ❤️ (Heart) <br>
1: ⚾ (Baseball)<br>
2: 😊 (Smile)<br>
3: 😞 (Disappointed)<br>
4: 🍴 (Fork and Knife)
### Result
#### RNN
<span>
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/A%20accuracy%20graph.png" width=45%>
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/A%20loss%20graph.png" width=45%>
</span>
  
#### LSTM
<span>
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/E%20accuracy%20graph.png" width=45%>
<img src="https://raw.githubusercontent.com/Seulin/Deep-Learning-Models-without-Frameworks/main/RNN-LSTM/E%20loss%20graph.png" width=45%>
</span>

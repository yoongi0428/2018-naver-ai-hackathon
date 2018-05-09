# Hala Madrid

### Author
* Yoon Ki Jeong (yoongi0428)

## Contents

### Problem

#### [NAVER KIN - Sentence Similarity](https://github.com/naver/ai-hackathon-2018/blob/master/missions/kin.md)
* Given two sentences, predict whether the two is similar or not. 

### Model

* ["Siamese Recurrent Architectures for Learning Sentence Simialrity" J. Mueller, A. Thyagarajan, 2016](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023)
* Modify this architecture using Character Level CNN 
* Embedding - Character Level CNN - Fully Connected - Manhattan Distance - Prediction

### Detail

* Character Level
* Wide CNN
* No dropout or regularization
* 2 FC layers
* Prediction = exp( -sum( Manhattan Distance of the two outputs ) )
* Loss : MSE

### Result

* Accuracy : 0.8023 (6th)


## HALA MADRID
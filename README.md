# Acoustic-Scene-Classification
June 2017
Contributors: Anas Barakat, Bilal Mouhib

Kaggle-like data challenge where the task was to do "acoustic scene classification". In particular, given an audio signal of ~30 seconds, the goal was to find its "context" or the "environment" corresponding to the signal such as a beach, a restaurant, or a metro station. In total we considered 15 different classes and 1170 audio files separated in training, validation and testing sets.

Final solution proposed: 

- Extraction of MFCC and delta MFCC features from the audio files. 
- Combination of early and late fusion to build the design matrix. 
- Scaling matrixes
- Majority vote combining MLP, QDA and Random Forest predictions for the final prediction. 

Related articles used: 
- Acoustic Scene Recognition with Deep Neural Networks (DCASE chal- lenge 2016), Dai Wei, Juncheng Li, Phuong Pham, Samarjit Das, Shuhui Qu
- Acoustic scene classification : an evaluation of an extremely compact feature representation, Gustavo Sena, MafraNgoc Q. K. Duong, Alexey Ozerov, Patrick Pérez, Budapest 2016

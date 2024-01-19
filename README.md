# User Guide

- There are 3 models in total in our project; 2 baselines, 1 proposed model. Our baselines are CNN.py and FFN.py and our proposed model is hybrid_model.py. There is also another set of models with the same architectures but different dataloader, which stacks all 4 features into a single matrix as model input. The model weights are already saved in this directory therefore the training loops in the model codes are all commented.
- If you want to learn more about our model you can view our [paper](https://github.com/RabiaGondur/deepfake_audio_classification/blob/main/Final_Paper.pdf) and [presentation](https://github.com/RabiaGondur/deepfake_audio_classification/blob/main/DL_Presentation.pdf)

# To run
- Make sure you have all the necessary libraries installed and all the folders should be in the same directory. Once that's set you can just run all three .py files normally depending on which model you want to check.

- Note: We only included the weights for our best performing models with the 2 popular audio features due to space constraints. Also our dataset is very large >70GB so we included a very small portion here for demonstration, the results will slightly differ from the paper as we used the portion of the original dataset for training and testing.

# Dependencies
```
!pip install torch
!pip install torchvision
!pip install scikit-learn
!pip install scipy
!pip install seaborn
```

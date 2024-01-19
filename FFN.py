import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# -----------------------------------------------------------------------------
# hyperparams
# -----------------------------------------------------------------------------

np.random.seed(0)
torch.manual_seed(0)
batch_size = 64
num_epochs = 50


print('Starting....')
# -----------------------------------------------------------------------------

# data_folder = './ProcessedData'
# file_list = os.listdir(data_folder)

# npy_files = [file for file in file_list if file.endswith('.npy')]
# print(len(npy_files))

# ten_percent_data =  22600 + 22600 # (len(npy_files) * 10) // 100
# print(ten_percent_data)
# # quit()
# # -----------------------------------------------------------------------------
# # getting balanced data by enforcing a max value for a given label
# # -----------------------------------------------------------------------------

# zero_list = []
# one_list = []
# zero_count = 0
# one_count = 0
# total = zero_count + one_count
# for file in npy_files:
#     if (np.load(os.path.join(data_folder, file), allow_pickle=True)[2] == 0):
#         if zero_count < (ten_percent_data//2):
#         # print('yay')
#             #print(np.load(os.path.join(data_folder, file), allow_pickle=True)[2] == 0)
#             zero_list.append(file)
#             zero_count +=1
#             #print(zero_count)
#         else:
#             continue
#     elif (np.load(os.path.join(data_folder, file), allow_pickle=True)[2] == 1):
#         if one_count < (ten_percent_data//2):
#             #print(np.load(os.path.join(data_folder, file), allow_pickle=True)[2] == 1)
#             one_list.append(file)
#             one_count += 1
#             #print(one_count)
#         else:
#             continue
#     total = zero_count + one_count
#     if total >= (ten_percent_data):
#         break


# print(zero_list)
# print(f' length list 1: {len(one_list)} \n {one_list}')
# print(f'length list 0: {len(zero_list)} \n {zero_list}')
# print(f' length list 1: {len(one_list)} ') # 22617
# print(f'length list 0: {len(zero_list)} ') # 30591
 
# one_list_save = np.save('./one_list.npy', one_list)
# zero_list_save = np.save('./zero_list.npy', zero_list)
# one_list = np.load('./one_list.npy', allow_pickle=True)
# zero_list = np.load('./zero_list.npy', allow_pickle=True)
# print(f' length list 1: {len(one_list)}')
# print(f'length list 0: {len(zero_list)}')
# print('Done!')

# -----------------------------------------------------------------------------
# shuffling + splitting for test and train
# -----------------------------------------------------------------------------
# label_0_files = zero_list
# label_1_files = (one_list)


# print(type(label_0_files), type(label_1_files))
# len_zero = (len(zero_list)) //2
# len_one = (len(one_list)) //2

# # print(f'Half {len(len_zero)}')
# # print(label_0_files[:100].shape+ label_1_files[:100].shape)

# np.random.shuffle(label_0_files)
# np.random.shuffle(label_1_files)

# half = 11300
# train_files = list(label_0_files)[:half] + list(label_1_files)[:half]
# test_files = list(label_0_files)[half:] + list(label_1_files)[half:]


# -----------------------------------------------------------------------------
# creating a customdataset
# -----------------------------------------------------------------------------
# class CustomDataset(Dataset):
#     def __init__(self, file_list, data_folder):
#         self.file_list = file_list
#         self.data_folder = data_folder

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
     
#         file_path = os.path.join(self.data_folder, self.file_list[idx])
#         data = np.load(file_path, allow_pickle=True)

#         mfcc_lfcc_array = data[0]
#         cqcc_lpcc_array = data[1]
#         label = data[2]
#         features1 = mfcc_lfcc_array[:43, :,:]

#         concatenated_features = np.concatenate((mfcc_lfcc_array[0], mfcc_lfcc_array[1]), axis=-1)
#         concatenated_features = np.expand_dims(concatenated_features, axis=0)
        

#         return features1, label


# train_dataset = CustomDataset(train_files, data_folder)
# test_dataset = CustomDataset(test_files, data_folder)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
selected_data = np.load('./selected_data.npy', allow_pickle=True)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc_lfcc_array = self.data[idx][0]
        cqcc_lpcc_array = self.data[idx][1]
        label = self.data[idx][2]

        features1 = mfcc_lfcc_array[:43, :, :]
        features2 = cqcc_lpcc_array

        return features1, label

custom_dataset = CustomDataset(selected_data)

train_size = int(0.2 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# getting the shapes for inputs and labels
# -----------------------------------------------------------------------------
print('-----------Going through dataloaders to see the shapes')
# for inputs, labels in train_loader:
#     print(inputs.shape)
#     print(labels.shape)
#     break

# for inputs, labels in test_loader:
#     print(inputs.shape)
#     print(labels.shape)
#     break


# labels_count = {0: 0, 1: 0}

# for features, label in train_dataset:
#     labels_count[label] += 1

# print("Label 0 count:", labels_count[0])
# print("Label 1 count:", labels_count[1])

input_size =  1118 #1716 for feature 2 #1118 for feature 1
hidden_size = 16
hidden_size1= 10
hidden_size2 = 12


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)  # 2 classes: fake or real

    def forward(self, x):
        # print('------------')
        x = self.flatten(x)
  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))
        return x

model = AudioClassifier()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.001)


# # -----------------------------------------------------------------------------
# # training loop
# # -----------------------------------------------------------------------------
# print('-----------Training')
# loss_values = []
# for epoch in range(num_epochs):
#     model.train()
#     for inputs, labels in train_loader:
   
#         inputs = torch.tensor(inputs, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.long)
#         # inputs = input.double() 

       
#         # print(inputs.shape)
#         outputs = model(inputs)
#         outputs = torch.squeeze(outputs, 1)
#         # print(outputs.shape, labels.shape)

  
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     loss_values.append(loss.item())
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# torch.save(model.state_dict(), './FFN.pth')
# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------
PATH = './FFN.pth'
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)
print('-----------Starting Evaluation')

def calculate_eer( classifier_model, test_loader):
   
    classifier_model.eval()

    correct = 0
    predicted_labels = []
    total = 0
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            outputs = classifier_model(inputs)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")

    f1 = classification_report(true_labels, predicted_labels, digits=4)
    print(f"F1 Score:\n{f1}")


    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return eer * 100.0  


eer = calculate_eer(model, test_loader)
print(f'Equal Error Rate (EER): {eer:.2f}%')

# from torchsummary import summary
# summary(model)


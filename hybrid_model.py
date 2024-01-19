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



np.random.seed(0)
torch.manual_seed(0)

batch_size = 64
num_epochs = 50
in_size = 43

print('Starting....')
# -----------------------------------------------------------------------------

# data_folder = './ProcessedData'
# file_list = os.listdir(data_folder)

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

#         # mean = np.mean(mfcc_lfcc_array)
#         # std = np.std(mfcc_lfcc_array)
#         # normalized_mfcc_lfcc_array = (mfcc_lfcc_array - mean) / std

#         # print(mfcc_lfcc_array.shape, cqcc_lpcc_array.shape)
#         features1 = mfcc_lfcc_array[:43, :,:]
#         features2 = cqcc_lpcc_array
#         # print((features1.shape[0]))

#         # concatenated_features = np.concatenate((mfcc_lfcc_array[0], mfcc_lfcc_array[1]), axis=-1)
#         # concatenated_features = np.expand_dims(concatenated_features, axis=0)
        

#         return features1, label

# train_dataset = CustomDataset(train_files, data_folder)
# test_dataset = CustomDataset(test_files, data_folder)
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


# # -----------------------------------------------------------------------------
# # AE
# # -----------------------------------------------------------------------------


input_size =  2 # 4
hidden_size = 24
hidden_size1= 64
hidden_size2 = 128
latent_size = 8

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size)
        self.bottle_neck = nn.Linear(hidden_size, latent_size)
    

        self.fc_decode = nn.Linear(latent_size, hidden_size)
        self.fc_output1 = nn.Linear(hidden_size, hidden_size1)
        self.fc_output2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc_output3 = nn.Linear(hidden_size2, input_size)

    def encode(self, x):
        # print(x.shape)
       
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        # print(x.shape)
        z = self.bottle_neck(x)
        return z
   

    def decode(self, z):
        z = F.elu(self.fc_decode(z))
        z = F.elu(self.fc_output1(z))
        z = F.elu(self.fc_output2(z))
        x = torch.sigmoid(self.fc_output3(z))
        return x

    def forward(self, x):
        z = self.encode(x)
        
        x_recon = self.decode(z)
        return x_recon, z

AE_model = AE().double() 
# criterion = nn.MSELoss().double() 
# optimizer = optim.Adam(AE_model.parameters(), lr=0.0001)

# print('-------------AE training starting....')
# loss_values_ae = []
# for epoch in range(num_epochs):
#     for data, _ in train_loader:
#         # print(data.shape)
    
#         inputs = data.double() 
#         # print(inputs.shape)

      
#         optimizer.zero_grad()

       
#         recon, z = AE_model(inputs)
#         # print(mu.shape) # 64, 128
#         # print(z.shape)

        
#         reconstruction_loss = criterion(recon, inputs)
        
#         loss = reconstruction_loss

       
#         loss.backward()
#         optimizer.step()
#         loss_values_ae.append(loss.item())
#     if epoch % 5 ==0:
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# torch.save(AE_model.state_dict(), './AE.pth')

# # -----------------------------------------------------------------------------
# # CNN
# # -----------------------------------------------------------------------------

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_size, out_channels=16, kernel_size=3,  padding=2)
        self.relu1 = nn.ReLU()

        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, stride=1, padding=2)
        self.relu2 = nn.ReLU()
    
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2040, 12)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(12, 2)  

    def forward(self, x):
        # print('------------')
  
        x = self.conv1(x)

        x = self.relu1(x)
        # print(x.shape)
        # x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu2(x)
      
        # print(x.shape)
        # x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(f' after flattening {x.shape}')
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu3(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

model = AudioClassifier()
AE_model = AE()
PATH = './AE.pth'
state_dict = torch.load(PATH)
AE_model.load_state_dict(state_dict)
AE_model.float()
AE_model.eval()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.001)
# num_epochs = 50
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
#         z = AE_model.encode(inputs)
#         # z = torch.unsqueeze(z, 0)


       
#         # print(f' Shape of latent is {z.shape}')
#         outputs = model(z)
       

  
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     loss_values.append(loss.item())
#     if epoch % 5 == 0:
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# torch.save(model.state_dict(), './CNN_AE.pth')
PATH = './CNN_AE.pth'
state_dict = torch.load(PATH)
model.load_state_dict(state_dict)
# # -----------------------------------------------------------------------------
# # evaluation
# # -----------------------------------------------------------------------------

def calculate_eer(AE_model, classifier_model, test_loader):
    AE_model.eval()
    classifier_model.eval()

    correct = 0
    predicted_labels = []
    total = 0
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            z = AE_model.encode(inputs)
            # print(f'THE SIZE OF Z: {z.shape}')

            outputs = classifier_model(z)

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


eer = calculate_eer(AE_model, model, test_loader)
print(f'Equal Error Rate (EER): {eer:.2f}%')


# from torchsummary import summary
# summary(model)

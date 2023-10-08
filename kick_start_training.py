import copy
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_curve
import time
import matplotlib.pyplot as plt

scenario = 'desktop'

def contrastive_loss(x1, x2, label, margin: float = 1.0):
    dist = nn.functional.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss


class Kick_Start_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=100)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.batch_norm(x)
        x, _ = self.rnn(x)
        x = self.linear(x)
        embedding = self.softmax(x)[:, -1, :]
        return embedding


class PrepareData:
    def __init__(self, dataset, sequence_length, samples_considered_per_epoch):
        self.data = dataset
        self.len = samples_considered_per_epoch
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        user_idx = random.choice(list(self.data.keys()))
        session_idx = random.choice(list(self.data[user_idx].keys()))
        session_1 = self.data[user_idx][session_idx]
        session_1 = np.concatenate((session_1, np.zeros((self.sequence_length, np.shape(session_1)[1]))))[:self.sequence_length]
        diff_1 = np.reshape((session_1[:, 1] - session_1[:, 0]) / 1E3, (np.shape(session_1)[0], 1))
        ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1))
        session_1_processed = np.concatenate((diff_1, ascii_1), axis=1)

        label = random.choice([0, 1])
        if label == 0:
            session_idx_2 = random.choice([x for x in list(self.data[user_idx].keys()) if x != session_idx])
            session_2 = self.data[user_idx][session_idx_2]
        else:
            user_idx_2 = random.choice([x for x in list(self.data.keys()) if x != user_idx])
            session_idx_2 = random.choice(list(self.data[user_idx_2].keys()))
            session_2 = self.data[user_idx_2][session_idx_2]
        session_2 = np.concatenate((session_2, np.zeros((self.sequence_length, np.shape(session_2)[1]))))[:self.sequence_length]
        diff_2 = np.reshape((session_2[:, 1] - session_2[:, 0]) / 1E3, (np.shape(session_2)[0], 1))
        ascii_2 = np.reshape(session_2[:, 2] / 256, (np.shape(session_2)[0], 1))
        session_2_processed = np.concatenate((diff_2, ascii_2), axis=1)

        return (session_1_processed, session_2_processed), label

    def __len__(self):
        return self.len



file_loc = '{}/{}_dev_set.npy'.format(scenario, scenario)
data = np.load(file_loc, allow_pickle=True).item()


dev_set_users = list(data.keys())
random.shuffle(dev_set_users)
train_val_division = 0.8
train_users = dev_set_users[:int(len(dev_set_users)*train_val_division)]
val_users = dev_set_users[int(len(dev_set_users)*train_val_division):]


train_data = copy.deepcopy(data)
for user in list(data.keys()):
    if user not in train_users:
        del train_data[user]
val_data = copy.deepcopy(data)
for user in list(data.keys()):
    if user not in val_users:
        del val_data[user]
del data

batches_per_epoch_train = 16
batches_per_epoch_val = 4
batch_size_train = 512
batch_size_val = 512
sequence_length = 100

ds_t = PrepareData(train_data, sequence_length=sequence_length, samples_considered_per_epoch=batches_per_epoch_train*batch_size_train)
ds_v = PrepareData(val_data, sequence_length=sequence_length, samples_considered_per_epoch=batches_per_epoch_val*batch_size_val)

train_dataloader = DataLoader(ds_t, batch_size=batch_size_train)
val_dataloader = DataLoader(ds_v, batch_size=batch_size_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = Kick_Start_Model(input_size=2, hidden_size=32, output_size=32).double()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

MODEL_NAME = 'kick_start_model_{}'.format(scenario)


def train_one_epoch():
    running_loss = 0.
    batch_eers = []
    batch_losses = []
    for i, (input_data, labels) in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        input_data[0] = Variable(input_data[0]).double()
        input_data[1] = Variable(input_data[1]).double()
        input_data[0] = input_data[0].to(device)
        input_data[1] = input_data[1].to(device)
        pred1 = model(input_data[0])
        pred2 = model(input_data[1])
        loss = contrastive_loss(pred1, pred2, labels.to(torch.int64).double().to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        pred1 = pred1.cpu().detach()
        pred2 = pred2.cpu().detach()
        dists = nn.functional.pairwise_distance(pred1, pred2)
        fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
        fnr = 1 - tpr
        EER1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        batch_eers.append(np.mean([EER1, EER2])*100)
        batch_losses.append(running_loss)
    return np.sum(batch_losses), np.mean(batch_eers)


EPOCHS = 50

loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []

best_v_eer = 100.
for epoch in range(EPOCHS):
    start = time.time()
    print('EPOCH:', epoch)
    model.train()
    epoch_loss, epoch_eer = train_one_epoch()
    loss_t_list.append(epoch_loss)
    eer_t_list.append(epoch_eer)
    model.eval()
    running_loss_v = 0.
    with torch.no_grad():
        batch_eers_v = []
        batch_losses_v = []
        for i, (input_data, labels) in enumerate(val_dataloader, 0):
            input_data[0] = Variable(input_data[0]).double()
            input_data[1] = Variable(input_data[1]).double()
            input_data[0] = input_data[0].to(device)
            input_data[1] = input_data[1].to(device)
            pred1 = model(input_data[0])
            pred2 = model(input_data[1])            # criterion = torch.jit.script(nn.BCELoss())
            loss_v = contrastive_loss(pred1, pred2, labels.to(torch.int64).double().to(device))
            running_loss_v += loss_v.item()
            pred1 = pred1.cpu().detach()
            pred2 = pred2.cpu().detach()
            dists = nn.functional.pairwise_distance(pred1, pred2)
            fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
            fnr = 1 - tpr
            EER1_v = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            EER2_v = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
            batch_eers_v.append(np.mean([EER1_v, EER2_v])*100)
            batch_losses_v.append(running_loss_v)
    eer_v_list.append(np.mean(batch_eers_v))
    loss_v_list.append(np.sum(batch_losses_v))
    print("Epoch Loss on Training Set: " + str(epoch_loss))
    print("Epoch Loss on Validation Set: " + str(np.sum(batch_losses_v)))
    print("Epoch EER [%] on Training Set: "+ str(epoch_eer))
    print("Epoch EER [%] on Validation Set: " + str(np.mean(batch_eers_v)))
    if eer_v_list[-1] < best_v_eer:
        torch.save(model.state_dict(), MODEL_NAME + '.pt')
        print("New Best Epoch EER [%] on Validation Set: " + str(eer_v_list[-1]))
        best_v_eer = eer_v_list[-1]
    TRAIN_INFO_LOG_FILENAME = MODEL_NAME + '_log.txt'
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(TRAIN_INFO_LOG_FILENAME, "w") as output:
        output.write(str(log_list))
    end = time.time()
    print('Time for last epoch [min]: ' + str(np.round((end-start)/60, 2)))

with open('kick_start_model_{}_log.txt'.format(scenario), "r") as file:
    log_list = eval(file.readline())
figure, axis = plt.subplots(3)
figure.suptitle('Scenario: {}'.format(scenario))
axis[0].plot(log_list[0])
axis[0].set_title("Training Loss")
axis[0].set_ylabel('Loss')
axis[0].grid()
axis[1].plot(log_list[1])
axis[1].set_title("Validation Loss")
axis[1].set_ylabel('Loss')
axis[1].grid()
axis[2].plot(log_list[2], label='Training')
axis[2].plot(log_list[3], label='Validation')
axis[2].set_title("Training and Validation EER (%)")
axis[2].set_xlabel('Epochs')
axis[2].set_ylabel('EER (%)')
axis[2].legend()
axis[2].grid()
plt.show()

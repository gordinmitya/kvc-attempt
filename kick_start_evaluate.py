import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

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


def load_test_data_from_h5(file_path):
    import h5py
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])
    return data


scenario = 'mobile'
with open("{}/{}_comparisons.txt".format(scenario, scenario), "r") as file:
    comps = eval(file.readline())
# data = np.load('{}/{}_test_sessions.npy'.format(scenario, scenario), allow_pickle=True).item()
data = load_test_data_from_h5('{}/{}_test_sessions.h5'.format(scenario, scenario))

model = Kick_Start_Model(input_size=2, hidden_size=32, output_size=32).double()
model.load_state_dict(torch.load('kick_start_model_{}.pt'.format(scenario))) # , strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

def extract_keystroke_features(session_key, data_length = 100, dimension = 2):
    diff = np.reshape((session_key[:, 1] - session_key[:, 0]) / 1E3, (np.shape(session_key)[0], 1))
    ascii = np.reshape(session_key[:, 2] / 256, (np.shape(session_key)[0], 1))
    keys_features = np.concatenate((diff, ascii), axis = 1)
    output = np.concatenate((keys_features, np.zeros((data_length, dimension))))[:data_length]
    return output

data_length = 100
dimension = 2

preproc_data = {}
i = 0
for session in list(data.keys()):
    if i % 100 == 0:
        print(str(np.round(100*(i/len(list(data.keys()))), 2)) + '% raw data preprocessed', end='\r')
    preproc_data[session] = extract_keystroke_features(data[session], data_length=data_length, dimension=dimension)
    i = i + 1

batch_size = 5000
embeddings_list = []
model.eval()
with torch.no_grad():
    all_sessions = np.array(list(preproc_data.values()))
    for i in range(int(len(all_sessions)/batch_size)):
        input_data = np.reshape(all_sessions[i*batch_size:(i+1)*batch_size], (batch_size, data_length, dimension))
        input_data = Variable(torch.from_numpy(input_data)).double().to(device)
        embeddings_list.append(model(input_data).cpu())
        print(str(np.round(100 * ((i+1) / int(len(all_sessions)/batch_size)), 2)) + '% embeddings computed', end='\r')

embeddings_list = [item for sublist in embeddings_list for item in sublist]
embeddings = {}


tmp_list = list(preproc_data.keys())
for i in range(len(tmp_list)):
    embeddings[tmp_list[i]] = ''
embeddings.update(zip(embeddings, embeddings_list))


distances = {}
i = 0
for comp in comps:
    distance = nn.functional.pairwise_distance(embeddings[comp[0]], embeddings[comp[1]]).item()
    distances[str(comp)] = distance
    if i % 1000 == 0:
        print(str(np.round(100 * ((i + 1) / len(comps)), 2)) + '% distances computed', end='\r')
    i = i + 1

distances_list = list(distances.values())
max_dist = max(distances_list)
distances_list = [1-(x/max_dist) for x in distances_list]

with open('{}_predictions.txt'.format(scenario), "w") as file:
    file.write(str(distances_list))


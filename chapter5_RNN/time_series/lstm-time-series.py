import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable

np.random.seed(2017)
# load data
data_csv = pd.read_csv('./data.csv', usecols=[1])

# data preprocessing
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))


# create dataset
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


data_X, data_Y = create_dataset(dataset)

# split train set and test set
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)


# define network
class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1,
                 num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)  # seq, batch, hidden
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


model = lstm()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# In[195]:

total_epoch = 1000
for epoch in range(total_epoch):
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    out = model(var_x)
    loss = criterion(out, var_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('epoch {}, loss is {}'.format(epoch + 1, loss.data[0]))

torch.save(model.state_dict(), './lstm.pth')
model = model.eval()

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
predict = model(var_data)

predict = predict.cpu().data.numpy()

predict = predict.reshape(-1)

plt.plot(predict, 'r')
plt.plot(dataset, 'b')
plt.show()

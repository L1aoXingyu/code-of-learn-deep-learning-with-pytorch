import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd

# =============================Tensor================================
# Define 3x2 matrix with given values
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
print('a is: {}'.format(a))
print('a size is {}'.format(a.size()))  # a.size() = 3, 2

b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print('b is : {}'.format(b))

c = torch.zeros((3, 2))
print('zero tensor: {}'.format(c))

d = torch.randn((3, 2))
print('normal randon is : {}'.format(d))

a[0, 1] = 100
print('changed a is: {}'.format(a))

numpy_b = b.numpy()
print('conver to numpy is \n {}'.format(numpy_b))

e = np.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()
print('change data type to float tensor: {}'.format(f_torch_e))

if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)

# =============================Variable===================================

# Create Variable
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients
y.backward()  # same as y.backward(torch.FloatTensor([1]))
# Print out the gradients.
print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1

x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
print(y)

y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)


# ==============================nn.Module=================================
class net_name(nn.Module):
    def __init__(self, other_arguments):
        super(net_name, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        # other network layer

    def forward(self, x):
        x = self.conv1(x)
        return x

# ============================Dataset====================================


class myDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        data = (self.csv_data[idx], self.txt_data[idx])
        return data


dataiter = DataLoader(myDataset, batch_size=32, shuffle=True,
                      collate_fn=default_collate)

dset = ImageFolder(root='root_path', transform=None,
                   loader=default_loader)

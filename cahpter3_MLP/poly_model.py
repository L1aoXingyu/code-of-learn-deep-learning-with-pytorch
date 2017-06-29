__author__ = 'SherlockLiao'

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


# plot function curve and fitting curve
def plot_function(model):
    x_data = make_features(torch.arange(-1, 1, 0.01))
    y_data = f(x_data)
    if torch.cuda.is_available():
        y_pred = model(Variable(x_data).cuda())
    x = torch.arange(-1, 1, 0.01).numpy()
    y = y_data.numpy()
    y_p = y_pred.cpu().data.numpy()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'r', label='real curve')
    plt.plot(x, y_p, label='fitting curve')
    plt.legend(loc='best')
    plt.show()


# print funciton describe
def poly_desc(w, b):
    des = 'y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3'.format(
        b[0], w[0], w[1], w[2])
    return des


# get data
def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)


# Define model
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    # Get data
    batch_x, batch_y = get_batch()

    # Forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data[0]

    # Reset gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(print_loss, epoch))
print('==> Learned function:\t' + poly_desc(model.poly.weight.data.view(-1),
                                            model.poly.bias.data))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
plot_function(model)

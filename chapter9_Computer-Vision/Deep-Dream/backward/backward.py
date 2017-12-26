import torch
from torch.autograd import Variable

# simple gradient
a = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
b = a + 3
c = b * b * 3
out = c.mean()
out.backward()
print('*' * 10)
print('=====simple gradient======')
print('input')
print(a.data)
print('compute result is')
print(out.data[0])
print('input gradients are')
print(a.grad.data)

# backward on non-scalar output
m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True)
n = Variable(torch.zeros(1, 2))
n[0, 0] = m[0, 0]**2
n[0, 1] = m[0, 1]**3
n.backward(torch.FloatTensor([[1, 1]]))
print('*' * 10)
print('=====non scalar output======')
print('input')
print(m.data)
print('input gradients are')
print(m.grad.data)

# jacobian
j = torch.zeros(2, 2)
k = Variable(torch.zeros(1, 2))
m.grad.data.zero_()
k[0, 0] = m[0, 0]**2 + 3 * m[0, 1]
k[0, 1] = m[0, 1]**2 + 2 * m[0, 0]
k.backward(torch.FloatTensor([[1, 0]]), retain_variables=True)
j[:, 0] = m.grad.data
m.grad.data.zero_()
k.backward(torch.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('jacobian matrix is')
print(j)

# compute jacobian matrix
x = torch.FloatTensor([2, 1]).view(1, 2)
x = Variable(x, requires_grad=True)
y = Variable(torch.FloatTensor([[1, 2], [3, 4]]))

z = torch.mm(x, y)
jacobian = torch.zeros((2, 2))
z.backward(
    torch.FloatTensor([[1, 0]]), retain_variables=True)  # dz1/dx1, dz2/dx1
jacobian[:, 0] = x.grad.data
x.grad.data.zero_()
z.backward(torch.FloatTensor([[0, 1]]))  # dz1/dx2, dz2/dx2
jacobian[:, 1] = x.grad.data
print('=========jacobian========')
print('x')
print(x.data)
print('y')
print(y.data)
print('compute result')
print(z.data)
print('jacobian matrix is')
print(jacobian)
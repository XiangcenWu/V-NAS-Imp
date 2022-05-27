import torch


weight = torch.FloatTensor([1, 1])
print(weight.dtype, weight.shape)
loss = torch.nn.CrossEntropyLoss(weight=weight)



label = torch.tensor([1, 0])
pred_good = torch.tensor([
    [0.1, 10.],
    [10., 2.]
])

pred_bad = torch.tensor([
    [10., 0.1],
    [1.1, 2.9]
])


print(loss(pred_good, label))
print(loss(pred_bad, label))


import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        """
        x = x.reshape(x.shape[0], -1)
        x - self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) # in torch, not ort
        return output

def printSizes(model, name="Model"):
    print("\nParameters for {}".format(name))
    # print state dict
    print("Parameters and their Sizes")
    for param_name, param_values in model.state_dict().items():
        print("{} \t {}".format(param_name, param_values.size()))
        #print(param_values)

def compareModels(a, b):
    print("\n")
    for (_, a_vals), (_, b_vals) in zip(a.state_dict().items(), b.state_dict().items()):
        print("L2 Norm - {}".format(round(torch.norm(a_vals-b_vals).item(), 4)))
        a_np, b_np = a_vals.numpy().flatten(), b_vals.numpy().flatten()
        mse = ((a_np-b_np)**2).mean()
        me = np.abs(a_np-b_np).mean()
        print("MSE     - {}".format(np.round(mse, 4)))
        print("ME      - {}".format(np.round(me, 4)))
        print("\n")

# load models and restore weights
prefix = "models/"
torch_model = Net()
torch_model.load_state_dict(torch.load(prefix+"mnist_torch.pt"))

ort_model = Net()
ort_model.load_state_dict(torch.load(prefix+"mnist_ort.pt"))

printSizes(torch_model, "PyTorch")
printSizes(ort_model, "ORT")

compareModels(torch_model, ort_model)

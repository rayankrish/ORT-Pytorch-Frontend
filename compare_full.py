import torch
import torch.nn as nn
import numpy as np
import onnx
import os
from mpi4py import MPI
from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer
from onnxruntime.capi._pybind_state import set_cuda_device_id
from tqdm import tqdm
import pickle

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

def mnist_model_description():
    input_desc = IODescription('input1', ['batch', 784], torch.float32)
    label_desc = IODescription('label', ['batch', ], torch.int64, num_classes=10)
    loss_desc = IODescription('loss', [], torch.float32)
    probability_desc = IODescription('probability', ['batch', 10], torch.float32)
    return ModelDescription([input_desc, label_desc], [loss_desc, probability_desc])

def printSizes(model, name="Model"):
    print("\nParameters for {}".format(name))
    # print state dict
    print("Parameters and their Sizes")
    for param_name, param_values in model.state_dict().items():
        print("{} \t {}".format(param_name, param_values.size()))
        #print(param_values)
def compareModelsIndivdual(a, b, prnt = False): # b is ort
    for (name, a_vals) in a.state_dict().items():
        if name in b.state_dict():
            if prnt: print("{} Tensor".format(name))
            b_vals = b.state_dict()["Moment_2_"+name] if "Moment_2_"+name in b.state_dict() else b.state_dict()[name]
            a_np, b_np = a_vals.numpy().flatten(), b_vals.numpy().flatten()
            mse = ((a_np-b_np)**2).mean() 
            me = np.abs(a_np-b_np).mean()
            l2 = torch.norm(a_vals-b_vals).item()
            if prnt:
                print("MSE     - {}".format(np.round(mse, 4)))
                print("ME      - {}".format(np.round(me, 4)))
                print("L2 Norm - {}".format(round(torch.norm(a_vals-b_vals).item(), 4)))
                print("\n")

def compareModels(a, b, prnt = False): # b is ort
    mse_full = []
    me_full = []
    l2_full = []
    for (name, a_vals) in a.state_dict().items():
        if name in b.state_dict():
            b_vals = b.state_dict()["Moment_2_"+name] if "Moment_2_"+name in b.state_dict() else b.state_dict()[name]
            a_np, b_np = a_vals.numpy().flatten(), b_vals.numpy().flatten()
            mse_full.extend(((a_np-b_np)**2).tolist()) #mse = (a_np-b_np)**2).mean() 
            me_full.extend(np.abs(a_np-b_np).tolist()) #me = np.abs(a_np-b_np).mean()
            #l2 = torch.norm(a_vals-b_vals).item()
    #print(mse_full)
    #print(np.array(mse_full))
    mse = np.array(mse_full).mean()
    me = np.array(me_full).mean()
    l2 = np.linalg.norm(me_full)
    #print(l2)
    #print(mse)
    #print(me)
    return (l2, mse, me)

def compareLosses(a, b):
    #print(a)
    #print(b)
    #print(type(a[0]))
    #print(type(b[0]))
    loss_diffs = []
    for _a, _b, in zip(a, b):
        #print(abs(_a-_b))
        loss_diffs.append(abs(_a-_b))
    return loss_diffs

def setWeights(model, weight=1):
    new_dict = {}
    for name, vals in model.state_dict().items():
        new_dict[name] = torch.full(vals.size(), weight, dtype=torch.float32)
        #print(new_dict[name])
    return new_dict

use_cuda = False #torch.cuda.is_available()
comm = MPI.COMM_WORLD
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) if ('OMPI_COMM_WORLD_LOCAL_RANK' in os.environ) else 0
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK']) if ('OMPI_COMM_WORLD_RANK' in os.environ) else 0
world_size=comm.Get_size()
torch.cuda.set_device(local_rank)
if use_cuda:
    device = torch.device("cuda", local_rank)
else:
    device = torch.device("cpu")
n_gpu = 1
set_cuda_device_id(local_rank)

input_size = 784
hidden_size = 500
num_classes = 10
#model = NeuralNet(input_size, hidden_size, num_classes)

model_desc = mnist_model_description()
# use log_interval as gradient accumulate steps


# using the new, step data
run = 1 
l2 = []
mse = []
me = []
losses = []
for epoch in tqdm(range(1, 11, 1)):
    torch_losses, torch_parameters = torch.load("models/torch/run_{}_epoch_{}.pk".format(run, epoch))
    ort_losses, ort_parameters = torch.load("models/ort/run_{}_epoch_{}.pk".format(run, epoch))
    for i in range(len(torch_parameters)):
        torch_model = Net()
        torch_model.load_state_dict(torch_parameters[i])
        bin_str = ort_parameters[i].SerializeToString()
        model = onnx.load_model_from_string(bin_str)

        trainer = ORTTrainer(model, None, model_desc, "LambOptimizer", None, IODescription('Learning_Rate', [1,], torch.float32), device, gradient_accumulation_steps = 1, world_rank=world_rank, world_size=world_size, use_mixed_precision=False, allreduce_post_accumulation = True)

        errors = compareModels(torch_model, trainer)
        
        l2.append(errors[0])
        mse.append(errors[1])
        me.append(errors[2])
    losses.extend(compareLosses(torch_losses, ort_losses))
    


print(onnx.helper.printable_graph(model.graph))

# write comparison numbers to file
outfile = open("comparison_nums.pk", 'wb')
pickle.dump((losses, l2, mse, me), outfile)
outfile.close()




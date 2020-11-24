import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.input_dim=input_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        outputs = self.linear(x)
        return outputs

class LinearRegression(torch.nn.Module):
    def __init__(self,input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.input_dim=input_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        outputs = self.linear(x)
        return outputs

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        return x
'''


# MLP of one hidden layer
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.input_dim=input_dim
        self.fc1 = nn.Linear(input_dim, 1024) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)  
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
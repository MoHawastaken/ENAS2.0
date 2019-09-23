import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions.categorical as categorical
import torch.distributions.categorical as one_hot_categorical
import torch.distributions.bernoulli as bernoulli

import child_model as CM

# returns a ChildModelBatch containing num_samples models according to probabilities P_op and P_skip
def sampler(P_op, P_skip, num_samples):
    cat_op = categorical.Categorical(P_op)
    cat_sk = bernoulli.Bernoulli(P_skip)
    ops, sks = cat_op.sample([num_samples]), cat_sk.sample([num_samples])
    #print(ops.shape)
    #print(sks.shape)
    return CM.ChildModelBatch(ops, sks)

# returns the accuracies of a given list of torch child models on the test data set
def test(models, test_loader, loss=None):
    accuracies = []
    losses = []
    for model in models:
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                if not loss is None:
                    test_loss += loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct/len(test_loader.dataset)
        accuracies.append(accuracy)
        losses.append(test_loss)
    if loss is None:
        return accuracies
    else:
        return accuracies, losses

# trains the given model for one pass/epoch through the given training data
def train1(model, train_loader, optimizer, loss_func, log_interval=10):
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


class Controller(nn.Module):
    def __init__(self, num_nodes, num_child_samples=100, num_hidden=100, dim_w=64, learning_rate=0.00035, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Controller, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_hidden = num_hidden
        self.num_ops = len(CM.OPERATION_NAMES)
        self.num_child_samples = num_child_samples
        
        # define LSTM cells and output layers
        self.op_cell = nn.LSTMCell(self.num_hidden, self.num_hidden)
        self.op_out = nn.Linear(self.num_hidden, self.num_ops)
        self.sk_cell = nn.LSTMCell(self.num_hidden, self.num_hidden)
        
        wprev_init = nn.init.normal_(torch.Tensor(dim_w, self.num_hidden))
        wcurr_init = nn.init.normal_(torch.Tensor(dim_w, self.num_hidden))
        v_init = nn.init.normal_(torch.Tensor(dim_w, 1))
        
        self.W_prev = nn.Parameter(data=wprev_init)
        self.W_curr = nn.Parameter(data=wcurr_init)
        self.v = nn.Parameter(data=v_init)
        
        # ADAM parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # initialize 
        self.timestep = 0
        self.moment1 = [torch.zeros(p.size(), requires_grad=False) for p in self.parameters()]
        self.moment2 = [torch.zeros(p.size(), requires_grad=False) for p in self.parameters()]
             
    def update_step(self, cmb, Pop, Psk, R):
        self.zero_grad()
        dtheta = [torch.zeros(p.size()) for p in self.parameters()]

        for cell_i in range(2*Pop.size(0)):

            op_grad_node = [torch.zeros((self.num_ops, *list(p.size()))) for p in self.parameters()]

            node_ind = cell_i//2 # node index
            if node_ind % 2 == 0:
                for op in range(self.num_ops):
                    prob = Pop[node_ind, op] # probabiltiy of operation

                    # backward pass for this operation
                    v = torch.zeros(self.num_ops); v[op] = 1
                    Pop[node_ind].backward(v, retain_graph=True)

                    with torch.no_grad():
                        for pi, p in enumerate(self.parameters()):
                            if not p.grad is None:
                                grad = p.grad/cmb.batch_size()/prob
                                op_grad_node[pi][op].copy_(grad)
            else:
                # skip
                pass


            for samp_ind in range(cmb.batch_size()):
                op = cmb.ops[samp_ind, node_ind]
                for pi, p in enumerate(self.parameters()):
                    dtheta[pi] = dtheta[pi] + R[samp_ind]*op_grad_node[pi][op]
        
        # do ADAM update step
        with torch.no_grad():
            for pi, p in enumerate(self.parameters()):
                g = dtheta[pi]
                self.moment1[pi] = self.beta1*self.moment1[pi] + (1 - self.beta1)*g
                self.moment2[pi] = self.beta2*self.moment2[pi] + (1 - self.beta2)*g**2
                m1_hat = self.moment1[pi]/(1 - self.beta1**self.timestep)
                m2_hat = self.moment2[pi]/(1 - self.beta2**self.timestep)
                p -= self.learning_rate*m1_hat/(torch.sqrt(m2_hat) + self.epsilon)
        return True
    
    def controller_step(self, fixed_child_weights, testset_loader, g_emb=None):
        if g_emb is None:
            g_emb = torch.zeros(1, self.num_hidden, requires_grad=True)
        # forward pass through the LSTM controller
        Pop, Psk = self.forward(g_emb)
        
        # sample child models from the resulting probability distributions
        childmodelbatch = sampler(Pop, Psk, self.num_child_samples)
        #print(childmodelbatch.ops)
        #print(childmodelbatch.skips)
        torchchildmodels = childmodelbatch.to_torch_models(fixed_child_weights)
        
        # test child model performance
        acc = test(torchchildmodels, testset_loader)
        
        R = acc # for now the reward is equal to the accuracy
        
        self.update_step(childmodelbatch, Pop, Psk, R) # update controller weights with ADAM
                        
    def forward(self, g_emb):
        batch_size = g_emb.shape[0]
        h_prev = []
        h_prev.append(torch.zeros(batch_size, self.num_hidden))
        c_prev = []
        c_prev.append(torch.zeros(batch_size, self.num_hidden))
        P_skips = torch.zeros((int((self.num_nodes - 1)*self.num_nodes/2)))
        P_ops = torch.zeros((self.num_nodes, self.num_ops))
        
        sk_ind = 0
        for cell_i in range(2*self.num_nodes): # iterate over cells
            i = cell_i//2 # node index
            if cell_i % 2 == 0:
                # operation cell
                h_out, c_out = self.op_cell(g_emb, (h_prev[-1], c_prev[-1]))

                # calculate prob distribution operation at this node
                P_op = F.softmax(self.op_out(h_out), dim=1)
                P_ops[i] = P_op
            else:
                # skip connection cell
                h_out, c_out = self.sk_cell(g_emb, (h_prev[-1], c_prev[-1]))
                
                # calculate prob distribution for skip connections to this node
                for j in range(i):
                    Pij = torch.sigmoid(self.v.t() @ torch.tanh(self.W_prev @ h_prev[j].t() + self.W_curr @ h_out.t()))
                    P_skips[sk_ind] = Pij
                    sk_ind += 1
            # store hidden and cell state
            h_prev.append(h_out)
            c_prev.append(c_out)
            
        return P_ops, P_skips
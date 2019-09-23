import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from collections import OrderedDict
#from data import Image

#from . import data

# global variables & hyperparameters
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32        # batch size of training set
TEST_BATCH_SIZE = 1000 # batch size of test set
CHANNELS = 9           # number of (output) channels, constant throughout the network


## define possible layer operations

#def weight_sharing(f):
#    def g(node, *param):
#        layer = f(*param)    
#        if node ...:
#            layer.weight = load    
#
#@weight_sharing
def conv(in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=1, groups=1):
    same_padding = kernel_size//2
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=same_padding, groups=groups)
    
def conv3(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=3)

def conv5(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=5)

def depth_conv3(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=3, groups=in_channels)

def depth_conv5(in_channels=CHANNELS, out_channels=CHANNELS):
    return conv(in_channels, out_channels, kernel_size=5, groups=in_channels)

def max_pool(kernel_size=3, stride=3, padding=0):
    return nn.MaxPool2d(kernel_size=kernel_size, padding=0)

def avg_pool(kernel_size=3, stride=3, padding=0):
    return nn.AvgPool2d(kernel_size=kernel_size, padding=0)

def batch_norm(in_channels=CHANNELS):
    return nn.BatchNorm2d(in_channels, track_running_stats=False)

def relu():
    return nn.ReLU()

OPERATION_NAMES = ["conv3x3", "conv5x5", "depthconv3x3", "depthconv5x5", "maxpool", "avgpool"]
OPERATIONS = [conv3, conv5, depth_conv3, depth_conv5, max_pool, avg_pool]
#OPERATION_CATEGORIES = { 'conv ': [1,2], 'depth': [3,4]], 'pool': [5,6]}

#convert between string name and integer numbers
op_name_dict = {name: i for i,name in enumerate(OPERATION_NAMES) }
def enumerate_operation_names(operation_names):
    return [op_name_dict[name] for name in operation_names]

# returns pre and post concat image sizes of the given child model (and input size)
def generate_image_sizes1(child, input_size=32):
    N = child.number_of_nodes()
    
    current_size = input_size
    pre_concat_sizes = torch.zeros((N)) 
    post_concat_sizes = torch.zeros((N)) 
    
    for node in range(N):        
        #pooling layers resize the input
        if child.ops[node] == 4 or child.ops[node] == 5:
            current_size = current_size//3
        pre_concat_sizes[node] = current_size
        
        #concatenations from skip connections
        if node > 0:
            hood = node*(node - 1)//2 
            links = child.skips[hood:hood+node]
            links = links.float()
            
            max_neighboring_sizes = links*post_concat_sizes[:node]
            max_neighboring_size = torch.max(max_neighboring_sizes, dim=0)[0]
        else:
            max_neighboring_size = 0
        
        current_size = max(max_neighboring_size, current_size)
        
        post_concat_sizes[node] = current_size
    
    return pre_concat_sizes, post_concat_sizes

# Module implementing skip connections by concatenation
class SkipLayer(nn.Module):
    def __init__(self, node_index, link_indices, pre_imgsizes):
        super(SkipLayer, self).__init__()

        self.node_index = node_index
        self.pre_imgsizes = pre_imgsizes
        #print("Link indices:", link_indices)
        self.link_indices = (link_indices == 1).nonzero().squeeze(dim=1)
        #print(self.link_indices)
        
        if self.link_indices.size(0) > 0:
            self.node_inds = np.array([*self.link_indices, self.node_index]) # indices of nodes to be linked
            sizes = self.pre_imgsizes[self.node_inds] # relevant pre concat image sizes
            maxind = np.argmax(sizes) 
            self.out_size = sizes[maxind].int() # output image size must be the maximum
            
            self.pad_inds = [] # indices of nodes to be padded
            self.pad_list = nn.ModuleList() # list of padding modules
            
            for sind, size in enumerate(sizes):
                size = size.int()
                if size < self.out_size: # if size is smaller than output needs to be
                    shape_diff = int(self.out_size - size) # calculate difference
                    #print(shape_diff)
                    if shape_diff % 2 == 0: # even image dim difference
                        pad = shape_diff//2 # padding is exactly half of the difference
                        constpad = nn.ConstantPad2d(pad, 0)
                        #constpad = nn.ConstantPad2d((pad, pad, pad, pad), 0)
                    else: # odd image dim difference, TODO test ODD image dimensions
                        # need by-one-different padding for each side
                        pad_topleft = shape_diff//2
                        pad_bottomright = pad_topleft + 1
                        constpad = nn.ConstantPad2d((pad_topleft, pad_bottomright, pad_topleft, pad_bottomright), 0)
                    self.pad_list.append(constpad) # register padding operation
                    self.pad_inds.append(self.node_inds[sind])
            self.pad_inds = np.array(self.pad_inds)
            self.conv1 = conv(in_channels=CHANNELS*len(sizes)) # define conv1x1 to bring channel number back to CHANNELS
    
    def __call__(self, all_inputs):
        #print([inp.shape for inp in all_inputs])
        if len(self.link_indices) > 0: # there is at least one skip connection
            concat_inputs = []
            for inp_ind, inp in enumerate(all_inputs):
                if inp_ind in self.pad_inds: # if input needs to be padded
                    inpcopy = inp.clone() # copy input (because different paddings might be needed at two layers)
                    
                    pad_ind = np.where(self.pad_inds == inp_ind)[0][0]
                    padop = self.pad_list[pad_ind] # get padding operation
                    
                    padout = padop(inpcopy) # apply padding
                    concat_inputs.append(padout) # add padded output to concatenation list
                elif inp_ind in self.node_inds: # if this input is involved (but doesnt need padding)
                    concat_inputs.append(inp) # add to concat list
            catout = torch.cat(concat_inputs, dim=1) # concatenate all involved inputs
            convout = self.conv1(catout) # cross-correlate to change to CHANNELS channels
            return convout
        else: # return unchanged input
            return all_inputs[-1]

# PyTorch module for child model (shared_weights enables warm start, input_size is side length of the square input images, output_size is number of classes to be detected)
class TorchChildModel(nn.Module):
    def __init__(self, childmodel, shared_weights=None, input_size=32, output_size=10):
        super(TorchChildModel, self).__init__()
        
        self.childmodel = childmodel
        self.shared_weights = shared_weights
        
        self.pre_imgsizes, self.post_imgsizes = generate_image_sizes1(self.childmodel, input_size)
        
        # module containers for each layer and skip layer
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        self.input_layer = nn.Sequential(conv(in_channels=3),
                                         batch_norm()
                                         )
        
        if not self.shared_weights is None:
            new_state_dict = OrderedDict({"input_layer.0.weight": self.shared_weights.input_weights[0], # conv1x1
                                          "input_layer.0.bias": self.shared_weights.input_weights[1],
                                          "input_layer.1.weight": self.shared_weights.input_weights[2], # batch norm
                                          "input_layer.1.bias": self.shared_weights.input_weights[3]})
            self.load_state_dict(new_state_dict, strict=False)
        
        for nodeind in range(len(self.childmodel.ops)): # iterate over nodes
            opid = self.childmodel.ops[nodeind].int() # get current nodes operation
            op = OPERATIONS[opid]

            if opid == 0 or opid == 1: # conv3x3, conv5x5
                #if nodeind == 0:
                #    layer = nn.Sequential(op(in_channels=3),
                #                          batch_norm()
                #    )
                #else:
                layer = nn.Sequential(relu(),
                                      op(),
                                      batch_norm()
                )
            elif opid == 4 or opid == 5: # maxpool3x3, avgpool3x3
                padding = 1 # padding of 1 to avoid 2 columns/rows to be ignored
                curr_post_imgsize = self.post_imgsizes[nodeind]
                if curr_post_imgsize % 3 == 0: # if img size is divisible by 3 there is no need for padding
                    padding = 0
                #if nodeind == 0:
                #    layer = nn.Sequential(op(padding=padding),
                                          #conv(in_channels=3),
                #                          batch_norm()
                #    )
                #else:
                layer = nn.Sequential(relu(),
                                      op(padding=padding),
                                      batch_norm()
                )
            else: # depthwise separable 3x3, 5x5
                #if nodeind == 0:              
                #    layer = nn.Sequential(op(in_channels=3),
                #                          conv(), # additional conv1x1 for separable conv
                #                          batch_norm()
                #    )
                #else:
                layer = nn.Sequential(relu(),
                                      op(), 
                                      conv(), 
                                      batch_norm()
                )
            self.layers.append(layer)
            
            hood = nodeind*(nodeind - 1)//2 
            links = self.childmodel.skips[hood:hood + nodeind]
            skip_layer = SkipLayer(nodeind, links, self.pre_imgsizes)
            self.skip_layers.append(skip_layer)
            
            if not self.shared_weights is None:
                #before = self.layers[-1][1].weight.clone().detach()
                
                layer_keys = []
                for key in self.state_dict().keys():
                    if key.startswith("layers." + str(nodeind)):
                        layer_keys.append(key)
                #print(nodeind,opid)
                #print(layer_keys)
                #print(self.shared_weights[nodeind][opid])
                print("nodeind ",nodeind)
                assert len(layer_keys) == len(self.shared_weights.layer_weights[nodeind][opid]), "Not as many weights as should be, op {0}.".format(opid)
                new_state_dict = OrderedDict()
                for i_key, key in enumerate(layer_keys):
                    new_state_dict[key] = self.shared_weights.layer_weights[nodeind][opid][i_key]
                
                self.load_state_dict(new_state_dict, strict=False)
                
                #after = self.layers[-1][1].weight
                #print("Weight change:")
                #print((before - after == torch.zeros(after.size())).all())
        
        in_fully = self.post_imgsizes[-1]**2
        self.output_layer = nn.Linear(in_features=CHANNELS, out_features=output_size) # fully connected layer to classes
        
        if not self.shared_weights is None:
            new_state_dict = OrderedDict({"output_layer.weight": self.shared_weights.output_weights[0], # conv1x1
                                          "output_layer.bias": self.shared_weights.output_weights[1]
                                          })
            self.load_state_dict(new_state_dict, strict=False)
    
    def forward(self, x):
        outputs = []
        x = self.input_layer(x)
        for lid, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(x)
            skip_layer = self.skip_layers[lid]
            x = skip_layer(outputs)
        global_avg_pool_out = torch.flatten(x, start_dim=2, end_dim=-1).mean(dim=2, keepdim=False) # TODO change to spatial avg
        #print(global_avg_pool_out.size())
        output = self.output_layer(global_avg_pool_out)
        return F.softmax(output, dim=0)

class ChildModelBatch:
    def __init__(self, operations, skip_connections):
        self.ops = operations
        self.skips = skip_connections
        
        #sanity checks
        N = self.ops.size(1)
        n = self.skips.size(1)
        assert n == (N-1) * N / 2, 'size of skip connections incompatible with number of nodes'
        assert self.ops.size(0) == self.skips.size(0), 'batch size incoherent'
    
    def number_of_nodes(self): 
        return self.ops.size(1)
    
    def batch_size(self):
        return self.ops.size(0)

    def get_childmodel(self, i):
        assert i < self.batch_size(), 'index out of batch bounds'
        return ChildModel(self.ops[i], self.skips[i])

    def to_torch_models(self, weights):
        tm_list = []
        for i in range(self.batch_size()):
            cm = self.get_childmodel(i)
            tm_list.append(cm.to_torch_model(weights))
        return tm_list

class ChildModel:
    def __init__(self, operations, skip_connections):
        self.ops = operations
        self.skips = skip_connections
        
        #sanity checks
        N = self.ops.size(0)
        n = self.skips.size(0)
        assert n == (N-1) * N / 2, 'size of skip connections incompatible with number of nodes'
    
    def number_of_nodes(self): 
        return self.ops.size(0)

    def to_torch_model(self, weights):
        return TorchChildModel(self, weights)

def get_weight_sizes():
    # extracts the correct weight sizes
    all_ops = torch.Tensor([0, 1, 2, 3, 4, 5])
    len_skips = all_ops.size(0)*(all_ops.size(0) - 1)//2
    ch2 = ChildModel(all_ops,  torch.zeros(len_skips))
    tcm = TorchChildModel(ch2, input_size=32)
    
    input_weight_sizes = []
    input_weight_sizes.append(tcm.input_layer[0].weight.size()) # input conv1x1 
    input_weight_sizes.append(tcm.input_layer[0].bias.size())
    input_weight_sizes.append(tcm.input_layer[1].weight.size()) # input batch norm
    input_weight_sizes.append(tcm.input_layer[1].bias.size())
    
    layer_weight_sizes = []
    for j in range(len(OPERATIONS)):
        w_sizes_perop = []
        for key in tcm.state_dict().keys():
            if key.startswith("layers." + str(j)):
                #print(tcm.state_dict()[key].size())
                w_sizes_perop.append(tcm.state_dict()[key].size())
        layer_weight_sizes.append(w_sizes_perop)
        
    output_weight_sizes = []
    output_weight_sizes.append(tcm.output_layer.weight.size()) # output fully connected 
    output_weight_sizes.append(tcm.output_layer.bias.size())
    
    return input_weight_sizes, layer_weight_sizes, output_weight_sizes

# Weight sharing for each node and each possible operation
def He_init(ten, fan_in=32*32*CHANNELS): #He init: 2/FAN_IN with FAN_IN = 32*32*9
    std = np.sqrt(2/fan_in)
    return torch.nn.init.normal_(ten, std=std)

class SharedWeights:
    def __init__(self):
        self.input_weights = [] 
        self.output_weights = []
        self.layer_weights = []
    
    def init(self, inp, layer, out):
        self.input_weights = inp
        self.layer_weights = layer
        self.output_weights = out
    
    def clone(self):
        # clone input weights
        cloned_input_weights = []
        for i in range(len(self.input_weights)):
            cloned_input_weights.append(self.input_weights[i].clone())
            
        # clone input weights
        cloned_output_weights = []
        for i in range(len(self.output_weights)):
            cloned_output_weights.append(self.output_weights[i].clone())
                      
        # clone layer weights
        cloned_layer_weights = []
        for node in range(len(self.layer_weights)):
            node_W = []
            for op in range(len(self.layer_weights[node])):
                op_W = []
                for w in range(len(self.layer_weights[node][op])):
                    op_W.append(self.layer_weights[node][op][w].clone())
                node_W.append(op_W)
            cloned_layer_weights.append(node_W)
        # create new shared weights object
        cloned_shared_weights = SharedWeights()
        cloned_shared_weights.init(cloned_input_weights, cloned_layer_weights, cloned_output_weights)
        return cloned_shared_weights
    
def clone_weights(W):
    cloned_W = []
    for node in range(len(W)):
        node_W = []
        for op in range(len(W[node])):
            node_W.append(W[node][op])
        cloned_W.append(node_W)
    return cloned_W
    
def initialize_weights(num_nodes, weight_sizes=None, init_func=He_init):
    if weight_sizes is None:
        weight_sizes = get_weight_sizes()
    inp_sizes, layer_sizes, out_sizes = weight_sizes

    W = SharedWeights()
    # initialize input layer
    for size in inp_sizes:
        if len(size) == 4: # input conv weight
            inp_channels = size[1]
            fan_in = 32*32*inp_channels # adjust weight fan_in
        W.input_weights.append(init_func(torch.zeros(size, requires_grad=True), fan_in=fan_in))
    # initialize output layer
    for size in out_sizes:
        W.output_weights.append(init_func(torch.zeros(size, requires_grad=True)))
    # initialize hidden layers
    for node_ind in range(num_nodes):
        add_lst = []
        for op_ind in range(len(OPERATIONS)):
            add_lst2 = []
            for size in layer_sizes[op_ind]:
                add_lst2.append(init_func(torch.zeros(size, requires_grad=True)))
            add_lst.append(add_lst2)
        W.layer_weights.append(add_lst)
    return W
        
def generate_image_sizes(children, init_size=32):
    N = children.number_of_nodes()
    m = children.batch_size()
    
    current_sizes = torch.ones(m) * init_size
    pre_concat_sizes = torch.zeros((m,N)) 
    post_concat_sizes = torch.zeros((m,N)) 
    
    
    for node in range(N):
        print("\nNode ", str(node))
        
        #pooling layers
        pools = (children.ops[:, node] == 4) | (children.ops[:, node] == 5)
        current_sizes = torch.where(pools, current_sizes // 3, current_sizes)
        
        #print('curr', current_sizes)
    
        pre_concat_sizes[:,node] = current_sizes
        
        #concatinations from skip connections
        if node > 0:
            hood = node*(node-1) // 2 
            links = children.skips[:, hood:hood+node]
            links = links.float()
            
            print('previous sizes: ', post_concat_sizes[:, :node])
            print('link indicators: ', links)
            
            print('max neighboring sizes: ', max_neighboring_sizes)
            max_neighboring_sizes = links * pre_concat_sizes[:, :node]
            print('max neighboring sizes: ', max_neighboring_sizes)
            
            max_neighboring_sizes = links * post_concat_sizes[:, :node]
            max_neighboring_sizes = torch.max(max_neighboring_sizes, dim=1)[0]
            print('max neigbouring: ', max_neighboring_sizes)
            
        else:
            max_neighboring_sizes = torch.zeros(m)
        
        stacked = torch.stack( (max_neighboring_sizes, current_sizes) )
        current_sizes = torch.max(stacked, dim=0)[0]
        
        print('curr', current_sizes)
        
        post_concat_sizes[:,node] = current_sizes
    
    return pre_concat_sizes, post_concat_sizes
        

# Unit tests!
if __name__ == '__main__':
    
    #Define each child in test batch by hand
    
    operations_1 = [ "conv5x5", "maxpool", "conv3x3", "maxpool"]
    operations_1 = enumerate_operation_names(operations_1) 
    skip_connections_1 = [ 
            0,
            0, 0,
            0, 1, 1,
    ]
    
    
    operations_2 = [ "conv3x3", "avgpool", "conv5x5", "maxpool"]
    operations_2 = enumerate_operation_names(operations_2) 
    skip_connections_2 = [ 
            1,
            0, 0,
            0, 1, 0,
    ]
    
    operations_3 = [ "conv3x3", "avgpool", "conv5x5", "maxpool"]
    operations_3 = enumerate_operation_names(operations_3) 
    skip_connections_3 = [ 
            1,
            1, 0,
            0, 0, 0,
    ]
    
    operations_4 = [ "conv3x3", "maxpool", "maxpool", "maxpool"]
    operations_4 = enumerate_operation_names(operations_4) 
    skip_connections_4 = [ 
            1,
            1, 1,
            0, 0, 0,
    ]
    
    
    operations_5 = [ "conv3x3", "conv3x3", "conv3x3", "maxpool"]
    operations_5 = enumerate_operation_names(operations_5) 
    skip_connections_5 = [ 
            0,
            0, 1,
            0, 0, 0,
    ]
    
    
    # Now put them together to one batch
    children = 5
    operations = [eval("operations_" + str(child+1)) for child in range(children)]
    operations = torch.tensor(operations)
    #print(operations)
    skip_connections = [eval("skip_connections_" + str(child+1)) for child in range(children)]
    skip_connections = torch.tensor(skip_connections)
    
    #print('operations: ', operations.size())
    #print('skip connections: ', skip_connections.size())
    
    children = ChildModelBatch(operations,skip_connections)
    #pre, post = generate_image_sizes(children)
    
    #print('pre:', pre)
    #print('post:', post)
    
    #weight_sizes = get_weight_sizes()
    W = initialize_weights(3)
    ch = ChildModel(torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 1]))
    tcm = TorchChildModel(ch, shared_weights=W)
    
    #cifar10_img = Image(batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)
    #batch = iter(cifar10_img.train).next()
    #print(batch[0].shape)
    tcm.forward(torch.randn(1,3,32,32))
    #print(tcm.layers[1][1].weight-tcm.state_dict()["layers.1.1.weight"])
    #print(tcm.layers[1][1].weight-W[1][1][0])
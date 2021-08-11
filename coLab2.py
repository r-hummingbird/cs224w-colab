from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
root = './enzymes'
name = 'ENZYMES'

# The ENZYMES dataset
pyg_dataset= TUDataset('./enzymes', 'ENZYMES')

# You can find that there are 600 graphs in this dataset
print(pyg_dataset)
def get_num_classes(pyg_dataset):
  # TODO: Implement this function that takes a PyG dataset object
  # and return the number of classes for that dataset.

  num_classes = 0

  ############# Your code here ############
  ## (~1 line of code)
  ## Note
  ## 1. Colab autocomplete functionality might be useful.
  num_classes=pyg_dataset.num_classes
  #########################################

  return num_classes
# print(get_num_classes(pyg_dataset))
def get_num_features(pyg_dataset):
  # TODO: Implement this function that takes a PyG dataset object
  # and return the number of features for that dataset.

  num_features = 0

  ############# Your code here ############
  ## (~1 line of code)
  ## Note
  ## 1. Colab autocomplete functionality might be useful.
  num_features=pyg_dataset.num_features
  #########################################

  return num_features
num_classes = get_num_classes(pyg_dataset)
num_features = get_num_features(pyg_dataset)
print("{} dataset has {} classes".format(name, num_classes))
print("{} dataset has {} features".format(name, num_features))
def get_graph_class(pyg_dataset, idx):
  # TODO: Implement this function that takes a PyG dataset object,
  # the index of the graph in dataset, and returns the class/label
  # of the graph (in integer).

  label = -1

  ############# Your code here ############
  ## (~1 line of code)
  label=pyg_dataset[idx].y
  #########################################

  return label

# Here pyg_dataset is a dataset for graph classification
graph_0 = pyg_dataset[0]
print(graph_0)
idx = 100
label = get_graph_class(pyg_dataset, idx)
print('Graph with index {} has label {}'.format(idx, label))
def get_graph_num_edges(pyg_dataset, idx):
  # TODO: Implement this function that takes a PyG dataset object,
  # the index of the graph in dataset, and returns the number of
  # edges in the graph (in integer). You should not count an edge
  # twice if the graph is undirected. For example, in an undirected
  # graph G, if two nodes v and u are connected by an edge, this edge
  # should only be counted once.

  num_edges = 0

  ############# Your code here ############
  ## Note:
  ## 1. You can't return the data.num_edges directly
  ## 2. We assume the graph is undirected
  ## (~4 lines of code)
  num_edges=pyg_dataset[idx].num_edges//2
  #########################################

  return num_edges

idx = 200
num_edges = get_graph_num_edges(pyg_dataset, idx)
print('Graph with index {} has {} edges'.format(idx, num_edges))

dataset_name = 'ogbn-arxiv'
# Load the dataset and transform it to sparse tensor
dataset = PygNodePropPredDataset(name=dataset_name,
                                 transform=T.ToSparseTensor())
print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

# Extract the graph
data = dataset[0]
print(data)
def graph_num_features(data):
  # TODO: Implement this function that takes a PyG data object,
  # and returns the number of features in the graph (in integer).

  num_features = 0

  ############# Your code here ############
  ## (~1 line of code)
  num_features=data.num_features
  #########################################

  return num_features

num_features = graph_num_features(data)
print('The graph has {} features'.format(num_features))

dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name,
                                 transform=T.ToSparseTensor())
data = dataset[0]

# Make the adjacency matrix to symmetric
data.adj_t = data.adj_t.to_symmetric()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If you use GPU, the device should be cuda
print('Device: {}'.format(device))

data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)


class GCN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
               dropout, return_embeds=False):
    # TODO: Implement this function that initializes self.convs,
    # self.bns, and self.softmax.

    super(GCN, self).__init__()

    # A list of GCNConv layers
    self.convs = None

    # A list of 1D batch normalization layers
    self.bns = None

    # The log softmax layer
    self.softmax = None

    ############# Your code here ############
    ## Note:
    ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
    ## 2. self.convs has num_layers GCNConv layers
    ## 3. self.bns has num_layers - 1 BatchNorm1d layers
    ## 4. You should use torch.nn.LogSoftmax for self.softmax
    ## 5. The parameters you can set for GCNConv include 'in_channels' and
    ## 'out_channels'. More information please refer to the documentation:
    ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
    ## More information please refer to the documentation:
    ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    ## (~10 lines of code)
    self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
    self.convs.extend([GCNConv(hidden_dim, hidden_dim) for i in range(num_layers - 2)])
    self.convs.extend([GCNConv(hidden_dim, output_dim)])

    self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers - 1)])
    self.softmax = torch.nn.LogSoftmax()
    #########################################

    # Probability of an element to be zeroed
    self.dropout = dropout

    # Skip classification layer and return node embeddings
    self.return_embeds = return_embeds

  def reset_parameters(self):
    for conv in self.convs:
      conv.reset_parameters()
    for bn in self.bns:
      bn.reset_parameters()

  def forward(self, x, adj_t):
    # TODO: Implement this function that takes the feature tensor x,
    # edge_index tensor adj_t and returns the output tensor as
    # shown in the figure.

    out = None

    ############# Your code here ############
    ## Note:
    ## 1. Construct the network as showing in the figure
    ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
    ## More information please refer to the documentation:
    ## https://pytorch.org/docs/stable/nn.functional.html
    ## 3. Don't forget to set F.dropout training to self.training
    ## 4. If return_embeds is True, then skip the last softmax layer
    ## (~7 lines of code)
    for i in range(len(self.convs) - 1):
      x = self.convs[i](x, adj_t)
      x = self.bns[i](x)
      x = F.relu(x)
      x = F.dropout(x, self.dropout, self.training)
    out = self.convs[len(self.convs) - 1](x, adj_t)  # GCNVonv
    if not self.return_embeds:
      out = self.softmax(out)
    #########################################

    return out


def train(model, data, train_idx, optimizer, loss_fn):
  # TODO: Implement this function that trains the model by
  # using the given optimizer and loss_fn.
  model.train()
  loss = 0

  ############# Your code here ############
  ## Note:
  ## 1. Zero grad the optimizer
  ## 2. Feed the data into the model
  ## 3. Slicing the model output and label by train_idx
  ## 4. Feed the sliced output and label to loss_fn
  ## (~4 lines of code)
  optimizer.zero_grad()
  out = model(data.x, data.adj_t)
  loss = loss_fn(out[train_idx], data.y[train_idx].squeeze(1))
  #########################################

  loss.backward()
  optimizer.step()

  return loss.item()


# Test function here
@torch.no_grad()
def test(model, data, split_idx, evaluator):
  # TODO: Implement this function that tests the model by
  # using the given split_idx and evaluator.
  model.eval()

  # The output of model on all data
  out = None

  ############# Your code here ############
  ## (~1 line of code)
  ## Note:
  ## 1. No index slicing here
  out = model(data.x, data.adj_t)
  #########################################

  y_pred = out.argmax(dim=-1, keepdim=True)

  train_acc = evaluator.eval({
    'y_true': data.y[split_idx['train']],
    'y_pred': y_pred[split_idx['train']],
  })['acc']
  valid_acc = evaluator.eval({
    'y_true': data.y[split_idx['valid']],
    'y_pred': y_pred[split_idx['valid']],
  })['acc']
  test_acc = evaluator.eval({
    'y_true': data.y[split_idx['test']],
    'y_pred': y_pred[split_idx['test']],
  })['acc']

  return train_acc, valid_acc, test_acc


# Please do not change the args
args = {
  'device': device,
  'num_layers': 3,
  'hidden_dim': 256,
  'dropout': 0.5,
  'lr': 0.01,
  'epochs': 100,
}
args
model = GCN(data.num_features, args['hidden_dim'],
            dataset.num_classes, args['num_layers'],
            args['dropout']).to(device)
evaluator = Evaluator(name='ogbn-arxiv')
import copy

# reset the parameters to initial random value
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = F.nll_loss

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
  loss = train(model, data, train_idx, optimizer, loss_fn)
  result = test(model, data, split_idx, evaluator)
  train_acc, valid_acc, test_acc = result
  if valid_acc > best_valid_acc:
    best_valid_acc = valid_acc
    best_model = copy.deepcopy(model)
  print(f'Epoch: {epoch:02d}, '
        f'Loss: {loss:.4f}, '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')
best_result = test(best_model, data, split_idx, evaluator)
train_acc, valid_acc, test_acc = best_result
print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')

# Load the dataset
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

split_idx = dataset.get_idx_split()

# Check task type
print('Task type: {}'.format(dataset.task_type))
# Load the data sets into dataloader
# We will train the graph classification task on a batch of 32 graphs
# Shuffle the order of graphs for training set
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)
# Please do not change the args
args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.001,
    'epochs': 30,
}
args
### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(hidden_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = None

        ############# Your code here ############
        ## Note:
        ## 1. Initialize the self.pool to global mean pooling layer
        ## More information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## (~1 line of code)
        self.pool=global_mean_pool
        #########################################

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # TODO: Implement this function that takes the input tensor batched_data,
        # returns a batched output tensor for each graph.
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct node embeddings using existing GCN model
        ## 2. Use global pooling layer to construct features for the whole graph
        ## More information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## 3. Use a linear layer to predict the graph property
        ## (~3 lines of code)
        out=self.gnn_node(embed,edge_index)
        out=self.pool(out,batch)
        out=self.linear(out)

        #########################################

        return out


def train(model, device, data_loader, optimizer, loss_fn):
  # TODO: Implement this function that trains the model by
  # using the given optimizer and loss_fn.
  model.train()
  loss = 0

  for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
    batch = batch.to(device)

    if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
      pass
    else:
      ## ignore nan targets (unlabeled) when computing training loss.
      is_labeled = batch.y == batch.y

      ############# Your code here ############
      ## Note:
      ## 1. Zero grad the optimizer
      ## 2. Feed the data into the model
      ## 3. Use `is_labeled` mask to filter output and labels
      ## 4. You might change the type of label
      ## 5. Feed the output and label to loss_fn
      ## (~3 lines of code)
      optimizer.zero_grad()
      out=model(batch)
      loss=loss_fn(out[is_labeled],batch.y[is_labeled].type_as(out))
      #########################################

      loss.backward()
      optimizer.step()

  return loss.item()
# The evaluation function
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
model = GCN_Graph(args['hidden_dim'],
            dataset.num_tasks, args['num_layers'],
            args['dropout']).to(device)
evaluator = Evaluator(name='ogbg-molhiv')
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss()

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
  print('Training...')
  loss = train(model, device, train_loader, optimizer, loss_fn)

  print('Evaluating...')
  train_result = eval(model, device, train_loader, evaluator)
  val_result = eval(model, device, valid_loader, evaluator)
  test_result = eval(model, device, test_loader, evaluator)

  train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]
  if valid_acc > best_valid_acc:
      best_valid_acc = valid_acc
      best_model = copy.deepcopy(model)
  print(f'Epoch: {epoch:02d}, '
        f'Loss: {loss:.4f}, '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')
train_acc = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
valid_acc = eval(best_model, device, valid_loader, evaluator)[dataset.eval_metric]
test_acc = eval(best_model, device, test_loader, evaluator)[dataset.eval_metric]

print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')
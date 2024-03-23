###############################################################################
# trains quantum circuit 
# By default, it uses nesterov momentum optimizer to train 200 iterations 
# with batch size of 25. 
# Both MSE Loss and cross entropy loss can be used for circuit training. 
# Number of total parameters (total_params) 
# need to be adjusted when testing different QCNN structures.
###############################################################################

# Implementation of Quantum circuit training procedure
import circuit
import data_matt
import math
import pennylane as qml
from pennylane import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

# def square_loss(labels, predictions):
#     loss = 0
#     for l, p in zip(labels, predictions):
#         loss = loss + (l - p) ** 2
#     loss = loss / len(labels) 
#     return loss


# def cross_entropy(labels, predictions):
#     loss = 0
#     for l, p in zip(labels, predictions):
#         print(l,p)
#         c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
#         loss = loss + c_entropy
#     return -1 * loss

# # def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
# def dev_circuits(params, X, Y, hierq):
#     predictions = [circuit.QCNN(X=x, params=params, hierq=hierq) for x in X]
#     return predictions

# def cost(params, X, Y, hierq):
#     predictions = [circuit.QCNN(X=x, params=params, hierq=hierq)() for x in X]
    
#     # if cost_fn == 'mse':
#         # loss = square_loss(Y, predictions)
#     # elif cost_fn == 'cross_entropy':

#     loss = cross_entropy(Y, predictions)
#     return loss

# def cost(motif, symbols, x, y):
#     motif.set_symbols(symbols)
#     circuit = get_circuit(motif, x)
#     y_hat = circuit()
#     loss = nn.BCELoss()
#     # loss = nn.MSELoss() # use MSE

#     # index 1 corresponds to predictions for being in class 1
#     loss = loss(y_hat[:, 1], torch.tensor(y.values, dtype=torch.double))
#     return loss

def cost2(y_hat, y):
    # loss = nn.BCELoss()
    loss_fn = nn.MSELoss() # use MSE
    # loss.requires_grad = True
    # loss = nn.MSELoss() # use MSE
    # index 1 corresponds to predictions for being in class 1
    loss = loss_fn(y_hat[:, 1], torch.tensor(y, dtype=torch.double))
    # loss = loss(y_hat[:, 1], y)
    return loss

# Circuit training parameters
# learning_rate = 0.01
# batch_size = 32
# batch: defines the number of samples to work through before updating the internal model parameters.
# iteration: passing through the training examples in a batch
# epoch: number times that the learning algorithm will work through the entire training dataset.
# def circuit_training(X_train, Y_train, hierq):
#     n_params = hierq.n_symbols

#     # random params every time? @TODO 
#     params = np.random.randn(total_params, requires_grad=True)
#     opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
#     loss_history = []

#     # @TODO what abut epochs?
#     steps = math.ceil(len(X_train)/batch_size)
#     for it in range(steps):
#         start = it*batch_size
#         end = start + batch_size if (len(X_train)-1) >= (start + batch_size) else (len(X_train)-1)
#         X_batch = [X_train[i] for i in range(start, end)]
#         Y_batch = [Y_train[i] for i in range(start, end)]

#         params, cost_new = opt.step_and_cost(
#         #   lambda v: 
#             # cost(v, X_batch, Y_batch, hierq),
#             # params
#             cost(params, X_batch, Y_batch, hierq),
#             params
#         )

#         loss_history.append(cost_new)
#         if it % 10 == 0:
#             print("iteration: ", it, " cost: ", cost_new)
#     return loss_history, params

# set up train loop
def train(X_train, Y_train, motif, epochs=70, lr=0.1, batch_size=10, verbose=True):
    n_symbols = motif.n_symbols
    if n_symbols > 0:
        symbols = torch.rand(n_symbols, requires_grad=True)
        opt = torch.optim.Adam([symbols], lr=lr)

        dataset_train = data_matt.Build_Data(X_train, Y_train)
        train_loader_10 = DataLoader(dataset=dataset_train, batch_size=batch_size)
        size = len(train_loader_10.dataset)

        for epoch in range(epochs):  # loop over the dataset multiple times
            step = 0
            for i, data in enumerate(train_loader_10, 0):
                x = data[0]
                y = data[1]
                # get outputs of prediction
                outputs = circuit.net(motif, symbols, x)
                # calculating the loss between original and predicted data points
                # loss = cost(motif, symbols, x, y)
                loss = cost2(outputs, y)
                # backward pass for computing the gradients of the loss w.r.t to learnable parameters
                loss.requires_grad = True
                loss.backward()
                # updating the parameters after each iteration
                opt.step()
                print(symbols.grad)
                opt.zero_grad() # zeroing gradients after each iteration
                # if verbose:
                #     print(f"Loss at epoch {epoch} step {step}: {loss}")
                step += 1

                if i % 5 == 0:
                    loss, current = loss.item(), i * batch_size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if verbose:
                print(f"Loss at epoch {epoch}: {loss}")
    else:
        symbols = None
        loss = cost(motif, [], x, y)
    return symbols, loss

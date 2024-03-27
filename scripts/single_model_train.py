from collections import namedtuple
import pandas as pd
import sympy as sp
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from hierarqcal import Qcycle, Qmask, Qinit, Qunitary, Qmotif
import pennylane as qml
from pennylane.templates.embeddings import AngleEmbedding
from pennylane import numpy as np
import torch
from torch import nn

# seed for reproducibility
torch.manual_seed(111)
np.random.seed(111)

##############################################################################
### CIRCUIT & TRAINING
##############################################################################

def get_circuit(hierq, x=None):
    dev = qml.device("default.qubit.torch", wires=hierq.tail.Q, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit():
        if isinstance(next(hierq.get_symbols(), False), sp.Symbol):
            # Pennylane doesn't support symbolic parameters, so if no symbols were set (i.e. they are still symbolic), we initialize them randomly
            hierq.set_symbols(np.random.uniform(0, 2 * np.pi, hierq.n_symbols))
        if x is not None:
            AngleEmbedding(x, wires=hierq.tail.Q, rotation="Y")
        hierq(backend="pennylane")  # This executes the compute graph in order
        return qml.probs(wires=hierq.head.Q[0])

    return circuit

def train(x, y, motif, N=100, lr=0.1, verbose=True):
    n_symbols = motif.n_symbols
    torch.manual_seed(111)
    np.random.seed(111)
    if n_symbols > 0:
        symbols = torch.rand(n_symbols, requires_grad=True)
        if verbose:
            print(f"Initial symbols: {symbols}")
        opt = torch.optim.Adam([symbols], lr=lr)
        for it in range(N):
            opt.zero_grad() # reset gradients
            y_hat = net(motif, symbols, x)
            loss = objective_function(y_hat, y)
            loss.backward()
            opt.step()

            if verbose:
                if it % 25 == 0:
                    print(f"Loss at step {it}: {loss}")
    else:
        symbols = None
        loss = objective_function(motif, [], x, y)
    return symbols, loss

def net(motif, symbols, x):
    motif.set_symbols(symbols)
    circuit = get_circuit(motif, x)
    y_hat = circuit()

    return y_hat

def objective_function(y_hat, y):
    loss = nn.BCELoss()
    # loss = nn.MSELoss()
    assert(len(y_hat) == len(y))
    # index 1 corresponds to predictions for being in class 1
    loss = loss(y_hat[:, 1], torch.tensor(y, dtype=torch.double))
    return loss

def accuracy(y_hat, y_test):
    y_hat = torch.argmax(y_hat, axis=1).detach().numpy()
    assert(len(y_hat) == len(y_test))
    accuracy = sum(
            [y_hat[k] == y_test[k] for k in range(len(y_hat))]
        ) / len(y_hat)

    return accuracy

##############################################################################
### CIRCUIT UTILS
##############################################################################

def penny_gate_to_function(gate):
    return lambda bits, symbols: gate(*symbols, wires=[*bits])

primitive_gates = ["CRZ", "CRX", "CRY", "RZ", "RX", "RY", "Hadamard", "CNOT", "PauliX"]
penny_gates = [getattr(qml, gate_name) for gate_name in primitive_gates]
hierq_gates = {
    primitive_gate: Qunitary(
        penny_gate_to_function(penny_gate),
        n_symbols=penny_gate.num_params,
        arity=penny_gate.num_wires,
    )
    for primitive_gate, penny_gate in zip(primitive_gates, penny_gates)
}

def draw_circuit(circuit, **kwargs):
    fig, ax = qml.draw_mpl(circuit)(**kwargs)

##############################################################################
### ANSATZES
##############################################################################

# ========================================== 
# == CONVOLUTION ANSATZES

def ansatz_conv_g(bits, symbols):  # 10 params
    qml.RX(symbols[0], wires=bits[0])
    qml.RX(symbols[1], wires=bits[1])
    qml.RZ(symbols[2], wires=bits[0])
    qml.RZ(symbols[3], wires=bits[1])
    qml.CRZ(symbols[4], wires=[bits[1], bits[0]])
    qml.CRZ(symbols[5], wires=[bits[0], bits[1]])
    qml.RX(symbols[6], wires=bits[0])
    qml.RX(symbols[7], wires=bits[1])
    qml.RZ(symbols[8], wires=bits[0])
    qml.RZ(symbols[9], wires=bits[1])
U_ansatz_conv_g = Qunitary(ansatz_conv_g, n_symbols=10, arity=2)

def ansatz_conv_a(bits, symbols=None):  # 2 params
  qml.RY(symbols[0], wires=[bits[0]])
  qml.RY(symbols[1], wires=[bits[1]])
  qml.CNOT(wires=[bits[0], bits[1]])
U_ansatz_conv_a = Qunitary(ansatz_conv_a, n_symbols=2, arity=2)

# The same could be written in HierarQcal
# ansatz_ex2 = (
#     Qinit(2)
#     + Qmotif(E=[(0, 1)], mapping=hierq_gates["CRY"])
#     + Qmotif(E=[(1, 0)], mapping=hierq_gates["CRY"])
#     + Qmotif(E=[(0, 1)], mapping=hierq_gates["CNOT"])
# )

# ========================================== 
# == POOLING ANSATZES

def ansatz_pool_1(bits, symbols=None):
  qml.CRZ(symbols[0], wires=[bits[0], bits[1]])
  qml.PauliX(wires=[bits[0]])
  qml.CRX(symbols[1], wires=[bits[0], bits[1]])
U_ansatz_pool_1 = Qunitary(ansatz_pool_1, n_symbols=2, arity=2)

# The same could be written in HierarQcal
# ansatz_pool_2 = (
    # Qinit(2)
    # + Qmotif(E=[(0, 1)], mapping=hierq_gates["CRZ"])
    # + Qmotif(E=[(0,)], mapping=hierq_gates["PauliX"])
    # + Qmotif(E=[(0, 1)], mapping=hierq_gates["CRX"])
# )

def ansatz_pool_2(bits, symbols=None):
    qml.CNOT(wires=[bits[0], bits[1]])
U_ansatz_pool_2 = Qunitary(ansatz_pool_2, n_symbols=0, arity=2)


##############################################################################
### MOTIF BUILDER
##############################################################################

def qcnn_motif(ansatz_c=U_ansatz_conv_a, conv_stride=1, conv_step=1, conv_offset=0, share_weights=True, ansatz_p=U_ansatz_pool_1, pool_filter="!*"):
    qcnn = (
        Qinit(8)
        + (
            Qcycle(
                stride=conv_stride,
                step=conv_step,
                offset=conv_offset,
                mapping=ansatz_c,
                share_weights=share_weights,
            )
            + Qmask(pool_filter, mapping=ansatz_p)
        )
        * 3
    )

    return qcnn
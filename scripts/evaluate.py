import single_model_train
import data
from single_model_train import U_ansatz_conv_a, U_ansatz_conv_g, U_ansatz_conv_b, U_ansatz_pool_2, U_ansatz_pool_1

# ===================================
# This file is only used for single model training and test
# An example scenario is to run the multi model search and then test the 
# best configuration with this code
# ================================

qcnn = single_model_train.qcnn_motif(
  ansatz_c="g",
  ansatz_p="2", 
  conv_stride=7, conv_step=1, conv_offset=0, 
  pool_filter="!*!", pool_stride=3, 
  share_weights=False
)
samples_preprocessed = data.data_load_and_process(["rock", "country"])

print(f"# params: {qcnn.n_symbols}")

symbols, loss = single_model_train.train(samples_preprocessed.x_train, samples_preprocessed.y_train.values, qcnn)
circuit = single_model_train.get_circuit(qcnn, samples_preprocessed.x_test)
y_hat = circuit()
acc = single_model_train.accuracy(y_hat,samples_preprocessed.y_test.values)
print(f"accuracy: {acc}")
print(f"symbols: {symbols}")

# from numpy import arange
# from matplotlib.pylab import plt
# epochs = range(1, 100)
# plt.plot(epochs, loss_history, label='Training Loss')
 
# # Add in a title and axes labels
# plt.title('Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
 
# # Set the tick locations
# plt.xticks(arange(0, 100, 2))
 
# # Display the plot
# plt.legend(loc='best')
# plt.show()
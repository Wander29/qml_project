import single_model_train
import data
from single_model_train import U_ansatz_conv_a, U_ansatz_conv_g, U_ansatz_conv_b, U_ansatz_pool_2, U_ansatz_pool_1

# ===================================
# This file is only used for single model training and test
# An example scenario is to run the multi model search and then test the 
# best configuration with this code
# ================================

qcnn = single_model_train.qcnn_motif(
  ansatz_c=U_ansatz_conv_a,
  ansatz_p=U_ansatz_pool_1, 
  conv_stride=1, conv_step=1, conv_offset=0, 
  pool_filter="*!", pool_stride=2, 
  share_weights=True
)
samples_preprocessed = data.data_load_and_process(["rock", "country"])

print(f"# params:{qcnn.n_symbols}")

symbols, loss = single_model_train.train(samples_preprocessed.x_train, samples_preprocessed.y_train.values, qcnn)
circuit = single_model_train.get_circuit(qcnn, samples_preprocessed.x_test)
y_hat = circuit()
acc = single_model_train.accuracy(y_hat,samples_preprocessed.y_test.values)
print(f"accuracy: {acc}")
print(f"symbols: {symbols}")
import single_model_train
import data
from single_model_train import U_ansatz_conv_a, U_ansatz_conv_g, U_ansatz_conv_b, U_ansatz_pool_2

qcnn = single_model_train.qcnn_motif(ansatz_c=U_ansatz_conv_g, conv_stride=7, conv_step=1, conv_offset=0, ansatz_p=U_ansatz_pool_2, pool_filter="!*!", pool_stride=3, share_weights=True)
samples_preprocessed = data.data_load_and_process(["rock", "country"])

print(qcnn.n_symbols)

symbols, loss = single_model_train.train(samples_preprocessed.x_train, samples_preprocessed.y_train.values, qcnn)
circuit = single_model_train.get_circuit(qcnn, samples_preprocessed.x_test)
y_hat = circuit()
acc = single_model_train.accuracy(y_hat,samples_preprocessed.y_test.values)
print(f"accuracy: {acc}")
print(f"symbols: {symbols}")
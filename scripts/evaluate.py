import single_model_train
import data
from single_model_train import U_ansatz_conv_a

qcnn = single_model_train.qcnn_motif(U_ansatz_conv_a, 4, "!*")
samples_preprocessed = data.data_load_and_process(["rock", "country"])

symbols, loss = single_model_train.train(samples_preprocessed.x_train, samples_preprocessed.y_train.values, qcnn)
circuit = single_model_train.get_circuit(qcnn, samples_preprocessed.x_test)
y_hat = circuit()
acc = single_model_train.accuracy(y_hat,samples_preprocessed.y_test.values)
print(f"accuracy: {acc}")
print(f"symbols: {symbols}")

# # ====================
# #-- Motif 2
# # try setting share weights False
# # setting share_weights to False increases accuracy
# qcnn = (
#     Qinit(8)
#     + (
#         Qcycle(
#             stride=1,
#             step=1,
#             offset=0,
#             mapping=U_ansatz_conv_a,
#             share_weights=True,
#         )
#         + Qmask("01", mapping=U_ansatz_pool_1)
#     )
#     * 3
# )
# # plot circuit
# fig, ax = qml.draw_mpl(get_circuit(qcnn))()
# # train qcnn
# symbols, loss = train(samples_preprocessed.x_train, samples_preprocessed.y_train, qcnn)
# # get predictions
# circuit = get_circuit(qcnn, samples_preprocessed.x_test)
# y_hat = circuit()
# acc = accuracy(y_hat,samples_preprocessed.y_test.values)
# print(f"accuracy: {acc}")
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout
from keras.models import Model
from tensorflow_addons.layers import CRF


def create_model(maxlen, dic, word_size, infer=False):
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(len(dic) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    model = Model(sequence, output)
    if not infer:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


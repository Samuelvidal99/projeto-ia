from __future__ import annotations

import json
import pickle as pkl
import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def run_model(params: dict, threshold: int, random: int) -> float:
    # Configurar seeds
    np.random.seed(random)
    tf.random.set_seed(random)

    # Configurar hiperparâmetros estáticos
    valid_size = 0.25
    embedding_dim = 300

    # Carregar dados
    def read_json(path):
        with open(path) as f:
            data = json.loads(f.read())
        return data

    train_set = read_json('data/train_set.json')
    test_set = read_json('data/test_set.json')

    X_train = [i['text'] for i in train_set]
    X_test = [i['text'] for i in test_set]

    y_train = np.array([float(i['label']) for i in train_set], dtype='float32')
    y_test = np.array([float(i['label']) for i in test_set], dtype='float32')

    # Ajustar tokenizador
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    # Configurar alguns parâmetros
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 30

    # Converter os textos para sequências
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=maxlen)

    '''
    Para selecionarmos o método de otimização usamos o parâmetro limiar e
    optimização. Se a optimização ultrapassar o limiar, o método selecionado será
    o Adam, caso contrário, será o SGD.
    '''
    optimization = None
    if params['optimization'] >= threshold:
        optimization = 'adam'
    else:
        optimization = 'rmsprop'

    '''
    Para selecionarmos a função de ativação usamos o parâmetro limiar e
    ativação. Se o valor de ativação ultrapassar o limiar, a função selecionada
    será o ReLU, caso contrário, será o Tanh (Tangente Hiperbólica).
    '''
    activation = None
    if params['activation'] >= threshold:
        activation = 'relu'
    else:
        activation = 'tanh'

    embedding_matrix = pkl.load(open('artifacts/embedding_matrix.pkl', 'rb'))

    # Criar modelos com base nos parâmetros
    inputs = keras.Input(shape=(maxlen,), dtype='int32')
    embedded = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=True
    )(inputs)
    x = layers.Bidirectional(layers.LSTM(params['neurons']))(embedded)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(params['neurons']*2, activation=activation)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(params['neurons']*2, activation=activation)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation=activation)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=optimization, loss='mse', metrics='mae')

    model.fit(X_train_pad, y_train, epochs=params['epochs'], validation_split=valid_size, batch_size=params['batch_size'])
    
    # Avaliar modelo
    mae = model.evaluate(X_test_pad, y_test, verbose=1)[0]
    return mae

def main():
    start, end = 1, 20
    params = {
        'epochs': random.randint(start, end),
        'batch_size': random.randint(start, end),
        'neurons': random.randint(start, end),
        'activation': random.randint(start, end),
        'optimization': random.randint(start, end),
    }

    print('Neural network hyperparameters:', params)

    mae = run_model(
        params,
        end // 2,
        random=42
    )

    print('Model MAE: ', mae)


if __name__ == '__main__':
    main()

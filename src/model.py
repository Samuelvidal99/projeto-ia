from __future__ import annotations

import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def run_model(params: dict, threshold: int, data: tuple[str, str], random: int) -> float:
    # Configurar seeds
    np.random.seed(random)
    torch.manual_seed(random)

    # Configurar hiperparâmetros estáticos
    lr = 0.001
    weight_decay = 0.0001
    criterion = nn.BCELoss()
    test_size = 0.25

    # Carregar dados
    features = pd.read_csv(data[0])
    labels = pd.read_csv(data[1])

    features = np.array(features, dtype='float32')
    labels = np.array(labels, dtype='float32')

    # Separar os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size
    )

    # Transformar arrays em tensores e instanciar o train loader
    X_train_tensors = torch.tensor(np.array(X_train), dtype=torch.float)
    y_train_tensors = torch.tensor(np.array(y_train), dtype=torch.float)
    X_test_tensors = torch.tensor(np.array(X_test), dtype=torch.float)
    y_test_tensors = torch.tensor(np.array(y_test), dtype=torch.float)

    dataset = torch.utils.data.TensorDataset(
        X_train_tensors,
        y_train_tensors,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True
    )

    '''
    Para selecionarmos o método de otimização usamos o parâmetro limiar e
    optimização. Se a optimização ultrapassar o limiar, o método selecionado será
    o Adam, caso contrário, será o SGD.
    '''
    optimization = None
    if params['optimization'] >= threshold:
        optimization = torch.optim.Adam
    else:
        optimization = torch.optim.SGD

    '''
    Para selecionarmos a função de ativação usamos o parâmetro limiar e
    ativação. Se o valor de ativação ultrapassar o limiar, a função selecionada
    será o ReLU, caso contrário, será o Tanh (Tangente Hiperbólica).
    '''
    activation = None
    if params['activation'] >= threshold:
        activation = nn.ReLU
    else:
        activation = nn.Tanh

    # Criar modelos com base nos parâmetros
    model = nn.Sequential(
        nn.Linear(30, params['neurons']),
        activation(),
        nn.Linear(params['neurons'], params['neurons']),
        activation(),
        nn.Linear(params['neurons'], 1),
        nn.Sigmoid()
    )

    # Instanciar optimizador
    optimizer = optimization(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Treinar modelo
    for epoch in range(params['epochs']):
        running_loss = 0.

        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loss = running_loss / len(train_loader)
        
        print(f'Epoch {epoch + 1}: loss {loss:.5f}')

    # Avaliar modelo
    model.eval()
    predictions = model.forward(X_test_tensors)
    predictions = np.array(predictions > 0.5)
    accuracy = accuracy_score(y_test_tensors, predictions)
    return accuracy

def main():
    start, end = 1, 20
    params = {
        'epochs': random.randint(start, end),
        'batch_size': random.randint(start, end),
        'neurons': random.randint(start, end),
        'activation': random.randint(start, end),
        'optimization': random.randint(start, end),
    }
    acc = run_model(
        params,
        end // 2,
        ('data/breast_in.csv',
        'data/breast_out.csv'),
        random=42
    )

    print('Model accuracy: ', acc)


if __name__ == '__main__':
    main()

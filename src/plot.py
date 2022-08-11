import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv('results/best_score.csv', index_col=False)
ax = x.plot(
    ylabel='Erro Quadrático Médio',
    xlabel='Gerações',
    figsize=(10,7),
)
ax.legend([
    '10 Individuos',
    '15 Individuos',
    '20 Individuos',
    '25 Individuos',
    '30 Individuos'
])
ax.get_figure().dpi = 150
ax.get_figure().savefig('plot.pdf')
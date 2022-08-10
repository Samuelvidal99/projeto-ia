from concurrent.futures import ProcessPoolExecutor
import numpy as np
import spacy
from model import run_model


MIN_BOUND = 1
MAX_BOUND = 20


def initialize(population_size: int, n_features: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    shape = (population_size, n_features)
    result = np.random.randint(1, 20, shape)
    np.random.shuffle(result)
    return result


def score(chromosome: np.ndarray) -> float:
    param_keys = ["epochs", "batch_size", "neurons", "activation", "optimization"]
    params = dict(zip(param_keys, chromosome))
    nlp = spacy.load("pt_core_news_sm")
    mse = run_model(
        nlp,
        params,
        threshold=10,
        data=("data/essay_in.csv", "data/essay_out.csv"),
        random=42,
    )
    return mse

def fitness_score(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    population: after initialization
    """

    population_list = map(lambda arr: list(map(int, arr)), population)
    with ProcessPoolExecutor(4) as executor:
        scores = list(executor.map(score, population_list))
    scores = np.array(scores)

    # obtendo indices que ordenam pelo maior score
    inds = np.argsort(scores) # crescente
    # inds = np.argsort(scores)[::-1] # decresente

    ordered_scores = scores[inds]
    ordered_population = population[inds, :] 

    return ordered_scores, ordered_population


def selection(population: np.ndarray, n_parents: int | None = None) -> np.ndarray:
    """
    population: after score ordering
    """
    # por padrão, metade da população passará a próxima etapa
    if n_parents is None:
        n_chromo = population.shape[0]
        n_parents = n_chromo// 2
    return population[:n_parents]


def crossover(population: np.ndarray, div_index: int | None = None) -> np.ndarray:
    """
    population: after selection
    """
    n_features = population.shape[1]
    if div_index is None:
        div_index =  n_features // 2

    pop_nextgen = list(population)

    # itera de 2 em 2
    for i in range(0, len(population), 2):
        # obtém os pais da criança
        chromo1, chromo2 = pop_nextgen[i], pop_nextgen[i + 1]

        # passa metade de cada cromosomo para a próxima geração
        new_par = np.concatenate((chromo1[: div_index], chromo2[div_index :]))

        # adiciona a criança mantendo os pais
        pop_nextgen.append(new_par)

    return np.array(pop_nextgen)

def mutation(population: np.ndarray, mutation_rate) -> np.ndarray:
    """
    population: after crossover
    """
    n_feat = population.shape[1]

    mutation_range = int(mutation_rate * n_feat)

    next_population = []

    for chromo in population:

        # Calculando quais genes do cromosomo serão mutados
        indexes = np.random.randint(0, n_feat -1, size=mutation_range)
        
        # Realizando Mutação no cromosomo
        for i in indexes:
            chromo[i] = np.clip(chromo[i] + np.random.randint(-2, 2), MIN_BOUND, MAX_BOUND)
        next_population.append(chromo)

    return np.array(next_population)


def main(
    n_generations: int = 10,
    population_size: int = 20,
    n_features: int = 5 # -> epochs, batch_size, neurons, activation, optimization
):
    # no inicio nada existia, então eu criei o mundo...
    population = initialize(population_size, n_features)

    score, population = fitness_score(population)
    best_score = score[0]
    best_chromo = population[0]

    # planejei o fim do mundo e quantas gerações devem existir...
    for i in range(n_generations):

        # elevei meu fiel preferido diante dos outros...
        print('Melhor Pontuação:', score[0], 'feito por:', population[0])

        # matei os que não mereciam viver e preservei os mais fieis...
        population = selection(population)

        # arrangei os casamentos e deixei que tivessem filhos...
        population = crossover(population)

        # mudei a aparencia dos mais feios, segundo a minha vontade...
        population = mutation(population)

        # então eu julguei todos perante a minha vontade...
        score, population = fitness_score(population)
        if score[0] > best_score:
            best_score = score[0]
            best_chromo = population[0]
    
    print('Melhor Pontuação:', best_score, 'feito por:', best_chromo)
    return best_score, best_chromo



if __name__ == "__main__":
    main()


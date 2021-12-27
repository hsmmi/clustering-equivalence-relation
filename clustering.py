import numpy as np
from my_io import read_dataset_to_X_and_y, read_dataset_with_pandas
from copy import deepcopy
import matplotlib.pyplot as plt


class UniSet():
    def __init__(self, file, range_feature, range_label,
                 normalization=None, shuffle=False, about_nan='class_mean'):
        np.random.seed(1)
        sample, label = read_dataset_to_X_and_y(
            file, range_feature, range_label, normalization, shuffle=shuffle,
            about_nan=about_nan)
        self.number_of_feature = sample.shape[1]
        self.size_of_universal = sample.shape[0]
        self.universal = sample.astype(float)
        self.label = label
        self.diffrent_label = np.unique(label)
        self.number_of_diffrent_label = self.diffrent_label.shape[0]
        self.relation = None
        self.equivalence_relation = None


uni_total = UniSet(
    'dataset/hcvdat0.csv', (2, 14), (1, 2),
    normalization='z_score', shuffle=True, about_nan='class_mean')


def split_train_test(universe: UniSet, train_size: float) -> list[UniSet]:
    train = deepcopy(universe)
    test = deepcopy(universe)
    train.size_of_universal = \
        int(universe.size_of_universal*train_size)
    train.universal = \
        universe.universal[0:train.size_of_universal]
    train.label = \
        universe.label[0:train.size_of_universal]
    test.size_of_universal = (
        universe.size_of_universal - train.size_of_universal)
    test.universal = \
        universe.universal[train.size_of_universal:]
    test.label = \
        universe.label[train.size_of_universal:]

    return train, test


uni_train, uni_test = split_train_test(uni_total, 0.8)


def distance(sample1: np.ndarray, sample2: np.ndarray) -> float:
    return np.linalg.norm(sample1-sample2)


def find_relation(universal: UniSet) -> np.ndarray:
    dis = np.array(
        list(map(lambda x: list(map(
            lambda y: distance(
                universal.universal[x], universal.universal[y]),
            range(universal.size_of_universal))),
            range(universal.size_of_universal))))
    return 1 - dis / np.max(dis)


uni_train.relation = find_relation(uni_train)


def max_min(sample1: np.ndarray, sample2: np.ndarray) -> float:
    both_sample = np.vstack((sample1, sample2))
    return np.max(np.min(both_sample, axis=0))


def composition_RoR(relation: np.ndarray) -> np.ndarray:
    result = np.array(
        list(map(lambda x: list(map(
            lambda y: max_min(relation[x], relation[y]),
            range(relation.shape[0]))),
            range(relation.shape[0]))))
    return result


def union_two_relation(
        relation1: np.ndarray, relation2: np.ndarray) -> np.ndarray:
    both_relation = np.dstack((relation1, relation2))
    return np.max(both_relation, axis=2)


def make_transitive(relation: np.ndarray) -> np.ndarray:
    R = None
    Rp = np.copy(relation)
    iter = 0
    while((Rp != R).any()):
        R = np.copy(Rp)
        RoR = composition_RoR(R)
        Rp = union_two_relation(R, RoR)
        iter += 1
        print(iter)
    return Rp


# uni_train.equivalence_relation = make_transitive(uni_train.relation)
ER = read_dataset_with_pandas('dataset/ER.train.csv')[1]
ER = ER.to_numpy()[:, 1:]
uni_train.equivalence_relation = ER


def is_reflexive(relation: np.ndarray) -> bool:
    return (relation.diagonal() != 0).all()


def is_symmetric(relation: np.ndarray) -> bool:
    return np.array_equal(relation, relation.T)


def is_transitive(relation: np.ndarray) -> bool:
    RoR = composition_RoR(relation)
    Rp = union_two_relation(relation, RoR)
    return (Rp == relation).all()


def is_equivalece(relation: np.ndarray) -> bool:
    return is_reflexive(relation) & is_symmetric(relation) & \
        is_transitive(relation)


# print(is_equivalece(uni_train.equivalence_relation))


def find_similarity_class(
        universal: UniSet, target_sample: int, alpha: float) -> np.ndarray:
    size_of_universal = universal.shape[0]
    similarity_class = []
    for sample in range(size_of_universal):
        if(universal[sample, target_sample] >= alpha):
            similarity_class.append(sample)
    return np.array(similarity_class)


def find_cluster(relation: np.ndarray, alpha: float, label=True):
    size_of_universal = relation.shape[0]
    classes = []
    predicted_label = np.full((size_of_universal, 1), -1.0)
    number_of_class = 0.0
    for sample in range(size_of_universal):
        if(predicted_label[sample] == -1):
            new_class = find_similarity_class(relation, sample, alpha)
            predicted_label[new_class] = number_of_class
            number_of_class += 1
            classes.append(new_class)
    number_of_class = int(number_of_class)
    if(label is True):
        return predicted_label, number_of_class
    return classes, number_of_class


def evaluate(gold_label: np.ndarray, predict_label: np.ndarray,
             method: str = 'f1-score') -> float:
    diffrent_label_in_gold_label = np.unique(gold_label)
    diffrent_label_in_predict_label = np.unique(predict_label)
    conf_mat = np.array(
        list(map(lambda k: list(map(
            lambda s: sum((predict_label == k)*(gold_label == s))[0],
            diffrent_label_in_gold_label)),
            diffrent_label_in_predict_label)))
    precision = np.sum(np.max(conf_mat, axis=1)) / np.sum(conf_mat)
    recall = np.sum(np.max(conf_mat, axis=0)) / np.sum(conf_mat)
    if(method == 'precision'):
        return precision
    if(method == 'recall'):
        return recall
    if(method == 'f1-score'):
        return 2 * ((precision*recall)/(precision+recall))


def find_best_alpha_cut(universal: UniSet, plotter: bool = False) -> float:
    alpha_cut = []
    accuracy = []
    last_point = -1.0
    for alpha in np.unique(ER):
        if(alpha - last_point >= 0.001):
            alpha_cut.append(alpha)
            accuracy.append(evaluate(
                universal.label,
                find_cluster(universal.equivalence_relation, alpha, True)[0]))
            last_point = alpha
    if(plotter is True):
        plt.plot(alpha_cut, accuracy)
        plt.show()


best_alpha_cut, best_alpha_cut_accuracy = find_best_alpha_cut(uni_train)

predicted_label_test = find_cluster(
    uni_test.equivalence_relation, best_alpha_cut)
test_accuracy = evaluate(uni_test.label, predicted_label_test)

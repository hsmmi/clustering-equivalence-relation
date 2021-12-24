import numpy as np
from my_io import read_dataset_to_X_and_y
from normalization import zero_mean_unit_variance
from copy import deepcopy


class UniSet():
    def __init__(self, file, range_feature, range_label, normalization=False):
        np.random.seed(1)
        sample, label = read_dataset_to_X_and_y(
            file, range_feature, range_label, shuffle=False,
            about_nan='class_mean')
        self.number_of_feature = sample.shape[1]
        self.size_of_universal = sample.shape[0]
        self.universal = sample.astype(float)
        self.label = label
        self.diffrent_label = np.unique(label)
        self.number_of_diffrent_label = self.diffrent_label.shape[0]
        if normalization is True:
            self.universal = zero_mean_unit_variance(self.universal)
        self.relation = None
        self.equivalence_relation = None


uni_total = UniSet('dataset/hcvdat0.csv', (4, 14), (1, 2), normalization=True)


def split_train_test(universe: UniSet, train_size: float) -> list[UniSet]:
    train = deepcopy(universe)
    test = deepcopy(universe)
    train.size_of_universal = \
        int(universe.size_of_universal*train_size)
    train.universal = \
        universe.universal[0:train.size_of_universal]
    test.size_of_universal = (
        universe.size_of_universal - train.size_of_universal)
    test.universal = \
        universe.universal[train.size_of_universal:]
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


uni_train.equivalence_relation = make_transitive(uni_train.relation)


def is_reflexive(relation: np.ndarray) -> bool:
    return (relation.diagonal() != 0).all()


def is_symmetric(relation: np.ndarray) -> bool:
    return (relation == relation.T).all()


def is_transitive(relation: np.ndarray) -> bool:
    RoR = composition_RoR(relation)
    Rp = union_two_relation(relation, RoR)
    return (Rp == relation).all()


def is_equivalece(relation: np.ndarray) -> bool:
    return is_reflexive(relation) & is_symmetric(relation) & \
        is_transitive(relation)


print(is_equivalece(uni_train.equivalence_relation))


def find_similarity_class(
        universal: UniSet, target_sample: int, alpha: float) -> np.ndarray:
    size_of_universal = universal.shape[0]
    similarity_class = []
    for sample in range(size_of_universal):
        if(universal[sample, target_sample] >= alpha):
            similarity_class.append(sample)
    return np.array(similarity_class)


def find_cluster(relation: np.ndarray, alpha: float) -> list:
    size_of_universal = relation.shape[0]
    classes = []
    mark = np.zeros(size_of_universal)
    for sample in range(size_of_universal):
        if(mark[sample] == 0):
            new_class = find_similarity_class(relation, sample, alpha)
            mark[new_class] = 1
            classes.append(new_class)
    return classes

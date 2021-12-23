import numpy as np
from my_io import read_dataset_to_X_and_y
from normalization import range_min_to_max


class Relation():
    def __init__(self, file, range_feature, range_label):
        np.random.seed(1)
        sample, label = read_dataset_to_X_and_y(
            file, range_feature, range_label, shuffle=True)
        self.number_of_feature = sample.shape[1]
        self.size_of_universal = sample.shape[0]
        self.universal = sample.astype(float)
        self.label = label
        self.diffrent_label = np.unique(label)
        self.number_of_diffrent_label = self.diffrent_label.shape[0]
        self.size_of_train_universal = None
        self.train_universal = None
        self.size_of_test_universal = None
        self.test_universal = None
        self.train_relation = None
        self.train_equivalence_relation = None

    def change_missing_value_with_class_mean(self):
        for label in self.diffrent_label:
            class_label = self.universal[(self.label == label).flatten()]
            for feature in range(self.number_of_feature):
                mean_feature_label = np.nanmean(class_label[:, feature])
                for sample in range(self.size_of_universal):
                    if np.isnan(self.universal[sample, feature]):
                        self.universal[sample, feature] = mean_feature_label

    def split_train_test(self, train_size):
        self.universal = range_min_to_max(self.universal, 0.1, 1)
        self.size_of_train_universal = int(self.size_of_universal*train_size)
        self.train_universal = self.universal[0:self.size_of_train_universal]
        self.size_of_test_universal = (
            self.size_of_universal - self.size_of_train_universal)
        self.test_universal = self.universal[self.size_of_train_universal:]

    def similarity(self, sample1, sample2):
        score = 0.0
        sample1 = self.train_universal[sample1]
        sample2 = self.train_universal[sample2]
        for feature in range(self.number_of_feature):
            score += min(
                (sample1[feature] / sample2[feature]),
                (sample2[feature] / sample1[feature]))
        score /= self.number_of_feature
        return score

    def find_relation(self):
        self.train_relation = np.array(
            list(map(lambda x: list(map(lambda y: self.similarity(x, y),
                 range(self.size_of_train_universal))),
                 range(self.size_of_train_universal))))

    def max_min(self, sample1, sample2):
        both_sample = np.vstack((sample1, sample2))
        return np.max(np.min(both_sample, axis=0))

    def composition_RoR(self, relation):
        result = np.array(
            list(map(lambda x: list(map(
                lambda y: self.max_min(relation[x], relation[y]),
                range(relation.shape[0]))),
                range(relation.shape[0]))))
        return result

    def union_two_relation(self, relation1, relation2):
        both_relation = np.dstack((relation1, relation2))
        return np.max(both_relation, axis=2)

    def make_transitive(self, relation):
        R = None
        Rp = np.copy(relation)
        iter = 0
        while((Rp != R).any()):
            R = np.copy(Rp)
            RoR = self.composition_RoR(R)
            Rp = self.union_two_relation(R, RoR)
            iter += 1
            print(iter)
        return Rp

    def is_reflexive(self, relation):
        return (relation.diagonal() != 0).all()

    def is_symmetric(self, relation):
        return (relation == relation.T).all()

    def is_transitive(self, relation):
        RoR = self.composition_RoR(relation)
        Rp = self.union_two_relation(relation, RoR)
        return (Rp == relation).all()

    def is_equivalece(self, relation):
        is_reflexive = self.is_reflexive(relation)
        is_symmetric = self.is_symmetric(relation)
        is_transitive = self.is_transitive(relation)
        return is_reflexive & is_symmetric & is_transitive

    def find_similarity_class(self, relation, target_sample, alpha):
        size_of_relation = relation.shape[0]
        similarity_class = []
        for sample in range(size_of_relation):
            if(relation[sample, target_sample] >= alpha):
                similarity_class.append(sample)
        return np.array(similarity_class)

    def find_cluster(self, relation, alpha):
        size_of_relation = relation.shape[0]
        classes = []
        mark = np.zero(size_of_relation)
        for sample in range(size_of_relation):
            if(mark[sample] == 0):
                new_class = self.find_similarity_class(relation, sample, alpha)
                mark[new_class] = 1
                classes.append(new_class)

    def relation_runner(self):
        self.change_missing_value_with_class_mean()
        self.split_train_test(0.8)
        self.find_relation()
        self.is_equivalece(self.train_relation)
        self.train_equivalence_relation = \
            self.make_transitive(self.train_relation)
        print(self.is_transitive(self.train_equivalence_relation))
        print('hi')

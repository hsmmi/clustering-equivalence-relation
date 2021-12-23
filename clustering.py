import numpy as np
from my_io import read_dataset_to_X_and_y
from normalization import range_min_to_max


class Relation():
    def __init__(self, file, range_feature, range_label):
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
        for feature in range(self.number_of_feature):
            score += min(
                (self.train_universal[sample1, feature] /
                 self.train_universal[sample2, feature]),
                (self.train_universal[sample2, feature] /
                 self.train_universal[sample1, feature]))
        score /= self.number_of_feature
        return score

    def find_relation(self):
        self.train_relation = np.array(
            list(map(lambda x: list(map(lambda y: self.similarity(x, y),
                 range(self.size_of_train_universal))),
                 range(self.size_of_train_universal))))

    def composition_RoR(self, relation):

        print('hi')

    def make_transitive(self):
        print('hi')

    def relation_runner(self):
        self.change_missing_value_with_class_mean()
        self.split_train_test(0.8)
        self.find_relation()
        self.make_transitive()

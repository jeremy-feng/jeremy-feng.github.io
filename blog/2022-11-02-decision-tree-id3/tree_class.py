import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature_index=None, children=None, info_gain=None, value=None):
        """constructor"""

        # for decision node
        self.feature_index = feature_index
        self.children = children
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        """constructor"""

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        """recursive function to build the tree"""

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best feature
            best_feature = self.get_best_feature(dataset, num_features)
            # check if information gain is positive
            if best_feature["info_gain"] > 0:
                children = []
                for child in best_feature["children"]:
                    # 对每一个 child，迭代调用 build_tree 函数
                    children.append(
                        (child[0], self.build_tree(child[1], curr_depth + 1))
                    )
                return Node(
                    best_feature["feature_index"], children, best_feature["info_gain"]
                )

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_feature(self, dataset, num_features):
        """function to find the best split"""

        # dictionary to store the best split
        best_feature = {}
        max_info_gain = -float("inf")

        # 求数据集的标签
        y = dataset[:, -1]
        # 求数据集的样本数
        D = y.shape[0]
        # 求数据集的经验熵
        H_D = self.entropy(y)

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            # 获取该特征的所有去重值
            feature_values = np.unique(feature_values)
            # 计算该特征的经验条件熵
            H_D_A = 0
            for i in feature_values:
                # 求该特征的值等于 i 时的标签
                y_i = dataset[dataset[:, feature_index] == i][:, -1]
                # 求该特征的值等于 i 时的经验熵
                H_D_i = self.entropy(y_i)
                # 求该特征的值等于 i 的比例
                weight_i = y_i.shape[0] / D
                # 求该特征的经验条件熵
                H_D_A += weight_i * H_D_i
            # 求该特征的信息增益
            curr_info_gain = H_D - H_D_A
            # update the best split if needed
            if curr_info_gain > max_info_gain:
                best_feature["feature_index"] = feature_index
                best_feature["children"] = [
                    (i, dataset[dataset[:, feature_index] == i]) for i in feature_values
                ]
                best_feature["info_gain"] = curr_info_gain
                max_info_gain = curr_info_gain

        return best_feature

    def entropy(self, y):
        """function to compute entropy"""

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def calculate_leaf_value(self, Y):
        """function to compute leaf node"""

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent="    "):
        """function to print the tree"""
        # 如果没有传入 tree 这个参数，则默认需要打印 self.root 这棵树
        if not tree:
            tree = self.root
        # 如果是叶子节点，则打印叶子节点的值
        if tree.value is not None:
            print("<{}>".format(tree.value))
        # 如果是决策节点，则打印决策节点的信息，包括特征名称和子节点
        else:
            # print('\n')
            print("_{}_".format(str(columns[tree.feature_index])))

            for i in tree.children:
                print("{}{}:".format(indent, i[0]), end="")
                self.print_tree(i[1], indent + indent)

    def fit(self, X, Y):
        """function to train the tree"""

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """function to predict new dataset"""

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        """function to predict a single data point"""

        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        child_index = [
            idx for idx, tup in enumerate(tree.children) if tup[0] == feature_val
        ][0]
        return self.make_prediction(x, tree.children[child_index][1])

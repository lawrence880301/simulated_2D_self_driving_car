import random
import numpy as np
class Datapreprocessor():
    def __init__(self) -> None:
        pass

    def readfile(url):
        read = open(url)
        file = read.readlines()
        read.close()
        return file
    def text_to_numlist(dataset):
        """load text dataset to numeracial list dataset

        Args:
            dataset (string): txt or other file

        Returns:
            dataset: float_list
        """
        dataset = [list(map(float,data)) for data in dataset]
        return dataset
    def train_test_split(data,split_ratio):
        """shuffle data and spilt it into train and test data

        Args:
            data (list): numeracial list
            split_ratio (float): portion of train

        Returns:
            train_data, test_data: float_list
        """
        sample_position = round(len(data)*(split_ratio))
        random.shuffle(data)
        train_data, test_data = data[:sample_position], data[sample_position:]
        return train_data, test_data

    def feature_label_split(dataset):
        feature = []
        label = []
        for row in dataset:
            feature.append(row[:-1])
            label.append(row[-1])
        return feature, label

    def normalize_2d_np(array):
        row_sums = array.sum(axis=1)
        array = array / row_sums[:, np.newaxis]
        return array

    def normalize_1d_np(array):
        array = (array - np.amin(array)) / (np.amax(array) - np.amin(array))
        return array

    
    def label_preprocess(dataset):
        """label origin label to new label, which start from 0

        Args:
            dataset ([list]): dataset to be processed

        Returns:
            [list]: processed dataset
        """
        existing_label = {}
        new_max_label = 0
        for row in dataset:
            if str(row[-1]) not in existing_label:
                existing_label[str(row[-1])] = new_max_label
                row[-1] = existing_label[str(row[-1])]
                new_max_label+=1
            else:
                row[-1] = existing_label[str(row[-1])]
        return dataset

    def group_dataset_by_label(dataset):
        label_0 = []
        label_1 = []
        for row in dataset:
            if row[-1] == 0:
                label_0.append(row)
            else:
                label_1.append(row)
        return label_0, label_1



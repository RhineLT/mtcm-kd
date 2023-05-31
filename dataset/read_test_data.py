import os

__all__ = ["read_test_data"]

def read_test_data(dataset_dir):
    """
    parameters:
        dataset_dir: the directory of the dataset
        
    return:
        data: a list of dictionary, each dictionary contains the information of the dataset
    """
    data_samples = os.listdir(dataset_dir)
    data = []
    data.append({"test_samples": data_samples,})
    return data   
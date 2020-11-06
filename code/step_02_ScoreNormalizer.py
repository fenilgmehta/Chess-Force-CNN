import copy
import numpy as np

import common_services as cs
import tensorflow.keras as keras


class ScoreNormalizer:
    """
    All static methods under this class should follow the naming convention REGEX:
        normalize_[0-9]{3}

    Everything after underscore ("_") will be used in naming/prefixing/suffixing generated files
    """

    @staticmethod
    def normalize_000(data_np: np.ndarray) -> np.ndarray:
        """
        Does NOTHING
        :param data_np:
        :return: Normalized `np.ndarray`
        """
        return copy.deepcopy(data_np)

    @staticmethod
    def normalize_004(data_np: np.ndarray) -> np.ndarray:
        """
        Following is the cp_scores normalization details:
            • (   -inf,     0 ) -> [   -1,   -1 ]
            • [      0,     0 ] -> [    0,    0 ]
            • (      0,   inf ) -> [    1,    1 ]

        :param data_np:
        :return: Normalized `np.ndarray`
        """
        data_np_new: np.ndarray = copy.deepcopy(data_np)
        data_np_new[data_np < 0] = -1.0
        data_np_new[data_np == 0] = 0.0
        data_np_new[data_np > 0] = 1.0
        return data_np_new

    @staticmethod
    def normalize_005(data_np: np.ndarray) -> np.ndarray:
        np_category = np.zeros((data_np.shape[0], 2))
        np_category[(data_np == -1).ravel(), 0] = 1
        np_category[(data_np != -1).ravel(), 1] = 1
        return np_category
        # return keras.utils.to_categorical(data_np, 2)

    @staticmethod
    def str_to_method(n_str):
        for i in cs.get_class_common_prefixed(ScoreNormalizer, prefix_to_search="normalize_"):
            if n_str == ScoreNormalizer.get_suffix_str(i):
                return eval("ScoreNormalizer." + i)
        raise Exception('Invalid value of "n_str"')

    @staticmethod
    def num_to_method(n: int):
        return ScoreNormalizer.str_to_method(f"{n:03}")

    @staticmethod
    def get_suffix_str(normalize_method):
        if type(normalize_method) is not str:
            nm_name = normalize_method.__name__
        else:
            nm_name = normalize_method
        return nm_name[nm_name.rindex("_") + 1:]

    @staticmethod
    def get_all_SN_suffix_str():
        return [
            ScoreNormalizer.get_suffix_str(i)
            for i in cs.get_class_common_prefixed(ScoreNormalizer, prefix_to_search="normalize_")
        ]

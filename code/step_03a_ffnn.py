import multiprocessing
import os
import warnings
from pathlib import Path
from typing import Union, Tuple, List, Callable, Type, Dict

import chess
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

# WARNING/ERROR: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# EXPLANATION: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# SOLUTION 1: https://stackoverflow.com/questions/51681727/tensorflow-on-macos-your-cpu-supports-instructions-that-this-tensorflow-binary?rq=1
# SOLUTION 2: manually compile and install tensorflow 2.0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TENSORFLOW 2.0 installation with GPU support
# Not tested SOLUTION: https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#Anaconda_main_win
# conda install tensorflow -c anaconda
# TESTED SOLUTION: https://anaconda.org/anaconda/tensorflow-gpu
# conda install -c anaconda tensorflow-gpu

# WARNING/ERROR: numpy FutureWarning
# SOLUTION: https://github.com/tensorflow/tensorflow/issues/30427
# import tensorflow as tf
# from tensorflow.python import keras
import tensorflow.keras as keras

import common_services as cs
import step_02_preprocess as step_02
import step_02_ScoreNormalizer as step_02_SN
import step_02_BoardEncoder as step_02_BE


########################################################################################################################

class ModelVersion:
    def __init__(self,
                 prefix: str,
                 model_generator: int,
                 board_encoder: int,
                 score_normalizer: int,
                 epochs: int,
                 weight_or_model: str,
                 version: int,
                 file_extension="h5"):
        self.prefix = prefix
        self.model_generator = model_generator
        self.board_encoder = board_encoder
        self.score_normalizer = score_normalizer
        self.epochs = epochs
        self.weight_or_model = weight_or_model
        self.version = version
        self.file_extension = file_extension

    def __str__(self):
        return ModelVersion.model_name(self.prefix,
                                       self.model_generator, self.board_encoder, self.score_normalizer, self.epochs,
                                       self.weight_or_model, self.version, self.file_extension)

    @staticmethod
    def create_obj(file_name: str):
        name, extension = str(Path(file_name).name).split('.')
        arr = name.split('-')
        if len(arr) != 7: raise Exception("Invalid ModelVersion str name")
        return ModelVersion(arr[0], int(arr[1][2:]), int(arr[2][2:]), int(arr[3][2:]), int(arr[4][2:]), arr[5], int(arr[6][1:]))

    @staticmethod
    def model_name(prefix: str, model_generator, board_encoder, score_normalizer, epochs, weight_or_model, version: int,
                   file_extension="h5"):
        # return f"{prefix}-v{version:03}-mg{model_generator:03}-be{board_encoder:03}-"
        return "-".join([
            f"{prefix}",
            f"mg{model_generator:03d}",
            f"be{board_encoder:05d}",
            f"sn{score_normalizer:03d}",
            f"ep{epochs:05d}",
            f"{weight_or_model}",
            f"v{version:03d}",
        ]) + f".{file_extension}"


########################################################################################################################
# NOTE: `c` before each method name means that it is custom
# Neural Network - Keras
class NNKeras:
    def __init__(self, model_generator: 'KerasModels.__call__',
                 board_encoder: Type[step_02.BoardEncoder.EncodeBase],
                 score_normalizer: Callable[[np.ndarray], np.ndarray],
                 model_version: ModelVersion,
                 model_save_path: str = "../../Chess-Force-Models/",
                 callback=False,
                 generate_model_image=False):
        """
        NOTE: "board_encoder" is used only then making predictions on new data, NOT while training/fitting
        NOTE: "score_normalizer" is NEVER used

        :param model_generator:
        :param board_encoder:
        :param score_normalizer:
        :param model_version:
        :param model_save_path:
        :param callback:
        :param generate_model_image:
        """
        self.model: keras.Sequential = model_generator(board_encoder.INPUT_DIMENSIONS)
        self.board_encoder: Type[step_02.BoardEncoder.EncodeBase] = board_encoder
        self.score_normalizer = score_normalizer
        self.model_version: ModelVersion = model_version
        self.model_save_path: str = model_save_path
        # self.model_save_path_dir = str(Path(model_save_path).parent)

        # CREATE a callback that saves the model's weights
        if callback:
            self.cp_callback = [
                keras.callbacks.ModelCheckpoint(
                    filepath=str(
                        Path(self.model_save_path) / (self.generate_name()[:-3] + "_ep{epoch:05d}-vl{val_loss:.5f}.h5")
                    ),
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1)
            ]
        else:
            self.cp_callback = None

        # PRINT model summary
        # self.model.summary()

        # SAVE the model graph
        if generate_model_image:
            image_name_prefix = f"ffnn_keras-{self.generate_name()}"
            for i in range(1, 100):
                image_name = f'{image_name_prefix}_{i:03}.png'
                if not Path(image_name).exists():
                    keras.utils.plot_model(self.model, image_name, show_shapes=True)
                    print(f"Saving the image: '{image_name}'")
                    break
        return

    def c_save_weights(self, model_name: str = None, model_path: Union[str, Path] = None, write_name: str = 'z_last_model_weight_name.txt'):
        if model_path is None:
            model_path = self.model_save_path
        if model_name is None:
            model_name = self.generate_name()

        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(Path(model_path) / model_name), overwrite=True)

        try:
            dict_file = f'{Path(model_path) / write_name}'
            if os.path.exists(dict_file):
                dict_prefix: Dict = eval(open(f'{Path(model_path) / write_name}', 'r').read().strip())
                dict_prefix.update({self.model_version.prefix: model_name})
            else:
                dict_prefix = {self.model_version.prefix: model_name}
            open(f'{model_path}/{write_name}', "w+").write(str(dict_prefix))
        except:
            pass

        print(f"Path: {model_path}")
        print(f"Model weights successfully saved: {model_name}")
        return

    def c_load_weights(self, model_name: str, model_path: Union[str, Path] = None):
        if model_path is None:
            model_path = self.model_save_path
        if not (Path(model_path) / model_name).exists():
            print(f"ERROR: model does not exists: {Path(model_path) / model_name}")
            raise FileNotFoundError(f"'{Path(model_path) / model_name}'")
        self.model.load_weights(str(Path(model_path) / model_name))
        self.model_version: ModelVersion = ModelVersion.create_obj(model_name)
        print(f"Model weights successfully loaded: {model_name}")
        return

    def c_train_model(self, x_input: np.ndarray, y_output: np.ndarray, epochs: int, batch_size: int,
                      validation_split: float):
        self.update_version()
        if self.cp_callback is not None:
            self.cp_callback[0].filepath = \
                str(Path(self.model_save_path) / (self.generate_name()[:-3] + "_ep{epoch:05d}-vl{val_loss:.5f}.h5"))

        self.model.trainable = True
        # compile the keras model
        # self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # mae = Mean Absolute Error
        # with tf.device('/gpu:0'):
        self.model.fit(x_input, y_output, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                       verbose=2, workers=multiprocessing.cpu_count(), use_multiprocessing=True,
                       callbacks=self.cp_callback)

        self.model.trainable = False
        return

    def c_evaluate_model(self, x_input_test: np.ndarray, y_output_test: np.ndarray, verbose=2):
        loss, mae = self.model.evaluate(x_input_test, y_output_test, verbose=verbose)
        print(f"Evaluated: Loss = {loss:5.3f}")
        print(f"Evaluated: MAE = {mae:5.3f}")

    def c_predict(self, encoded_board: np.ndarray, verbose=0) -> np.ndarray:
        """
        Takes a 2D np.ndarray where each row is a chess board made up of the
        floating point numbers using methods of class `self.board_encoder`
        :param encoded_board:
        :param verbose:
        :return: np.ndarray - 1D
        """
        return self.model.predict(
            encoded_board,
            verbose=verbose, workers=multiprocessing.cpu_count(), use_multiprocessing=True
        )

    def c_predict_board_1(self, board_1: chess.Board, verbose=0) -> np.float32:
        return self.c_predict(
            np.array([
                self.board_encoder.encode_board_1(board_1)
            ]),
            verbose
        )[0]

    def c_predict_board_n(self, board_n: Union[List[chess.Board], Tuple[chess.Board]], verbose=0) -> np.ndarray:
        return self.c_predict(
            self.board_encoder.encode_board_n(
                board_n
            ),
            verbose
        )

    def c_predict_fen_1(self, board_1_fen: str, verbose=0) -> np.float32:
        return self.c_predict(
            np.array([
                self.board_encoder.encode_board_1_fen(board_1_fen)
            ]),
            verbose
        )[0]

    def c_predict_fen_n(self, board_n_fen: Union[List[str], Tuple[str], np.ndarray], verbose=0) -> np.ndarray:
        return self.c_predict(
            self.board_encoder.encode_board_n(
                [chess.Board(i) for i in board_n_fen]
            ),
            verbose
        )

    def generate_name(self):
        return self.model_version.__str__()

    def update_version(self, number: int = -1):
        if number <= 0:
            self.model_version.version += 1
        else:
            self.model_version.version = number


class KerasModels:

    @staticmethod
    def get_all_model_nums() -> List[str]:
        return [i[i.rindex("_") + 1:] for i in cs.get_class_common_prefixed(KerasModels, prefix_to_search="model_")]

    @staticmethod
    def get_model_from_suffix(model_suffix):
        return eval(f"KerasModels.model_{model_suffix}")

    # optimizer = keras.optimizers.adam(lr=0.001,
    #                                        beta_1=0.9,
    #                                        beta_2=0.999,
    #                                        epsilon=1e-07,
    #                                        amsgrad=False, )
    # __init__(
    #     learning_rate=0.001,  learning_rate: A Tensor or a floating point value. The learning rate.
    #     beta_1=0.9,    beta_1: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
    #     beta_2=0.999,  beta_2: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
    #     epsilon=1e-07, epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma
    #                    and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
    #     amsgrad=False, amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond".
    #     name='Adam',   name: Optional name for the operations created when applying gradients. Defaults to "Adam". @compatibility(eager)
    #                    When eager execution is enabled, learning_rate, beta_1, beta_2, and epsilon can each be a callable that takes no
    #                    arguments and returns the actual value to use. This can be useful for changing these values across different
    #                    invocations of optimizer functions. @end_compatibility
    #     **kwargs       **kwargs: keyword arguments. Allowed to be {clipnorm, clipvalue, lr, decay}. clipnorm is clip gradients by norm;
    #                    clipvalue is clip gradients by value, decay is included for backward compatibility to allow time inverse decay of
    #                    learning rate. lr is included for backward compatibility, recommended to use learning_rate instead.
    # )

    @staticmethod
    def model_006(input_dim) -> keras.Sequential:
        # REFER: https://github.com/bajcmartinez/kickass-chess
        model: keras.Sequential = keras.Sequential()
        model.add(keras.layers.Conv2D(16, kernel_size=1, activation='relu', input_shape=input_dim))
        model.add(keras.layers.Conv2D(16, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=1, activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=1, activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=['accuracy'])

        return model


class NNBuilder:
    @staticmethod
    def build(model_weights_file_name: str, callback=False, generate_model_image=False) -> NNKeras:
        model_version = ModelVersion.create_obj(model_weights_file_name)
        return NNBuilder.build_from_model_version(model_version, callback, generate_model_image)

    @staticmethod
    def build_from_model_version(model_version, callback=False, generate_model_image=False) -> NNKeras:
        try:
            model_generator, board_encoder, score_normalizer = None, None, None
            for i in KerasModels.get_all_model_nums():
                if model_version.model_generator == int(i):
                    model_generator = KerasModels.get_model_from_suffix(i)
            if model_generator is None:
                raise Exception(f"Invalid ModelGenerator={model_version.model_generator}")

            for i in step_02_BE.BoardEncoder.get_all_uniq_name():
                if model_version.board_encoder == int(i):
                    board_encoder = step_02_BE.BoardEncoder.get_uniqstr_to_classobj(i)
            if board_encoder is None:
                raise Exception(f"Invalid BoardEncoder={model_version.board_encoder}")

            for i in step_02_SN.ScoreNormalizer.get_all_SN_suffix_str():
                if model_version.score_normalizer == int(i):
                    score_normalizer = step_02_SN.ScoreNormalizer.str_to_method(i)
            if score_normalizer is None:
                raise Exception(f"Invalid ScoreNormalizer={model_version.score_normalizer}")

            return NNKeras(model_generator,
                           board_encoder,
                           score_normalizer,
                           model_version,
                           callback=callback,
                           generate_model_image=generate_model_image)

        except Exception as e:
            print(f"ERROR: ModelVersion: {e}")

########################################################################################################################

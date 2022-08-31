import argparse
import gc
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import Union, List, Callable

import joblib
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

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
import tensorflow as tf
from tensorflow.python.client import device_lib

import common_services as cs
import step_02_preprocess as step_02
import step_03a_ffnn as step_03a


########################################################################################################################


def train_on_file(keras_obj: step_03a.NNKeras, file_path: str, data_load_transform, y_normalizer, epochs, batch_size, validation_split):
    data_x_encoded, data_y_normalized = data_load_transform(file_path)
    if y_normalizer is not None:
        data_y_normalized = y_normalizer(data_y_normalized)

    keras_obj.c_train_model(x_input=data_x_encoded,
                            y_output=data_y_normalized,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split)
    del data_x_encoded, data_y_normalized


def train_on_folder(keras_obj: step_03a.NNKeras,
                    input_dir: str, move_dir: str, file_suffix: str,
                    data_load_transform, y_normalizer,
                    epochs: int, batch_size: int, validation_split: float):
    Path(move_dir).mkdir(parents=True, exist_ok=True)
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Source path does not exists: '{input_dir}'")
    if not Path(move_dir).exists():
        raise FileNotFoundError(f"Destination path does not exists: '{move_dir}'")

    # NOTE: glob uses bash like expression expansion, i.e. `*` => any string of any length
    training_files = sorted(Path(input_dir).glob(f"*{file_suffix}"))
    # The order of training files is shuffled randomly so that the model does not get biased
    random.shuffle(training_files)
    with tqdm(training_files, ncols=100) as t:
        print(f"Input files = {len(training_files)}")
        print(f"Processed files = {len(list(Path(move_dir).glob(f'*{file_suffix}')))}")
        if len(list(Path(input_dir).glob("*"))) > 0 and len(list(Path(move_dir).glob("*"))) == 0:
            keras_obj.model_version.epochs += epochs
            keras_obj.model_version.version = 0

        for ith_file in t:
            t.set_description(desc=f"File: {Path(ith_file).name}", refresh=True)
            print("\n")
            train_on_file(keras_obj=keras_obj,
                          file_path=str(ith_file),
                          data_load_transform=data_load_transform,
                          y_normalizer=y_normalizer,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_split=validation_split)
            # keras_obj.update_version()
            keras_obj.c_save_weights(write_name='z_last_model_weight_name.txt')
            shutil.move(src=str(Path(input_dir) / Path(ith_file).name), dst=str(Path(move_dir)))

            gc.collect()
            time.sleep(5)
    print("\n\nModel training finished :)\n\n")
    keras_obj.c_save_weights(write_name='z_last_model_weight_name.txt')


def get_available_gpus() -> List:
    """
    Available GPU's device names

    :return:
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


########################################################################################################################

def train(gpu_id: int,
          builder_model: int,
          board_encoder: int,
          version_number: int,
          epochs: int,
          batch_size: int,
          validation_split: float,
          generate_model_image: bool,
          input_dir: str,
          move_dir: str,
          file_suffix: str,
          y_normalizer: str,
          callback: bool,
          name_prefix: str,
          auto_load_new: bool,
          saved_weights_file: str = '',
          weights_save_path: str = '') -> None:
    """
    Train a Keras model
    :param gpu_id: ID of the GPU to be used for training [0,1,...], use -1 if CPU is to be used for training
    :param builder_model: Which NNBuilder keras model to load
    :param board_encoder: Which "BoardEncoder" to use
    :param version_number: Set the version number of the model loaded/created
    :param epochs: Number of epochs to be executed on each file
    :param batch_size: Number of rows to be used at one time for weight adjustment, like one of these
                       [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    :param validation_split: What fraction of data to be used for validation of the trained model
    :param generate_model_image: Whether to write the model structure to a pgn file or not ?
    :param input_dir: Path to the directory where input data is stored
    :param move_dir: Path to the directory where processed files are to be moved so that training can be resumed if interrupted
    :param file_suffix: Suffix of the input files, i.e. file extension ['csv', 'pkl']
    :param y_normalizer: 'None' is expected if output(y) is to be used as it is, otherwise use
                          one of ['normalize_001', 'normalize_002'] to normalize the expected output(y)
    :param callback: Whether to save intermediate weights if the results have improved
    :param name_prefix: Prefix to be used while saving the new ANN weights in h5 file
    :param auto_load_new: Whether to automatically load the last saved model or not if `name_prefix` is same ? This overrides `saved_weights_file` option
    :param saved_weights_file: If this file exists, then load it, else create the model with random weights
    :param weights_save_path: Path to directory where new weights of the model shall be saved
    :return: None
    """
    Path(move_dir).mkdir(parents=True, exist_ok=True)
    # if not (4 <= builder_model <= 5):
    #     print(f"ERROR: Invalid `builder_model={builder_model}`, builder_model should be `4 <= builder_model <= 5`")
    #     return
    if not (0 <= version_number):
        print(f"ERROR: Invalid `version_number={version_number}`, it should be `0 <= version_number`")
        return
    if not (1 <= epochs):
        print(f"ERROR: Invalid `epochs={epochs}`, it should be `1 <= epochs`")
        return
    if not (1 <= batch_size):
        print(f"ERROR: Invalid `batch_size={batch_size}`, it should be `1 <= batch_size`")
        return
    if not (0.0 <= validation_split <= 1.0):
        print(f"ERROR: Invalid `validation_split={validation_split}`, it should be `0.0 <= validation_split <= 1.0`")
        return
    if not Path(input_dir).exists():
        print(f"ERROR: `input_folder={input_dir}` does NOT exists")
        return
    if not Path(move_dir).exists():
        print(f"ERROR: `move_folder={move_dir}` does NOT exists")
        return

    data_load_transform = None

    if file_suffix == 'csv':
        data_load_transform = pd.read_csv
    elif file_suffix == 'pkl':
        data_load_transform = joblib.load
    else:
        print(f"ERROR: only 'csv' and 'pkl' file can be used to read/load data")
        return

    if 0 <= gpu_id < len(get_available_gpus()):
        # NOTE: IMPORTANT: TRAINING on GPU ***
        # tf.device("/gpu:0")
        tf.device(f"/gpu:{gpu_id}")
    elif gpu_id != -1:
        print(f"WARNING: Invalid parameter for `gpu_id={gpu_id}`, using CPU for training")

    y_normalizer_obj: Union[Callable[[np.ndarray], np.ndarray], None] = None
    if (y_normalizer != "None") and (y_normalizer is not None):
        try:
            for i in step_02.ScoreNormalizer.get_all_SN_suffix_str():
                if int(y_normalizer) == int(i):
                    y_normalizer_obj = step_02.ScoreNormalizer.num_to_method(int(i))
                    break
        except Exception as e:
            print(f"EXCEPTION: {type(e)}: {e}")
    if y_normalizer_obj == step_02.ScoreNormalizer.normalize_000:
        y_normalizer_obj = None

    # TODO: verify the below checking
    # if y_normalizer != "None" and not (y_normalizer is None):
    if y_normalizer != "None" and (y_normalizer is None):
        print(type(y_normalizer))
        print(f"WARNING: Invalid parameter for `y_normalizer={y_normalizer}`, using default value `y_normalizer=None`")

    ffnn_keras_obj: step_03a.NNKeras = step_03a.NNBuilder.build_from_model_version(
        step_03a.ModelVersion(name_prefix, builder_model, board_encoder, int(y_normalizer), epochs, 'weight', version_number, 'h5'),
        callback,
        generate_model_image
    )

    if ffnn_keras_obj is None:
        print("ERROR: wrong Model parameters passed")
        return

    ffnn_keras_obj.model_save_path = weights_save_path

    saved_weights_file = Path(saved_weights_file)
    if auto_load_new and (saved_weights_file.parent / 'z_last_model_weight_name.txt').exists():
        print(f"INFO: auto_load_new: Trying")
        last_saved_file_name = eval(open(str(saved_weights_file.parent / 'z_last_model_weight_name.txt'), 'r').read().strip())
        # Path(params.saved_weights_file).parent /
        try:
            if name_prefix in last_saved_file_name.keys():
                saved_weights_file = saved_weights_file.parent / last_saved_file_name[name_prefix]
        except Exception as e:
            print(f"ERROR: auto_load_new: {e}")

    if auto_load_new and saved_weights_file.exists() and saved_weights_file.is_file():
        print(f"INFO: auto_load_new: loading: '{saved_weights_file}'")
        ffnn_keras_obj.c_load_weights(str(saved_weights_file.name), str(saved_weights_file.parent))
        print("INFO: auto_load_new: Model loaded :)")
    elif auto_load_new:
        print("WARNING: auto_load_new: failed")
    print()

    train_on_folder(ffnn_keras_obj,
                    input_dir,
                    move_dir,
                    file_suffix,
                    data_load_transform=data_load_transform,
                    y_normalizer=y_normalizer_obj,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split)


########################################################################################################################

if __name__ == '__main__':
    from docopt import docopt

    doc_string = '''
    Usage:
        step_03b_train.py get_available_gpus
        step_03b_train.py train \
--gpu=N --builder_model=N --board_encoder=N [--version_number=N] --epochs=N \
--batch_size=N --validation_split=FLOAT [--generate_model_image] \
--input_dir=PATH --move_dir=PATH --file_suffix=STR --y_normalizer=STR \
[--callback] --name_prefix=STR [--auto_load_new] --saved_weights_file=PATH \
--weights_save_path=PATH
        step_03b_train.py get_options_models
        step_03b_train.py (-h | --help)
        step_03b_train.py --version

    Options:
        --gpu=N                     The GPU to use for training. By default CPU is used [default: -1]
        --builder_model=N           Which NNBuilder keras model to load
        --board_encoder=N           Which NNBuilder keras model to load
        --version_number=N          Set the version number of the model loaded/created [default: 0]
        --epochs=N                  Number of epochs to be executed on each file
        --batch_size=N              Number of rows to be used at one time for weight adjustment, like one of these ---> 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
        --validation_split=FLOAT    What fraction of data to be used for validation of the trained model
        --generate_model_image      Flag to decide whether to write the model structure to a pgn file or not ?
        --input_dir=PATH            Path to the directory where input data is stored
        --move_dir=PATH             Path to the directory where processed files are to be moved so that training can be resumed if interrupted
        --file_suffix=STR           Suffix of the input files, i.e. file extension like 'csv', 'pkl'
        --y_normalizer=STR          'None' is expected if output(y) is to be used as it is, otherwise use one of ['001', '002'] to normalize the expected output(y)
        --callback                  Flag to decide whether to save intermediate weights if the results have improved
        --name_prefix=STR           Prefix to be used while saving the new ANN weights in h5 file
        --auto_load_new             Flag to decide whether to automatically load the last saved model or not if `name_prefix` is same ? This overrides the `saved_weights_file` option
        --saved_weights_file=PATH   If this file exists, then load it, else create the model with random weights
        --weights_save_path=PATH    Path to directory where new weights of the model shall be saved

        -h --help               Show this
        --version               Show version
    '''
    arguments = docopt(doc_string, argv=None, help=True, version=f"{cs.VERSION} - Training", options_first=False)
    print("\n\n", arguments, "\n\n", sep="")

    # Use the output of above print and paste it below, and comment the above two statements to use IDE Debugger
    # arguments = {'--auto_load_new': False,
    #              '--batch_size': '8192',
    #              '--board_encoder': '01588',
    #              '--builder_model': '6',
    #              '--callback': True,
    #              '--epochs': '8',
    #              '--file_suffix': 'pkl',
    #              '--generate_model_image': True,
    #              '--gpu': '-1',
    #              '--help': False,
    #              '--input_dir': '../../Chess-Force-CNN-Dataset/04_pkl_data',
    #              '--move_dir': '../../Chess-Force-CNN-Dataset/04_pkl_data_trained',
    #              '--name_prefix': 'cnn',
    #              '--saved_weights_file': '../../Chess-Force-Models/ffnn_keras-mg005-be00778-sn003-ep00005-weights-v031.h5',
    #              '--validation_split': '0.2',
    #              '--version': False,
    #              '--version_number': '0',
    #              '--weights_save_path': '../../Chess-Force-Models',
    #              '--y_normalizer': '004',
    #              'get_available_gpus': False,
    #              'train': True}
    # arguments = {'--auto_load_new': False,
    #     '--batch_size': '8192',
    #     '--board_encoder': '01588',
    #     '--builder_model': '7',
    #     '--callback': True,
    #     '--epochs': '4',
    #     '--file_suffix': 'pkl',
    #     '--generate_model_image': False,
    #     '--gpu': '0',
    #     '--help': False,
    #     '--input_dir': '../../Chess-Force-CNN-Dataset/04_pkl_data_combined',
    #     '--move_dir': '../../Chess-Force-CNN-Dataset/04_pkl_data_trained',
    #     '--name_prefix': 'cnn',
    #     '--saved_weights_file': '',
    #     '--validation_split': '0.2',
    #     '--version': False,
    #     '--version_number': '0',
    #     '--weights_save_path': '../../Chess-Force-Models',
    #     '--y_normalizer': '7',
    #     'get_available_gpus': False,
    #     'get_options_models': False,
    #     'train': True}


    if arguments['get_available_gpus']:
        print(get_available_gpus())
    elif arguments['get_options_models']:
        custom_obj_str = [
            int(i.lstrip("model_")) 
                for i in cs.get_class_common_prefixed(
                    step_03a.KerasModels,
                    prefix_to_search='model_'
                )
        ] 
        print(f"KerasModels = {custom_obj_str}")
    elif arguments['train']:
        train(int(arguments['--gpu']),
              int(arguments['--builder_model']),
              int(arguments['--board_encoder']),
              int(arguments['--version_number']),
              int(arguments['--epochs']),
              int(arguments['--batch_size']),
              float(arguments['--validation_split']),
              arguments['--generate_model_image'],  # bool
              arguments['--input_dir'],
              arguments['--move_dir'],
              arguments['--file_suffix'],
              arguments['--y_normalizer'],
              arguments['--callback'],  # bool
              arguments['--name_prefix'],
              arguments['--auto_load_new'],  # bool
              arguments['--saved_weights_file'],
              arguments['--weights_save_path'])
    else:
        print("ERROR: invalid option")

    # # GET list of devices
    # get_available_gpus()
    # # list all local devices
    # device_lib.list_local_devices()
    # # other command, effect not known
    # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

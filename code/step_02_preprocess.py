import copy
import glob
import itertools
import logging
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Union, Iterable, List, TextIO, Tuple

import chess.pgn
import joblib
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import common_services as cs
import step_01_engine
from step_02_BoardEncoder import BoardEncoder
from step_02_ScoreNormalizer import ScoreNormalizer

engine_sf = None
MINI_BATCH_SIZE: int = 1000
BATCH_SIZE: int = 10000

# REFER: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
# This sets the root logger to write to stdout (your console)
logging.basicConfig()


########################################################################################################################

class PreprocessPGN:
    def __init__(self, pgn_file_path: Union[str, Path]):
        if not pgn_file_path.endswith(".pgn"):
            raise Exception(f"ERROR: This is not pgn file: {pgn_file_path}")
        self.pgn_file_path: str = str(pgn_file_path)

        self.pgn_text_io: TextIO = self.__load_pgn()
        print(f"DEBUG: pgn file {self.pgn_file_path} successfully loaded", file=sys.stderr)

    def __load_pgn(self) -> Union[TextIO, Exception]:
        # https://python-chess.readthedocs.io/en/latest/pgn.html
        if not Path(self.pgn_file_path).exists():
            print(f"ERROR: {self.pgn_file_path} does not exist", file=sys.stderr)
            raise FileNotFoundError(f"'{self.pgn_file_path}'")

        pgn = open(self.pgn_file_path, mode="rt")
        return pgn

    def iterate_pgn(self) -> Iterable[chess.pgn.Game]:
        game = chess.pgn.read_game(self.pgn_text_io)
        while game is not None:
            yield game
            try:
                game = chess.pgn.read_game(self.pgn_text_io)
            except UnicodeDecodeError as e:
                print(
                    f"WARNING: it seems pgn file has been completely read, UnicodeDecodeError occurred:\n\t{e}",
                    file=sys.stderr
                )
                break
        return

    @staticmethod
    def iterate_game(nth_game: chess.pgn.Game) -> Iterable[chess.Board]:
        board = nth_game.board()
        for move in nth_game.mainline_moves():
            board.push(move)
            yield copy.deepcopy(board)
        return

    @staticmethod
    def generate_boards(nth_board: chess.Board) -> Iterable[chess.Board]:
        legal_move = nth_board.legal_moves
        for move in legal_move:
            nth_board.push(move)
            yield copy.deepcopy(nth_board)
            nth_board.pop()

    @staticmethod
    def generate_boards_list(nth_board: chess.Board) -> List[chess.Board]:
        return list(PreprocessPGN.generate_boards(nth_board))

    def reload_pgn(self):
        self.pgn_text_io: TextIO = self.__load_pgn()
        print(f"DEBUG: pgn file {self.pgn_file_path} successfully re-loaded", file=sys.stderr)

    def get_pgn_game_count(self) -> int:
        game_count = 0
        for i in self.iterate_pgn():
            game_count += 1

        self.reload_pgn()
        return game_count


########################################################################################################################

class FenToPkl:
    @staticmethod
    def get_suffix_to_append(board_encoder, score_normalizer):
        return f"-be{board_encoder.UNIQ_NAME_5}-sn{ScoreNormalizer.get_suffix_str(score_normalizer)}.pkl"

    @staticmethod
    def __load_transform(board_encoder,
                         score_normalizer,
                         file_path: str,
                         print_execution_time: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        with cs.ExecutionTime(file=sys.stderr, print=print_execution_time):
            data = pd.read_csv(file_path, dtype={cs.COLUMNS[0]: str, cs.COLUMNS[1]: np.float32})
            data_x = data[cs.COLUMNS[0]].values
            data_y = data[cs.COLUMNS[1]].values
            # print(data.head())

            data_x_encoded: np.ndarray = board_encoder.encode_board_n_fen(data_x)
            # data_x_encoded = step_02.BoardEncoder.Encode778.encode_board_n_fen(data_x)

            if score_normalizer is not None:
                data_y_normalized: np.ndarray = score_normalizer(data_y)
            else:
                data_y_normalized: np.ndarray = data_y
            # data_y_normalized = step_02.ScoreNormalizer.normalize_002(data_y)

        del data, data_x, data_y
        return data_x_encoded, data_y_normalized

    @staticmethod
    def convert_fen_to_pkl_file(file_path: str, output_dir: str, move_dir: str,
                                board_encoder_str, score_normalizer_str,
                                suffix_to_append: str = "-be?????-sn???.pkl",
                                print_execution_time: bool = False):
        suffix_to_append = FenToPkl.get_suffix_to_append(board_encoder_str, score_normalizer_str)
        data_x_encoded, data_y_normalized = FenToPkl.__load_transform(board_encoder_str,
                                                                      score_normalizer_str,
                                                                      file_path,
                                                                      print_execution_time=print_execution_time)

        # Path(...).stem returns file name only without extension
        # compress=1, performs basic compression. compress=0 means no compression
        joblib.dump((data_x_encoded, data_y_normalized,),
                    filename=f"{Path(output_dir) / Path(file_path).stem}{suffix_to_append}",
                    compress=1)
        shutil.move(file_path, move_dir)
        del data_x_encoded, data_y_normalized

    @staticmethod
    def convert_fen_to_pkl_folder(input_dir: str, output_dir: str, move_dir: str,
                                  board_encoder, score_normalizer,
                                  suffix_to_append: str = "-be?????-sn???.pkl",
                                  print_execution_time: bool = False):
        """
        Example:
            >>> FenToPkl.convert_fen_to_pkl_folder(
            ...     input_dir="../../aggregated output 03",
            ...     output_dir="../../aggregated output 03/be00778-sn002-pkl",
            ...     move_dir="../../aggregated output 03/03_csv",
            ...     board_encoder=BoardEncoder.Encode_5_8_8,
            ...     score_normalizer=ScoreNormalizer.normalize_004,
            ... )
        :param input_dir:
        :param output_dir:
        :param move_dir:
        :param board_encoder:
        :param score_normalizer:
        :param suffix_to_append:
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(move_dir).mkdir(parents=True, exist_ok=True)
        if not Path(input_dir).exists():
            raise FileNotFoundError(f"Source path does not exists: '{input_dir}'")
        if not Path(output_dir).exists():
            raise FileNotFoundError(f"Destination path does not exists: '{output_dir}'")
        if not Path(move_dir).exists():
            raise FileNotFoundError(f"Move path does not exists: '{move_dir}'")

        with tqdm(sorted(glob.glob(f"{Path(input_dir)}/*.csv")), ncols=100) as t:
            for ith_file in t:
                t.set_description(f"File: {Path(ith_file).name[:25].ljust(25)}")
                FenToPkl.convert_fen_to_pkl_file(ith_file, output_dir, move_dir, board_encoder, score_normalizer, suffix_to_append, print_execution_time)


########################################################################################################################

def combine_pkl_files(input_dir: str, output_file: str):
    import gc

    all_files_list = glob.glob(str(Path(input_dir) / "*.pkl"))
    resa, resb = joblib.load(all_files_list[0])

    cnt = 0
    print(f"pkl count = {len(all_files_list)}")
    for i in tqdm(all_files_list, ncols=100):
        cnt += 1
        atemp, btemp = joblib.load(i)
        print("    " + str(cnt) + f" --- {atemp.shape} --- {btemp.shape}")
        resa = np.vstack((resa, atemp))  # this is 2-D
        resb = np.hstack((resb, btemp))  # this is 1-D
        del atemp, btemp
        gc.collect()  # Necessary as data is too large. So, RAM get consumed completely
        time.sleep(5)

    # REFER: https://stackoverflow.com/questions/35490148/how-to-get-folder-name-in-which-given-file-resides-from-pathlib-path
    parent_folder = Path(output_file).absolute().parents[0]
    file_name = Path(output_file).name
    if not parent_folder.exists():
        os.makedirs(str(parent_folder))

    if (not parent_folder.exists()) or (not os.access(str(parent_folder), os.W_OK)):
        print(f"Unable to create or write to folder: \"{str(parent_folder)}\"")
        print(f"You have 3 attempts to select a custom folder, otherwise the output will be saved in the current working directory")
        for i in range(3):
            parent_folder = Path(input("Enter new folder location: "))
            try:
                os.makedirs(parent_folder, exist_ok=True)
                if parent_folder.exists() and os.access(str(parent_folder), os.W_OK):
                    break
                else:
                    if not parent_folder.exists():
                        print("    Folder does not exists and unable to create one\n")
                    else:
                        print("    Write access denied\n")
            except:
                pass

    print()
    if parent_folder.exists() and os.access(str(parent_folder), os.W_OK):
        print("Output directory test result = Success, folder exists and is writable")
    else:
        print("3 attempts over")
        print("Using current working directory to save the file")
        parent_folder = Path(".")

    try:
        joblib.dump((resa, resb), filename=str(parent_folder / file_name), compress=1)
    except Exception as e:
        print(f"ERROR: {type(e)}: {e}")
        # REFER: https://www.programiz.com/python-programming/datetime/current-datetime
        from datetime import datetime
        time_now = datetime.now().strftime("__%d_%m_%Y__%H_%M_%S")
        file_name = 'CombinedDataset' + time_now + '.pkl'
        print(f"Trying to save to current folder with name = {file_name}")
        try:
            joblib.dump((resa, resb), filename=file_name, compress=1)
        except Exception as e2:
            print(f"ERROR: {type(e2)}: {e2}")
            return
    print("Done :)")

########################################################################################################################

if __name__ == "__main__":
    from docopt import docopt

    doc_string = \
        """
        Usage: 
            step_02_preprocess.py get_options [--parameter=[BoardEncoder | ScoreNormalizer]]
            step_02_preprocess.py convert_fen_to_pkl_file --file_path=PATH --output_dir=PATH --move_dir=PATH --board_encoder=ENCODER_NAME [--score_normalizer=METHOD_NAME] [--execution_time]
            step_02_preprocess.py convert_fen_to_pkl_folder --input_dir=PATH --output_dir=PATH --move_dir=PATH --board_encoder=ENCODER_NAME [--score_normalizer=METHOD_NAME] [--execution_time]
            step_02_preprocess.py combine_pkls --input_dir=PATH --output_file=PATH
            step_02_preprocess.py (-h | --help)
            step_02_preprocess.py --version

        Options:
            -h --help    show this
            --execution_time        Print time taken to convert one file after each iteration
            --output_file=PATH      Name of the file with its path where it is to be saved
        """

    arguments = docopt(doc_string, argv=None, help=True, version=f"{cs.VERSION} - Preprocess", options_first=False)
    print(arguments, end="\n\n----------------------------------------\n\n")

    if arguments['convert_fen_to_pkl_file']:
        os.makedirs(arguments['--output_dir'], exist_ok=True)
        os.makedirs(arguments['--move_dir'], exist_ok=True)
        FenToPkl.convert_fen_to_pkl_file(
            arguments['--file_path'],
            arguments['--output_dir'],
            arguments['--move_dir'],
            BoardEncoder.get_uniqstr_to_classobj(arguments['--board_encoder']),  # ("BoardEncoder." + str(arguments['--board_encoder'])),
            ScoreNormalizer.num_to_method(int(arguments['--score_normalizer'])),
            arguments['--execution_time']
            # f"ScoreNormalizer.{arguments['--score_normalizer']:0>3}",  # NOTE: 0>3 ensure number if prefixed with zero's if required
        )
    elif arguments['convert_fen_to_pkl_folder']:
        os.makedirs(arguments['--output_dir'], exist_ok=True)
        os.makedirs(arguments['--move_dir'], exist_ok=True)
        FenToPkl.convert_fen_to_pkl_folder(
            arguments['--input_dir'],
            arguments['--output_dir'],
            arguments['--move_dir'],
            BoardEncoder.get_uniqstr_to_classobj(arguments['--board_encoder']),
            ScoreNormalizer.str_to_method(str(arguments['--score_normalizer'])),
            arguments['--execution_time']
        )
    elif arguments['combine_pkls']:
        combine_pkl_files(arguments['--input_dir'], arguments['--output_file'])
    elif arguments['get_options']:
        class_to_prefix = {'BoardEncoder': 'Encode_', 'ScoreNormalizer': 'normalize_'}
        to_print_both = True if arguments['--parameter'] is None else False

        def get_custom_obj_str():
            return cs.get_class_common_prefixed(
                eval(arguments['--parameter']),
                prefix_to_search=(class_to_prefix[arguments['--parameter']] if arguments['--parameter'] in class_to_prefix else None)
            )

        if to_print_both:
            arguments['--parameter'] = 'BoardEncoder'
        if arguments['--parameter'] == 'BoardEncoder':
            custom_obj_str = [
                eval(f"BoardEncoder.{i}").UNIQ_NAME_5
                for i in get_custom_obj_str()
            ]
            print(f"BoardEncoder = {custom_obj_str}")
            pass
        
        if to_print_both:
            arguments['--parameter'] = 'ScoreNormalizer'
        if arguments['--parameter'] == 'ScoreNormalizer':
            custom_obj_str = [
                ScoreNormalizer.get_suffix_str(i)
                for i in get_custom_obj_str()
            ]
            print(f"ScoreNormalizer = {custom_obj_str}")
            pass
    else:
        print("ERROR: invalid command line arguments")
    pass

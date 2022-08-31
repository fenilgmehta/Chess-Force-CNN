import multiprocessing
import sys
from typing import Union, List, Tuple

import chess
import numpy as np

import common_services as cs


class BoardEncoder:
    @staticmethod
    def is_check(board: chess.Board, side: chess.Color) -> bool:
        king = board.king(side)
        return king is not None and board.is_attacked_by(not side, king)

    @staticmethod
    def is_checkmate(board: chess.Board, side: chess.Color) -> bool:
        return board.is_checkmate() and board.turn == side

    class EncodeBase(object):
        """
        All subclasses should follow the naming convention:
            Encode_[0-9a-zA-Z_]

        All subclasses should:
            - define a value for
                - "UNIQ_NAME_5" which follows the REGES [0-9]{5}
                - "INPUT_DIMENSIONS" which tells the dimensions of a single encoded board
            - implement
                - encode_board_1
                - encode_board_1_fen
                - encode_board_n_fen
                - encode_board_n
        """
        # 5 letters long uniq name which matches [0-9a-zA-Z]{5}
        # Used in naming files
        UNIQ_NAME_5: Union[None, str] = None
        INPUT_DIMENSIONS: Union[None, Tuple[int]] = None

        @staticmethod
        def encode_board_1(board_1: chess.Board) -> np.ndarray:
            pass

        @staticmethod
        def encode_board_1_fen(board_1_fen: str) -> np.ndarray:
            pass

        @staticmethod
        def encode_board_n_fen(board_n_fen: Union[List[str], Tuple[str]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards from FEN notation to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n_fen:
            :return: np.ndarray
            """

        @staticmethod
        def encode_board_n(board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n:
            :return: np.ndarray
            """

    class Encode_5_8_8(EncodeBase):
        UNIQ_NAME_5: str = "01588"
        INPUT_DIMENSIONS: Tuple[int] = (5, 8, 8,)

        str_to_int_mapper = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}

        @staticmethod
        def encode_board_1(board_1: chess.Board) -> np.ndarray:
            """
            # REFER: https://github.com/bajcmartinez/kickass-chess/blob/master/ai/state.py#L15
            :param board_1: chess.Board object
            :return:
            """
            if not board_1.is_valid():
                print(f"ERROR: invalid board state :(", file=sys.stderr)
                raise Exception("Invalid board state")

            board_state = np.zeros(64, np.uint8)

            # First we save the board in a 8x8 grid (or vector in this case)
            for i in range(64):
                pp = board_1.piece_at(i)
                if pp is None:
                    continue
                board_state[i] = BoardEncoder.Encode_5_8_8.str_to_int_mapper[pp.symbol()]

            # In order to store castling we are going to change the 'piece type' of the first or last column,
            # if I have the right to do castling queen side, the first column will switch from 4 (Rook) to 7 (Rook
            # with castling allowed), and in the case of black pieces same thing, but with 15 instead of 7
            if board_1.has_queenside_castling_rights(chess.WHITE):
                assert board_state[0] == 4
                board_state[0] = 7
            if board_1.has_kingside_castling_rights(chess.WHITE):
                assert board_state[7] == 4
                board_state[7] = 7
            if board_1.has_queenside_castling_rights(chess.BLACK):
                assert board_state[56] == 8 + 4
                board_state[56] = 15
            if board_1.has_kingside_castling_rights(chess.BLACK):
                assert board_state[63] == 8 + 4
                board_state[63] = 15

            # We reserve now number 8 for en passant
            if board_1.ep_square is not None:
                assert board_state[board_1.ep_square] == 0
                board_state[board_1.ep_square] = 8

            board_state = board_state.reshape(8, 8)
            state: np.ndarray = np.zeros((5, 8, 8), np.uint8)

            # 0-3 columns to binary (eg 0010 represents 2)
            state[0] = (board_state >> 3) & 1
            state[1] = (board_state >> 2) & 1
            state[2] = (board_state >> 1) & 1
            state[3] = (board_state >> 0) & 1

            # Next we save who's next (1 means white, 0 means black)
            state[4] = board_1.turn * 1.0
            return state

        @staticmethod
        def encode_board_1_fen(board_1_fen: str) -> np.ndarray:
            return BoardEncoder.Encode_5_8_8.encode_board_1(chess.Board(board_1_fen))

        @staticmethod
        def encode_board_n_fen(board_n_fen: Union[List[str], Tuple[str]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards from FEN notation to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n_fen:
            :return: np.ndarray
            """
            with multiprocessing.Pool() as pool:
                return np.array(
                    pool.map(func=BoardEncoder.Encode_5_8_8.encode_board_1_fen, iterable=board_n_fen)
                )
            # return BoardEncoder.encode_board_1_778_fen(board_n_fen)

        @staticmethod
        def encode_board_n(board_n: Union[List[chess.Board], Tuple[chess.Board]]) -> np.ndarray:
            """
            Convert list of tuple of chess boards to 778 floating point 0's and 1's

            NOTE: this computation is performed in parallel processes

            :param board_n:
            :return: np.ndarray
            """
            # TODO: verify the implementation
            with multiprocessing.Pool() as pool:
                return np.array(
                    pool.map(func=BoardEncoder.Encode_5_8_8.encode_board_1, iterable=board_n)
                )
            # return BoardEncoder.Encode_5_8_8.encode_board_n_fen(
            #     [
            #         board_i.fen() for board_i in board_n
            #     ]
            # )

    @staticmethod
    def get_uniqstr_to_classobj(uniq_str: str) -> EncodeBase:
        for i in cs.get_class_common_prefixed(BoardEncoder, prefix_to_search="Encode_"):
            if uniq_str == eval("BoardEncoder." + i).UNIQ_NAME_5:
                return eval("BoardEncoder." + i)
        return None

    @staticmethod
    def get_all_uniq_name() -> List[str]:
        return [
            eval("BoardEncoder." + i).UNIQ_NAME_5
            for i in cs.get_class_common_prefixed(BoardEncoder, prefix_to_search="Encode_")
        ]

import numpy as np
import random
from itertools import product

import time

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
    
    def select_piece(self):
        """
        상대방(P1)이 다음 턴에 선택하도록 '전략적으로 약한 말'을 고릅니다.
        특성이 중복된 말을 우선 선택하여 상대방이 전략을 세우기 어렵게 만듭니다.
        """
        time.sleep(0.5)  # 시간 소모 확인용 (완성 후 삭제 가능)

        # 특성 중복이 많은 말을 선택
        selected_piece = min(self.available_pieces, key=lambda piece: len(set(piece)))

        return selected_piece

    def place_piece(self, selected_piece):
        """
        selected_piece: 상대방이 선택한 말 (예: (1, 0, 1, 0))
        """
        # 사용할 수 있는 위치
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]

        # selected_piece를 인덱스로 변환
        piece_index = self.pieces.index(selected_piece) + 1  # 1-based index로 변환

        # 1. 이길 수 있는 위치가 있으면 우선적으로 그 위치 선택
        for loc in available_locs:
            temp_board = np.copy(self.board)
            temp_board[loc[0]][loc[1]] = piece_index  # 인덱스를 보드에 저장
            if self.check_win(temp_board, loc):
                return loc

        # 2. 상대방의 승리를 방지하기 위한 위치 선택
        for loc in available_locs:
            temp_board = np.copy(self.board)
            temp_board[loc[0]][loc[1]] = piece_index
            if self.prevent_opponent_win(temp_board, loc):
                return loc

        # 3. 이도 저도 아니면 랜덤 위치 선택
        return random.choice(available_locs)

    def check_win(self, board, loc):
        row, col = loc
        for attr in range(4):  # 속성 0~3 반복
            # 가로
            row_line = [board[row][c] for c in range(4)]
            if all(row_line) and self.check_line_win(row_line):
                return True

            # 세로
            col_line = [board[r][col] for r in range(4)]
            if all(col_line) and self.check_line_win(col_line):
                return True

        # 대각선 체크
        if row == col:  # 좌상단 -> 우하단
            diag1 = [board[i][i] for i in range(4)]
            if all(diag1) and self.check_line_win(diag1):
                return True

        if row + col == 3:  # 좌하단 -> 우상단
            diag2 = [board[i][3 - i] for i in range(4)]
            if all(diag2) and self.check_line_win(diag2):
                return True

        return False

    def check_line_win(self, line):
        """
        하나의 라인(행, 열, 대각선)이 승리 조건을 만족하는지 확인.
        """
        # 조건: 한 라인에 동일한 특성이 완성될 때
        return all(cell != 0 and cell == line[0] for cell in line)

    def prevent_opponent_win(self, board, loc):
        """
        Prevents the opponent from winning by evaluating potential board states.
        """
        row, col = loc

        # Check rows, columns, and diagonals
        for attr in range(4):
            # Row
            row_line = [board[row][c] for c in range(4)]
            if len(row_line) > 0 and self.is_opponent_winning(row_line, attr):
                return True

            # Column
            col_line = [board[r][col] for r in range(4)]
            if len(col_line) > 0 and self.is_opponent_winning(col_line, attr):
                return True

        # Diagonals
        if row == col:  # Left-to-right diagonal
            diag1 = [board[i][i] for i in range(4)]
            if len(diag1) > 0 and self.is_opponent_winning(diag1, attr):
                return True

        if row + col == 3:  # Right-to-left diagonal
            diag2 = [board[i][3 - i] for i in range(4)]
            if len(diag2) > 0 and self.is_opponent_winning(diag2, attr):
                return True

        return False

    def is_opponent_winning(self, line, attr):
        """
        Checks if the opponent is in a winning position based on a line and attribute.
        """
        pieces = [self.get_piece_attributes(piece_index) for piece_index in line]
        # 상대방이 이길 수 있는지 확인
        return all(piece and piece[attr] for piece in pieces if piece is not None)

    def get_piece_attributes(self, piece_index):
        """
        Converts a piece index on the board to its attribute tuple.
        """
        if piece_index == 0:
            return None  # Empty space
        return self.pieces[piece_index - 1]  # Convert 1-based index to 0-based

    #def is_opponent_winning(self, line):
        """
        상대방이 승리할 수 있는 패턴인지 확인.
        """
        # 0이 하나이고 나머지가 동일한 특성으로 채워져 있을 때
        #return list(line).count(0) == 1 and len(set(line) - {0}) == 1
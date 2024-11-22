import numpy as np
import random
from itertools import product
import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces
        self.first_place = True
    
    def select_piece(self):
        best_piece = None
        best_score = float('inf')

        for piece in self.available_pieces:
            score = self.minimax(self.board, piece, depth=2, is_maximizing=False)
            if score < best_score:
                best_score = score
                best_piece = piece

        return best_piece

    def place_piece(self, selected_piece):
        if self.first_place:
            # 첫 번째 말 놓을 때는 무작위로 선택
            self.first_place = False  # 첫 번째 선택을 완료했으므로 이후에는 기존 방식으로
            empty_cells = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
            return random.choice(empty_cells)
        
        else:
            best_move = None
            best_score = float('-inf')

            for row, col in product(range(4), range(4)):
                if self.board[row][col] == 0:
                    temp_board = self.board.copy()
                    temp_board[row][col] = self.pieces.index(selected_piece) + 1

                    score = self.minimax(temp_board, selected_piece, depth=2, is_maximizing=False)
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)

            return best_move
    
    def minimax(self, board, piece, depth, is_maximizing, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or self.check_win(board):
            return self.evaluate_board(board)

        if is_maximizing:  # p1의 턴
            max_eval = float('-inf')
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:  # 빈 칸만 탐색
                    temp_board = board.copy()
                    temp_board[row][col] = self.pieces.index(piece) + 1

                    eval = self.minimax(temp_board, piece, depth - 1, False, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)  # 알파 업데이트

                # 베타보다 알파가 크면 더 이상 탐색하지 않음
                    if beta <= alpha:
                        break
            return max_eval
        else:  # p2의 턴
            min_eval = float('inf')
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:
                    temp_board = board.copy()
                    temp_board[row][col] = self.pieces.index(piece) + 1

                    eval = self.minimax(temp_board, piece, depth - 1, True, alpha, beta)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)  # 베타 업데이트

                # 알파보다 베타가 작거나 같으면 더 이상 탐색하지 않음
                    if beta <= alpha:
                        break
            return min_eval

    
    def evaluate_board(self, board):

        if self.check_win(board):
            return 100

        score = 0
        for line in self.get_all_lines(board):
            characteristics = [self.pieces[piece_idx - 1] for piece_idx in line if piece_idx > 0]

            for i in range(4):
                if len(set([characteristics[j][i] for j in range(len(characteristics))])) == 1:
                    score += 10

        return score


    def check_win(self, board):
    # 모든 라인(가로, 세로, 대각선, 2x2 서브그리드)을 가져옴
        lines = self.get_all_lines(board)

        for line in lines:
        # 0(빈 칸)이 포함된 줄은 건너뜀
            if any(piece_idx == 0 for piece_idx in line):
                continue

        # 라인의 말들을 가져옴
            pieces = [self.pieces[piece_idx - 1] for piece_idx in line]

        # 각 특성별로 공통 속성이 있는지 확인 (0: 첫 번째 특성, 1: 두 번째, ...)
            for i in range(4):
                if all(piece[i] == pieces[0][i] for piece in pieces):  # i번째 속성이 동일한지 확인
                    return True  # 승리 조건 충족

        return False  # 승리 조건을 만족하지 않음

    def get_all_lines(self, board):
        lines = []
        for i in range(4):
            lines.append([board[i][j] for j in range(4)])  # 가로
            lines.append([board[j][i] for j in range(4)])  # 세로
        lines.append([board[i][i] for i in range(4)])  # 대각 1
        lines.append([board[i][3 - i] for i in range(4)])  # 대각 2

        # 2x2검사
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]]
                lines.append(subgrid)

        return lines
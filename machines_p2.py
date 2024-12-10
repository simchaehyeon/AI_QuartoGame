import numpy as np

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        """
        상대방에게 줄 말을 선택합니다.
        미니맥스 알고리즘을 사용하여 상대방의 승리 가능성을 최소화하는 말을 선택합니다.
        """
        best_piece = None
        min_score = float('inf')

        for piece in self.available_pieces:
            # 미니맥스를 사용하여 상대방의 점수를 계산
            score = self.minimax_select(piece, depth=3, is_maximizing=False)

            # 최소 점수를 유발하는 말을 선택
            if score < min_score:
                min_score = score
                best_piece = piece

        return best_piece

    def place_piece(self, selected_piece):
        """
        주어진 말을 보드에 배치합니다.
        미니맥스 알고리즘을 사용하여 최적의 위치를 선택합니다.
        """
        best_move = None
        max_score = float('-inf')

        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:  # 빈 칸이라면
                    # 미니맥스를 사용하여 점수를 계산
                    self.board[row][col] = self.pieces.index(selected_piece) + 1
                    score = self.minimax_place(self.board, depth=3, is_maximizing=False)
                    self.board[row][col] = 0  # 배치를 되돌림

                    # 최대 점수를 유발하는 위치를 선택
                    if score > max_score:
                        max_score = score
                        best_move = (row, col)

        return best_move

    def minimax_select(self, piece, depth, is_maximizing):
        """
        미니맥스 알고리즘 (select_piece용).
        - piece: 현재 고려 중인 말
        - depth: 탐색 깊이 제한
        - is_maximizing: 최대화 플레이어 여부
        """
        if depth == 0 or self.check_win(self.board):
            return self.evaluate(self.board)

        if is_maximizing:
            max_eval = float('-inf')
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        self.board[row][col] = self.pieces.index(piece) + 1
                        eval_score = self.minimax_select(piece, depth - 1, False)
                        max_eval = max(max_eval, eval_score)
                        self.board[row][col] = 0  # 배치를 되돌림
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(4):
                for col in range(4):
                    if self.board[row][col] == 0:
                        self.board[row][col] = -1  # 임시 배치 (상대방 말)
                        eval_score = self.minimax_select(piece, depth - 1, True)
                        min_eval = min(min_eval, eval_score)
                        self.board[row][col] = 0  # 배치를 되돌림
            return min_eval

    def minimax_place(self, board, depth, is_maximizing):
        """
        미니맥스 알고리즘 (place_piece용).
        - board: 현재 보드 상태
        - depth: 탐색 깊이 제한
        - is_maximizing: 최대화 플레이어 여부
        """
        if depth == 0 or self.check_win(board):
            return self.evaluate(board)

        if is_maximizing:
            max_eval = float('-inf')
            for row in range(4):
                for col in range(4):
                    if board[row][col] == 0:
                        board[row][col] = -1  # 임시 배치 (플레이어 말)
                        eval_score = self.minimax_place(board, depth - 1, False)
                        max_eval = max(max_eval, eval_score)
                        board[row][col] = 0  # 배치를 되돌림
            return max_eval
        else:
            min_eval = float('inf')
            for row in range(4):
                for col in range(4):
                    if board[row][col] == 0:
                        board[row][col] = -1  # 임시 배치 (상대방 말)
                        eval_score = self.minimax_place(board, depth - 1, True)
                        min_eval = min(min_eval, eval_score)
                        board[row][col] = 0  # 배치를 되돌림
            return min_eval

    def evaluate(self, board):
        """
        현재 보드 상태를 평가하여 점수를 반환합니다.
        """
        score = 0

        # 가로와 세로 줄 평가
        for i in range(4):
            row = [board[i][j] for j in range(4) if board[i][j] != 0]
            col = [board[j][i] for j in range(4) if board[j][i] != 0]
            score += self.line_score(row)
            score += self.line_score(col)

        # 대각선 평가
        diag1 = [board[i][i] for i in range(4) if board[i][i] != 0]
        diag2 = [board[i][3-i] for i in range(4) if board[i][3-i] != 0]
        score += self.line_score(diag1)
        score += self.line_score(diag2)

        # 2x2 사각형 평가
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                subgrid = [idx for idx in subgrid if idx != 0]
                score += self.square_score(subgrid)

        return score

    def line_score(self, line):
        """
        한 줄에서 동일한 속성이 얼마나 많은지 기반으로 점수를 계산합니다.
        """
        if len(line) < 2:
            return 0

        attributes = np.array([self.pieces[idx-1] for idx in line])  # 말의 속성 가져오기
        score = 0

        for i in range(4):  
            if len(set(attributes[:, i])) == 1:  
                score += len(line) * 10  

        return score

    def square_score(self, subgrid):
        """
        2x2 사각형에서 동일한 속성이 얼마나 많은지 기반으로 점수를 계산합니다.
        """
        if len(subgrid) < 4:
            return 0

        attributes = np.array([self.pieces[idx-1] for idx in subgrid])
        score = 0

        for i in range(4): 
            if len(set(attributes[:, i])) == 1: 
                score += 50  

        return score

    def check_win(self, board=None):
        """
        현재 보드 상태에서 승리 조건을 확인합니다.
        - 가로/세로/대각선/2x2 사각형에서 동일한 속성이 있는지 확인합니다.
        """
        if board is None:
            board = self.board

        # 가로와 세로 승리 조건 확인
        for i in range(4):
            row = [board[i][j] for j in range(4) if board[i][j] != 0]
            col = [board[j][i] for j in range(4) if board[j][i] != 0]
            if self.is_winning_line(row) or self.is_winning_line(col):
                return True

        # 대각선 승리 조건 확인
        diag1 = [board[i][i] for i in range(4) if board[i][i] != 0]
        diag2 = [board[i][3 - i] for i in range(4) if board[i][3 - i] != 0]
        if self.is_winning_line(diag1) or self.is_winning_line(diag2):
            return True

        # 2x2 사각형 승리 조건 확인
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]]
                subgrid = [idx for idx in subgrid if idx != 0]  # 빈 칸 제거
                if self.is_winning_square(subgrid):
                    return True

        return False

    def is_winning_line(self, line):
        """
        한 줄(가로, 세로, 대각선)이 승리 조건을 충족하는지 확인합니다.
        """
        if len(line) < 4:
            return False

        attributes = np.array([self.pieces[idx - 1] for idx in line])  # 말 속성 가져오기
        for i in range(4):
            if len(set(attributes[:, i])) == 1:  # 모든 속성이 동일하면 승리
                return True
        return False

    def is_winning_square(self, subgrid):
        """
        2x2 사각형이 승리 조건을 충족하는지 확인합니다.
        """
        if len(subgrid) < 4:
            return False

        attributes = np.array([self.pieces[idx - 1] for idx in subgrid])  # 말 속성 가져오기
        for i in range(4): 
            if len(set(attributes[:, i])) == 1:  # 모든 속성이 동일하면 승리
                return True
        return False
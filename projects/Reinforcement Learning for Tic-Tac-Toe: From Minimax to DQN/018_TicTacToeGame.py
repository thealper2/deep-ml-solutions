class TicTacToeGame:
    """Stateful Tic-Tac-Toe environment wrapping the Part 1 engine."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.status = 'ongoing'
        return self.board
    
    def legal_moves(self):
        return get_legal_moves(self.board)
    
    def is_terminal(self):
        return self.status != 'ongoing'
    
    def step(self, row, col):
        if self.is_terminal():
            raise ValueError("Game is already over")
        
        if not is_cell_empty(self.board, row, col):
            raise ValueError("Cell is already occupied")
        
        self.board = place_move(self.board, row, col, self.current_player)
        self.status = get_game_status(self.board)
        
        if self.status == 'ongoing':
            self.current_player = switch_player(self.current_player)
        
        return self.board.copy(), self.status

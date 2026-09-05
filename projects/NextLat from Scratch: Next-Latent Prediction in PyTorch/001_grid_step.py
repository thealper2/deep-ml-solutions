def grid_step(pos: tuple, action: int, G: int) -> tuple:
    row, col = pos
    
    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        col -= 1
    elif action == 3:
        col += 1
    
    if 0 <= row < G and 0 <= col < G:
        return (row, col), True
        
    return pos, False
def expand_node(node, priors):
    board = node['board']
    to_play = node['to_play']
    legal_actions = valid_moves(board)

    node['children'] = {}
    for action in legal_actions:
        child_board = drop_piece(board, action, to_play)
        child_node = make_mcts_node(priors[action], parent=node)
        child_node['board'] = child_board
        child_node['to_play'] = other_player(to_play)
        node['children'][action] = child_node

    node['is_expanded'] = True

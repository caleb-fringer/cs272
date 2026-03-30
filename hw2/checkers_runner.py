from checkers import ActionType, Direction, CheckersEnv

env = CheckersEnv()
env.reset()

moves = [
    (14, ActionType.MOVE, Direction.FL),
    (4, ActionType.MOVE, Direction.FL),
    (11, ActionType.CAPTURE, Direction.FL), # Black captures red
    (1, ActionType.CAPTURE, Direction.FR), # Red captures black
    (13, ActionType.MOVE, Direction.FL),
    (6, ActionType.CAPTURE, Direction.FL), # Red captures black
    (17, ActionType.MOVE, Direction.FR), # Black moves out of the way
    (13, ActionType.MOVE, Direction.FL), # Red gets promoted to king
    (14, ActionType.MOVE, Direction.FL), 
    (17, ActionType.MOVE, Direction.BL), # Move red king back.
    (11, ActionType.MOVE, Direction.FR), 
    (14, ActionType.MOVE, Direction.BR), 
    (8, ActionType.CAPTURE, Direction.FL), # Black captures and gets promoted
]

for move in moves:
    env.step(move)
    env.render()

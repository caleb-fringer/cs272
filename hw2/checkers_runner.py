from checkers import Direction, CheckersEnv

env = CheckersEnv()
env.reset()

moves = [
    (14, Direction.FL),
    (4, Direction.FL),
    (11, Direction.FL), # Black captures red
    (1, Direction.FR), # Red captures black
    (13, Direction.FL),
    (6, Direction.FL), # Red captures black
    (17, Direction.FR), # Black moves out of the way
    (13, Direction.FL), # Red gets promoted to king
    (14, Direction.FL), 
    (17, Direction.BL), # Move red king back.
    (11, Direction.FR), 
    (14, Direction.BR), 
    (8, Direction.FL), # Black captures and gets promoted
]

for move in moves:
    env.step(move)
    env.render()

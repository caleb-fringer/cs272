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
]

for move in moves:
    env.step(move)
    env.render()

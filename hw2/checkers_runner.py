import checkers

env = checkers.CheckersEnv()
env.reset()

env.step((13,checkers.Direction.FR))
env.render()
env.step((3,checkers.Direction.FL))
env.render()

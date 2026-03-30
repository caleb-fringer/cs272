from checkers import ActionType, Direction, CheckersEnv

env = CheckersEnv()
env.reset()

moves = [
    # 1. Black moves 13 -> 10
    (13, ActionType.MOVE, Direction.FL),
    
    # 2. Red moves 3 -> 6 (Now Red is at pos 6)
    (3, ActionType.MOVE, Direction.FL),
    
    # 3. Black at 10 MUST capture Red at 6, landing on 3.
    (10, ActionType.CAPTURE, Direction.FL),
    
    # 4. Red moves 4 -> 7
    (4, ActionType.MOVE, Direction.FL),
    
    # 5. Black moves 16 -> 13
    (16, ActionType.MOVE, Direction.FR),
    
    # 6. Red moves 0 -> 4
    (0, ActionType.MOVE, Direction.FL),
    
    # 7. Black moves 3 -> 0, resulting in promption
    (3, ActionType.MOVE, Direction.FR),

    # 8. Red moves 4 -> 7
    (4, ActionType.MOVE, Direction.FR),

    # 9. Black moves 13 -> 10, forcing Red to capture
    (13, ActionType.MOVE, Direction.FL),

    # 10. Red captures 6 -> 13, forcing Black to capture
    (6, ActionType.CAPTURE, Direction.FL),

    # 11. Black must capture 17 -> 10
    (17, ActionType.CAPTURE, Direction.FL),

    # 12. Red blocks
    (1, ActionType.MOVE, Direction.FR),

    # Black moves
    (10, ActionType.MOVE, Direction.FL),

    # Red must double capture
    (4, ActionType.CAPTURE, Direction.FR),
    (9, ActionType.CAPTURE, Direction.FL),

    # Black moves into the wya
    (15, ActionType.MOVE, Direction.FR),

    # Red king captures backwards
    (16, ActionType.CAPTURE, Direction.BR),

    # Black king moves backwards
    (0, ActionType.MOVE, Direction.BR),

    (7, ActionType.MOVE, Direction.FL),

    # Black must capture
    (14, ActionType.CAPTURE, Direction.FL),

    # Red must capture
    (5, ActionType.CAPTURE, Direction.FR),

    (4, ActionType.MOVE, Direction.FR),

    (10, ActionType.MOVE, Direction.FL),
    
    (1, ActionType.MOVE, Direction.BL),

    (13, ActionType.MOVE, Direction.FL),

    (4, ActionType.MOVE, Direction.FL),

    (17, ActionType.MOVE, Direction.BR),

    (0, ActionType.MOVE, Direction.BR),

    (13, ActionType.MOVE, Direction.BL),

    (4, ActionType.MOVE, Direction.FL),

    (11, ActionType.MOVE, Direction.BR),

    (0, ActionType.MOVE, Direction.BL),

    (9, ActionType.MOVE, Direction.BL),

    (3, ActionType.CAPTURE, Direction.BR),

    (10, ActionType.CAPTURE, Direction.FR),

    (2, ActionType.CAPTURE, Direction.FR),
]

env = CheckersEnv()
obs = env.reset()

for i, action in enumerate(moves):
    pos, action_type, direction = action
    
    print(f"\n--- Move {i+1}: Player '{env.current_agent}' playing Pos {pos}, {action_type, direction} ---")
    
    # Submit the action to the environment
    obs = env.step(action)
    env.render()
    
    # Check if the game terminated (e.g., on the last illegal move)
    if any(env.terminations.values()):
        print("Game Terminated!")
        print("Rewards:", env.rewards)
        break

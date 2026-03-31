# CS-272 HW 2

## Checkers
| Import | `from mycheckersenv import CheckersEnv` |
|--------|----------------------------------|
| Actions | Discrete |
| Parallel API | No |
| Agents | `agents=["black", "red"]` |
| Agents | 2 |
| Action Shape | (3,) |
| Action Values | MultiDiscrete([18,2,4])
| Observation Shape | (4,6,6) |
| Observation Values | [0,1] |
| Action Mask Shape | (8,6,6) |
| Action Mask Values | [0,1] |

## Observation Space
For each agent, an observation is a dictionary of the following format:
{
    "observations": MultiBinary([4,6,6]),
    "action_mask": MultiBinary([8,6,6])
}

Both observations and action_mask contain values in [0,1]. 
The observations channels encode:
- Channel 1: Location of black's normal pieces (pawns).
- Channel 2: Location of red's normal pieces (pawns).
- Channel 3: Location of black's promoted pieces (kings).
- Channel 4: Location of red's promoted pieces (kings).

For the current agent, the action_mask channels encode:
- Channels 0-3: Regular 1-step moves (FR, FL, BR, BL)
- Channels 4-7: Capture 2-step moves (FR, FL, BR, BL)


## Action Space
A tuple of (position_no, ActionType, Direction). 

Positions are counted from the top-left corner to the bottom right corner, 
from left to right, skipping white squares where checkers cannot move. 
The actual board co-ordinates can be calculated from position_no as follows:
(row, col) = (position_no // 3, 2 * (position_no % 3) (+1 if row % 2 == 0))
Example: position_no = 8, (row, col) = (2, 5)

ActionTypes are numbered according to the ActionType enum, where 0 represents
a regular move and 1 represents a capture.

Directions are numbered according to the Direction enum and 
correspond to a direction vector defined as the Direction.vector property.

## Rewards
The agent receives a reward of +/-1 for a win/loss respectively at the end
of an episode. If the agent takes an illegal move, they receive a reward of -1
and the game terminates with a reward of 0 for the other agent. All other
rewards are 0.


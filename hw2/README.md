# CS-272 HW 2

## Checkers
| Import | `from checkers import CheckersEnv` |
|--------|----------------------------------|
| Actions | Discrete |
| Parallel API | No |
| Agents | `agents=["black", "red"]` |
| Agents | 2 |
| Action Shape | (2,) |
| Action Values | MultiDiscrete([18,4])
| Observation Shape | (6,6,8) |
| Observation Values | [0,1] |

## Observation Space
For each agent, an observation consists of a stack of 6x6 matrices
containing values in [0,1]. Each channel encodes the following information:
- Channel 1: Location of black's normal pieces (pawns).
- Channel 2: Location of red's normal pieces (pawns).
- Channel 3: Location of black's promoted pieces (kings).
- Channel 4: Location of red's promoted pieces (kings).
- Channel 5-8: Current agent's legal action mask of moves FR, FL, BR, BL, respectively. 1 represents a legal action

## Action Space
A tuple of (position_no, direction_no). Positions are counted from the
top-left corner to the bottom right corner, from left to right, skipping
white squares where checkers cannot move. The actual board co-ordinates
can be calculated from position_no as follows:
(row, col) = (position_no // 3, 2 * (position_no % 3) (+1 if row % 2 == 0))
Example: position_no = 8, (row, col) = (2, 5)

Directions are numbered according to the checkers.Direction enum and 
correspond to a direction vector defined as the Direction.vector property.

Example: Action = ()

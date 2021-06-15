#test getInitBoard
import othello.OthelloGame as a
g = a.OthelloGame(9)
g.getInitBoard()
from othello.OthelloLogic import Board
b = Board(9)
import numpy as np
print(np.array(b.pieces))
#same result, test pass

import othello.goGame as x
a = x.GoGame()
a.place_black(1,1)
a.place_white(2,1)
a._place_stone(-1, 5, 4)
a._place_stone(1, 5, 4)
print(a.board)

import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use thisss script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True


g = OthelloGame(5)

# all players
rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 350, 'cpuct':1.1})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)[0])


player2 = hp


arena = Arena.Arena(player2, n1p, g, display=OthelloGame.display)
y, x, z, xb = arena.playGames(2, verbose=True)
print("Bots win: ", x)
print("Human win: ", y)
print("Draw: ", z)
print("Bot Win with Black: ", xb )

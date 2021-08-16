import Arena
from MCTS import MCTS
from go.GoGame import GoGame
from go.GoPlayers import *
from go.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use thisss script to play any two agents against each other, or play manually with
any agent.
"""
args = dotdict({
    'size': 5,                  #board size
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.1,
    'arenaNumMCTSSims': 20,      # simulations for arena
    'instinctArena': True,      # if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'resignThreshold': -0.999999, # No Use. Resign Only in self-play Training

})
human_vs_cpu = True


g = GoGame(args)

# all players
rp = RandomPlayer(g).play
#gp = GreedyGoPlayer(g).play
hp = HumanGoPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/','five.pth.tar')
mcts1 = MCTS(g, n1, args)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0,instinctPlay=args.instinctArena)[0])


player2 = hp


arena = Arena.Arena(player2, n1p, g, display=GoGame.display)
y, x, z, xb = arena.playGames(2, verbose=True)
print("Bots win: ", x)
print("Human win: ", y)
print("Draw: ", z)
print("Bot Win with Black: ", xb )

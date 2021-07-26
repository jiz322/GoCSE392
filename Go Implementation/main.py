import logging

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'size': 9,                  #board size
    'numIters': 1000,
    'tempThreshold': 0,        # zero
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaNumMCTSSims': 200,      #simulation for arena
    'arenaCompare': 2,         # Tornament version alsways 2
    'cpuct': 1.1,
    'instinctArena': False,     #if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'firstIter': False,        #set true if it produce first chechpoint to save, for multuprocess, the followings has to be FALSE
    'checkpoint': './temp/',
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 1000,
    'resignThreshold': -2   #resign does not work for Arena

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

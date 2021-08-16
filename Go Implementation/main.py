import logging

import coloredlogs

from Coach import Coach
from go.GoGame import GoGame as Game
from go.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'size': 9,                  #board size
    'numIters': 100,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 30,        # infite
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 800,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.1,
    'arenaNumMCTSSims': 52,      #simulation for arena
    'instinctArena': False,     #if set true reset Arena's MTCL tree each time
    'balancedGame': True,      # if balanced, black should win over 6 scores
    'firstIter': False,        #set true if it produce first chechpoint to save, for multuprocess, the followings has to be FALSE
    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 100,
    'resignThreshold': -2   #resign when best Q value less than threshold Q[-1, 1]

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

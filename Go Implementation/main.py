import logging

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 50,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 1500,        # infite
    'updateThreshold': 0.65,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 2,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 8,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.1,
    'arenaNumMCTSSims': 25,
    'firstIter': True,        #set true if it produce first chechpoint to save, for multuprocess, the following has to be FALSE

    'checkpoint': './temp/',
    'load_model': False,      #load example only
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(5, False)

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
    #stage1 returns a signal whether to go to stage2
    continueStage2 = c.learn(stage1=True)
    #stage 2, use the best.pth, abondon examples
    if continueStage2:
        args_stage2 = dotdict({
            'numIters': 50,
            'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': 1500,        # infite
            'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
            'numMCTSSims': 52,          # Number of games moves for MCTS to simulate.
            'arenaCompare': 8,         # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': 1.1,
            'arenaNumMCTSSims': 52,
            'firstIter': False,        #set true if it produce first chechpoint to save, for multuprocess, the following has to be FALSE

            'checkpoint': './temp/',
            'load_model': True,        #load example only
            'load_folder_file': ('./temp','best.pth.tar'),
            'numItersForTrainExamplesHistory': 50,
 
        })
        log.info('Loading %s...', Game.__name__)
        g = Game(5, balanced=True) #stage 2 is balanced game

        log.info('Loading %s...', nn.__name__)
        nnet = nn(g)

        if args.load_model:
            log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        c = Coach(g, nnet, args_stage2)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the stage2 🎉')
        #stage2
        c.learn(stage1=False)


if __name__ == "__main__":
    main()

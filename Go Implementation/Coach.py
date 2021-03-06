import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.firstIter = args.firstIter #set true if it produce first chechpoint to save
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            #fastDecision data should not be collected (ketaGo Paper)
            pi, fastDecision, resign = self.mcts.getActionProb(canonicalBoard, temp=temp, training=1)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                if not fastDecision:  #only add example of slow decisions
                    trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            r = self.game.getGameEnded(board, self.curPlayer)
            if resign:
                r = 1 # previous player resigned          
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.

        Go edit:
        load best.pth.tar to compare and self-iteration
        temp.pth.tar only for training. it may be overwrite shortly
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            self.trainExamplesHistory = [] # empty the history 
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples()
            self.loadExamples() #see if this get larger
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')# training new network, keeping a copy of the old one              
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)
            if not self.firstIter:
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') #Load the best after training to maximize efficenty
                pmcts = MCTS(self.game, self.pnet, self.args)
                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=1, arena=1, instinctPlay=self.args.instinctArena)[0]),
                            lambda x: np.argmax(nmcts.getActionProb(x, temp=1, arena=1, instinctPlay=self.args.instinctArena)[0]), self.game)
                pwins, nwins, draws, pwins_black = arena.playGames(self.args.arenaCompare)
                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d ; PREV_WinOnBlack : %d' % (nwins, pwins, draws, pwins_black))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                    log.info('REJECTING NEW MODEL')
                    #load the current best after reject
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                    #self.loadTrainExamples(loadBest=True) 
                    # Keep the example the same!!!!!!!
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            else: #first iteratioin, just accept it so that we have best.pth.tar
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.firstIter = False

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    #add the way to save best example
    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "best.pth.tar.examples")
        with open(filename, "ab+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    #add the way to load best example
    def loadTrainExamples(self, loadBest=False):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if loadBest:
            examplesFile = "best.pth.tar.examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            # do not skip if it loads the best
            if not loadBest:
                self.skipFirstSelfPlay = True
    #load examples only
    def loadExamples(self):
        folder = self.args.checkpoint
        filename = os.path.join(folder, "best.pth.tar.examples")
        log.info("File with trainExamples found. Loading it...")
        with open(filename, "rb") as f:
            count = 0
            while True:
                try:
                    if count == 0:
                        self.trainExamplesHistory = Unpickler(f).load()
                        count += 1
                    else:
                        self.trainExamplesHistory += Unpickler(f).load()
                except EOFError:
                    break 
        while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            self.trainExamplesHistory.pop(0)
        log.info('Loading done!')




import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.capFlag = False #True if reaches simulation cap


    # Noise only add to training mode (not for Arena Nor pit)
    def getActionProb(self, canonicalBoard, temp=1, training=0, arena=0, instinctPlay=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        ##Reset the Tree here for a fair arena game (do not enable it when training)
        if instinctPlay:
            self.__init__(self.game, self.nnet, self.args) #to make the Arena work as expected instead of a large single tree.
        #comment: reset tree here make the Arena more fair, but it also loose somewhat acuracy
        #for test only
        #
        #Add noise Improve the quality of sample (see Keta-Go paper)  
        # 25% of all do as many as numMCTSS searches, 75% do a quick search
        # Quick search add no noise
        # Only slow searches are recorded, so return isFast and pass it to Coach
        fastDecision = int(0.25*self.args.numMCTSSims)                                          
        noised_numMCTSSims = np.random.choice([self.args.numMCTSSims, fastDecision], p=[0.25, 0.75])
        isFast = (noised_numMCTSSims == fastDecision)
        if training == 1: # in self-iteration
            i = 0
            while(True):
                i += 1
                if i == 8000: #cap at 8000 total
                    break
                self.search(canonicalBoard, noise=not isFast, sim=noised_numMCTSSims) # Dirichlet noise only in slow decision
                if self.capFlag:
                    self.capFlag = False
                    break
        if arena == 1: # in arena
            #Also add keta noise to arena, but not dirichlet.
            i = 0
            while(True):
                i += 1
                if i == 8000: #cap at 8000 total
                    break
                self.search(canonicalBoard, noise=False, sim=self.args.numMCTSSims)
                if self.capFlag:
                    self.capFlag = False
                    break
        if training == 0 and arena == 0:
            #For testing out of learning
            i = 0
            while(True):
                i += 1
                if i == 8000: #cap at 8000 total
                    break
                self.search(canonicalBoard, noise=False, sim=self.args.numMCTSSims)
                if self.capFlag:
                    self.capFlag = False
                    break
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        # For go, the highest NSA may not a legal move due to the rule of 'ko' 
        # Then, we mask it 0.
        valids = self.game.getValidMoves(canonicalBoard, 1)
        counts = counts * valids 
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1    
            resign = False
            try:
                if training == 0 and arena == 0: #print info for testing outside of learning
                    print(np.array(self.Ps[s][:-1]).reshape(9,9))
                    print(np.array(counts[:-1]).reshape(9,9))
                    print(self.Qsa[(s, bestA)] )
                if self.Qsa[(s, bestA)] < self.args.resignThreshold: #resign when best Q value less than threshold
                    resign = True 
            except KeyError as e:
                print("get Ket error")
            return probs, isFast, resign
        #print(counts)
        counts = [x ** (1. / temp) for x in counts]
        #print(counts)
        counts_sum = float(sum(counts))
        if counts_sum != 0:
            probs = [x / counts_sum for x in counts]
        else: #vary rare, but it may happen due to the 'ko'
            probs = [0 for x in counts]
            probs[-1] = 1
        return probs, isFast, False

    #For fastDecision, no further noise add to p
    def search(self, canonicalBoard, noise=True, sim=0, pre_pass=False, pre_pre_pass=False, level=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        #print('mcts')
        #print(canonicalBoard)
        #print('l78')

        #!! DIFFERENT LOGIC HERE FOR GO TO AVOID STACKOVERFLOW
        #if s not in self.Es:
            #print('if s not in self.Es:')
        if pre_pass and pre_pre_pass:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)



            #print("===Es[s]===")  
            #print(self.Es[s])
            #print("^^^Es[s]===")  
            if self.Es[s] != 0:
                #print('if self.Es[s] != 0:')
                #print(canonicalBoard)
                # terminal node

                #empty passes
                canonicalBoard.pre_previous_is_pass = False
                canonicalBoard.previous_is_pass = False
                return -self.Es[s]

        if s not in self.Ps:                        #bug take away: copy using .copy
            #print('if s not in self.Ps:')
            #print(canonicalBoard)
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)

            #print('nn')
            #print(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            valid_length = len(self.Ps[s]) - np.count_nonzero(self.Ps[s]==0)
            if noise: # add dirichlet noise to the root policy
                #print("here")
                #print(self.Ps[s])
                #print(self.Ps[s][0])
                self.Ps[s] = 0.75*self.Ps[s] + 0.25*np.random.dirichlet([0.03*canonicalBoard.board_size**2/valid_length]*len(self.Ps[s]))
                #print(newPs)
                #print(0.25*np.random.dirichlet([0.03*canonicalBoard.board_size**2/valid_length]*len(self.Ps[s])))
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            
            
            #print(v)
            if sum_Ps_s > 0:
                #print('if sum_Ps_s > 0:')
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                #print('else')
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            #print(canonicalBoard)
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1


        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            #print('for a in range(self.game.getActionSize()):')
            if valids[a]:
                #print((a,valids[a]))
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] - self.Nsa[(s, a)] + 1) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = 10*self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        if a == 81:
            if not pre_pass:
                v = self.search(next_s, noise=False, sim=sim, pre_pass=True)
            else:
                v = self.search(next_s, noise=False, sim=sim, pre_pass=True, pre_pre_pass=True)
        else:
            v = self.search(next_s, noise=False, sim=sim) #keta Paper: dirichlet noise only add to root

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
            if self.Nsa[(s, a)] >= sim:
                self.capFlag = True
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

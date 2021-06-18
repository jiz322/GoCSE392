from __future__ import print_function
import sys
import copy
sys.path.append('..')
from Game import Game
import numpy as np
from othello.goGame import GameUI 
from othello.group import Group, GroupManager
'''
from othello.board import Board
from othello.utils import Stone, make_2d_array
from othello.group import Group, GroupManager
from othello.exceptions import (
    SelfDestructException, KoException, InvalidInputException)
'''
class OthelloGame(Game):
    square_content = {
        -1: "w",
        +0: "-",
        +1: "b"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = 9
        self.goGame = None

    def getInitBoard(self):
        # return initial board (numpy board)
        self.goGame = GameUI()
        #goGame.game.board = np.array(goGame.game.board).append([False, False])
        #b = Board(self.n)
        #return np.array(b.pieces)
        return self.goGame.game.board

    def getBoardSize(self):
        # (a,b) tuple
        return (9, 9)

    def getActionSize(self):
        # return number of actions
        return 82 #9*9+1

    def getNextState(self, board, player, action):

        # if player takes action on board, return next (board,player)
        # ###action must be a valid move

        #if action is pass, record it
        if action == self.n*self.n: #81
          #  print("action is 81")
            if board.previous_is_pass == True: 
                board.pre_previous_is_pass = True
            #    print("prep set to true")
            board.previous_is_pass = True
         #   print("p set to true")
            return (board, -player)
        move = (int(action/self.n), action%self.n) #remain
        self.goGame.game.board = copy.deepcopy(board)
        self.goGame.game.gm = GroupManager(self.goGame.game.board, enable_self_destruct=False)
        self.goGame.game.gm._group_map = board._group_map
        self.goGame._place_stone(move, player) #^
        board._group_map = self.goGame.game.gm._group_map
        board.previous_is_pass = False
        print(action)
        print(self.goGame.game.board)
        return (self.goGame.game.board, -player) #^

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize() #keep
        #b = Board(self.n)
        ##goGame = GameUI() #^
        #b.pieces = np.copy(board)
        tempGame = copy.deepcopy(self.goGame)
        tempGame.game.board = copy.deepcopy(board) #^
        #legalMoves =  b.get_legal_moves(player)

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        legalMoves = [ (a,b) for a in x for b in y]
        ilegalMoves = [] 
        for x, y in legalMoves:
            legal = tempGame._place_stone((x, y), player)
            if legal == False:
                #print((x,y))
                #print("is false")
                ilegalMoves.append((x,y))
                tempGame.game.board = copy.deepcopy(board)
            else:
                #print((x,y))
                #print("is true")
                tempGame.game.board = copy.deepcopy(board)
        legalMoves = self.Diff(legalMoves, ilegalMoves)
       # print (legalMoves)    
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        
        return np.array(valids)

    def getGameEnded(self, board, player):
        
        
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        #b = Board(self.n)
        #b.pieces = np.copy(board)
        ##goGame = GameUI() #^
        self.goGame.game.board = board#^
        #print(goGame.game.board)       debug
        #end with 2 consective passes
        # 
        #Black should win when 43:38 (5 points higher) 
        #for simplicity, whoever get 41 will win
        #print('self.goGame.game.board.previous_is_pass')
        #print(self.goGame.game.board.previous_is_pass)
        #print(self.goGame.game.board.pre_previous_is_pass)
        #print('self.goGame.game.board.pre_previous_is_pass')
        if (self.goGame.game.board.previous_is_pass and self.goGame.game.board.pre_previous_is_pass):
            #print('enddd')
            diff = self.goGame.game.get_scores().get(player)>self.goGame.game.get_scores().get(-player)
            #print(self.goGame.game.board)
            if diff > 0:
                #print("current player win")
                return 1
            else:
                #print("oppo player win")
                return -1
        else:
          #  print('not end')
            return 0


    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        #b = Board(self.n)
        #b.pieces = np.copy(board)
        ##goGame = GameUI() #^
        self.goGame.game.board = board.copy() #^
        #return b.countDiff(player)
        return self.goGame.game.get_scores().get(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
    
    def Diff(self, li1, li2):
        return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

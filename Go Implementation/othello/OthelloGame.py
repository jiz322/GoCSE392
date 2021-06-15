from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .OthelloLogic import Board
import numpy as np
from othello.goGame import GameUI 
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

    def getInitBoard(self):
        # return initial board (numpy board)
        goGame = GameUI()
        #goGame.game.board = np.array(goGame.game.board).append([False, False])
        #b = Board(self.n)
        #return np.array(b.pieces)
        return goGame.game.board

    def getBoardSize(self):
        # (a,b) tuple
        return (9, 9)

    def getActionSize(self):
        # return number of actions
        return 82 #9*9+1

    def getNextState(self, board, player, action):
        #b = Board(self.n)
        goGame = GameUI() #^
        #b.pieces = np.copy(board)
        goGame.game.board = board #^initialize

        # if player takes action on board, return next (board,player)
        # ###action must be a valid move

        #if action is pass, record it
        if action == self.n*self.n: #81
            if goGame.game.board.previous_is_pass == True:
                pre_previous_is_pass = True
            goGame.game.board.previous_is_pass = True
            goGame._switch_turns() #^^
            return (board, -player)
        move = (int(action/self.n), action%self.n) #remain
        #b.execute_move(move, player)
        goGame._place_stone(move) #^
        goGame.game.board.previous_is_pass = False
        goGame._switch_turns() #^^
        #return (b.pieces, -player)
        return (goGame.game.board, -player) #^

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize() #keep
        #b = Board(self.n)
        goGame = GameUI() #^
        #b.pieces = np.copy(board)
        goGame.game.board = board #^
        #legalMoves =  b.get_legal_moves(player)

        x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        legalMoves = [ (a,b) for a in x for b in y]
        for x, y in legalMoves:
            if goGame._place_stone((x, y)) == False:
                legalMoves.remove((x,y))
            else:
                goGame.game.board.remove_stone(x,y)

            
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
        goGame = GameUI() #^
        goGame.game.board = board #^

        #end with 2 consective passes
        # 
        #Black should win when 43:38 (5 points higher) 
        #for simplicity, whoever get 41 will win
        if (goGame.game.board.previous_is_pass and goGame.game.board.pre_previous_is_pass):
            diff = goGame.game.get_scores().get(player)>goGame.game.get_scores().get(-player)
            if diff > 0:
                return 1
            else:
                return -1
        else:
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
        goGame = GameUI() #^
        goGame.game.board = board #^
        #return b.countDiff(player)
        return goGame.game.get_scores().get(player)

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

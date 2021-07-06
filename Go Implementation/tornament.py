import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

from varname import nameof
import numpy as np
import random
import operator
from utils import *

"""
use thisss script to play any two agents against each other, or play manually with
any agent.
"""


playouts = 300
instinctPlay = False

g = OthelloGame(5)

# all players
rp = RandomPlayer(g).play
#gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play


def createNetPlayer(tarName, sim=2, cpuct=1.1):
    n = NNet(g)
    n.load_checkpoint('./temp/', tarName)
    args = dotdict({'numMCTSSims': sim, 'cpuct':cpuct})
    mcts = MCTS(g, n, args)
    player = lambda x: np.argmax(mcts.getActionProb(x, temp=0, instinctPlay=instinctPlay)[0])
    return player

def playGame(player1, str1, player2, str2):
    arena = Arena.Arena(player1, player2, g)
    x, y, z, xb = arena.playGames(playouts, verbose=False)
    print(str1, " win: ", x)
    print(str2, " win: ", y)
    print(str1, " win black: ", xb)
    return x, y
#playGame(best1, 'best', best2, 'best2')

def tournament(playList):
    tournamentResult = dict.fromkeys(playList, 0)
    for a in playList:
        for b in playList:
            if a is not b:
                aWin, bWin = playGame(a, 'p1', b, 'p2')           
                tournamentResult[a] += (aWin - bWin + playouts)/2
                tournamentResult[b] += (bWin - aWin + playouts)/2
    print (
    '''
    く__,.ヘヽ.　　　　/　,ー､ 〉
    　　　＼ , !-─‐-i　/　/´
    　　　 　 ／｀ｰ　　　 L/／｀ヽ､
    　　 　 /　 ／,　 /|　 ,　 ,　　    ,
    　　　ｲ 　/ /-‐/　ｉ　L_ ﾊ ヽ!　 i
    　　　 ﾚ ﾍ 7ｲ｀ﾄ　 ﾚ-ﾄ､!ハ|　 |
    　　　　 !,/7 ✪　　 ´i✪ﾊiソ| 　 |　　　
    　　　　 |.从　　_　　 ,,,, / |./ 　 |
    　　　　 ﾚ| i＞.､,,__　_,.イ / 　.i 　|
    　　　 ﾚ| | / k_７_/ﾚヽ,　ﾊ.　|
    　　　 | |/i 〈|/　 i　,.ﾍ |　i　|
    　　　.|/ /　ｉ： 　 ﾍ!　　＼　|
    　　　 　 　 kヽ､ﾊ 　 _,.ﾍ､ 　 /､!
    　　　 !〈//｀Ｔ´, ＼ ｀7ｰr
    　　　 ﾚヽL__|___i,___,ンﾚ|ノ
    　　　 　　　ﾄ-,/　|___./
    　　　 　　　ｰ　　!_,.:
    '''
    )
    

    return list(tournamentResult.values())
best = createNetPlayer("best.pth.tar")
def_27 = createNetPlayer("def_27.pth.tar")
def_52 = createNetPlayer("def_52.pth.tar")
def_77 = createNetPlayer("def_77.pth.tar")
def_102 = createNetPlayer("def_102.pth.tar")
cha_27 = createNetPlayer("cha_27.pth.tar")
cha_52 = createNetPlayer("cha_52.pth.tar")
cha_77 = createNetPlayer("cha_77.pth.tar")
cha_102 = createNetPlayer("cha_102.pth.tar")
one = createNetPlayer("one.pth.tar")
five = createNetPlayer("five.pth.tar")
three = createNetPlayer("three.pth.tar")
no_noise = createNetPlayer("no_noise.pth.tar")
#playerList = [def_27, def_52, def_77, def_102, cha_27, cha_52, cha_77, cha_102, best]
playerList = [best, five]
result = tournament(playerList)
print(result)


            
               
                


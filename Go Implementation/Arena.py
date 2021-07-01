import logging

from tqdm import tqdm
import math

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)[0] == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                print("board", board)
                print("valids", valids)
                print("action", action)
                action = self.game.n*self.game.n #if action not valid, then pass
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)[0]))
            self.display(board)
        captured_sum = board._num_captured_stones[1]+board._num_captured_stones[-1]
        score_diff = self.game.getGameEnded(board, curPlayer)[1]
        return curPlayer * self.game.getGameEnded(board, curPlayer)[0], it, captured_sum, score_diff

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        oneWonOnBlack = 0
        twoWon = 0
        draws = 0
        go_stage2 = False
        sum_iters = 0
        sum_captures = 0
        sum_score_diff_squared = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult, it, captures, score_diff = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
                oneWonOnBlack +=1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            sum_iters += it
            sum_captures += captures
            sum_score_diff_squared += score_diff**2

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult, it, captures, score_diff = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            sum_iters += it
            sum_captures += captures
            sum_score_diff_squared += score_diff**2

        avg_iters = sum_iters/(num*2)
        avg_captures = sum_captures/(num*2)
        sd = math.sqrt((sum_score_diff_squared/(num*2)))
        if (avg_iters > self.game.getBoardSize()[0]**2 - self.game.getBoardSize()[0]) and (avg_captures < self.game.getBoardSize()[1]**2):
            go_stage2 = True
            if (oneWon > 2*twoWon) or (twoWon > 2*oneWon):
                go_stage2 = False            
            if 0.50*sd < 6: #if normal disribution, more than 31% of game win within 6 scores should stay stage 1
                go_stage2 = False
        print('sd: ',sd)
        print('avg capture: ', avg_captures)
        print('avg_iters: ', avg_iters)
        log.info(f'sd: "{sd}"')
        log.info(f'avg capture: "{avg_captures}"')
        log.info(f'avg_iters: "{avg_iters}"')

        return oneWon, twoWon, draws, oneWonOnBlack, go_stage2

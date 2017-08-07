"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!........
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    opp_location = game.get_player_location(game.get_opponent(player))
    if not opp_location:
        return 0.

    own_location = game.get_player_location(player)
    if not own_location:
        return 0.
    
    #reward chasing after your opponent 
    return float(-(sum(opp_location) - sum(own_location))**2)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function! 
    
    #Convert the board to a matrix you can change later.
    
    def board_to_matrix(board):
        r = np.zeros((board.height, board.width))
        p1_loc = board._board_state[-1]
        p2_loc = board._board_state[-2]

        for i in range(board.height):
            for j in range(board.width):
                idx = i + j * board.height
                if board._board_state[idx]:
                    if idx == p1_loc:
                        r[i,j] = 1
                    elif idx == p2_loc:
                        r[i,j] = 2
                    else:
                        r[i,j] = -1
        return r
    
    game_matrix= board_to_matrix(game)
    rotate_90 = list(zip(*game_matrix[::-1]))
    rotate_180 = list(zip(*rotate_90[::-1]))#is a list with 180-rotated game
    
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    #center = (game.width/2,height/2)
    
    reflect_count=1
    for move in own_moves:# [3,1]
        if move in rotate_180:
            reflect_count+=1
            
    #the more moves you can reflect the better of you are
    return float(len(own_moves)*reflect_count-len(opp_moves))
    
    

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    bad_spot = [(0, 0),
               (0, (game.width - 1)),
               ((game.height - 1), 0),
               ((game.height - 1), (game.width - 1))]
    
    own_moves = game.get_legal_moves(player)
    own_bad = [move for move in own_moves if move in bad_spot]
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_bad = [move for move in opp_moves if move in bad_spot]
    
    # Penalize/reward move count if some moves are in the corner
    return float(len(own_moves) - len(own_bad) - len(opp_moves) +  len(opp_bad))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        

        # TODO: finish this function!
        
        #decide move to take on that state for your player to follow  
        #try each move, choose the move that has max value among Min_values(forecast(game,move))
                              
        def max_value(game,depth):
            if self.time_left() < self.TIMER_THRESHOLD: #include in helper functions.
                raise SearchTimeout()
            #utility=self.score
            legal_moves = game.get_legal_moves()
            if depth==0 or not legal_moves:      #utility for not legal?
                return self.score(game,self)

            v=float("-inf")
            for a in legal_moves:
                v=max(v,min_value(game=game.forecast_move(a),depth=depth-1)) #remove self.score?
            return v
    
            
        def min_value(game,depth):
            if self.time_left() < self.TIMER_THRESHOLD: #include in helper functions.
                raise SearchTimeout()
            #utility=self.score
            legal_moves = game.get_legal_moves() #remove it?
            if depth==0 or not legal_moves:
                return self.score(game,self)

            v=float("inf")
            for a in legal_moves:
                v=min(v,max_value(game=game.forecast_move(a),depth=depth-1)) #remove self.score?
            return v
        
        legal_moves = game.get_legal_moves()
        if depth==0 or not legal_moves:
            return self.score(game,self)
        
        best_score=float("-inf")
        best_move= legal_moves[0]
        for a in legal_moves:
            score=min_value(game=game.forecast_move(a),depth=depth-1) #depth-1?
            if score>best_score:
                best_score=score
                best_move=a
        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            #iterative deepening
           
            depth=0
            while True:
                depth+=1
                best_move = self.alphabeta(game,depth)

                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                                          
             
        except SearchTimeout:
            pass


        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        # TODO: finish this function!
        
        def max_value(game,depth,alpha,beta):
            if self.time_left() < self.TIMER_THRESHOLD: #include in helper functions.
                raise SearchTimeout()
            #utility=self.score
            legal_moves = game.get_legal_moves()
            if depth<=0 or not legal_moves:
                return self.score(game,self)

            v=float("-inf")
            for a in legal_moves:
                v=max(v,min_value(game=game.forecast_move(a),depth=depth-1,alpha=alpha,beta=beta))
                if v>=beta:
                    return v
                alpha=max(alpha,v)                
            return v
                
        def min_value(game,depth,alpha,beta):
            if self.time_left() < self.TIMER_THRESHOLD: #include in helper functions.
                raise SearchTimeout()
            #utility=self.score
            legal_moves = game.get_legal_moves()
            if depth<=0 or not legal_moves:
                return self.score(game,self)

            v=float("inf")
            for a in legal_moves:
                v=min(v,max_value(game=game.forecast_move(a),depth=depth-1,alpha=alpha,beta=beta))
                if v<=alpha:
                    return v
                beta=min(beta,v)
            return v
                
        legal_moves = game.get_legal_moves()
        if depth<=0 or not legal_moves:
            return self.score(game,self)
        
        best_move = (-1,-1)
        best_score = float("-inf")
        #update alpha,beta in first search
        
        for a in legal_moves:
            v=min_value(game=game.forecast_move(a),depth=depth-1,alpha=alpha,beta=beta)
            if v>=best_score:
                best_score = v
                best_move = a
            #attempt to prune         
            if v>=beta:
                return best_move
            alpha=max(alpha,v)
            
        return best_move
    
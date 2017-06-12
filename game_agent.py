"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
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
    # Return on edge cases
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    return score_ensemble(game, player)


def within_borders(game, row, col):
    return 0 <= row < game.height and 0 <= col < game.width


def get_center_point(game):
    return game.width // 2, game.height // 2


def score_toward_center(game, player):
    player_loc = game.get_player_location(player)
    center = get_center_point(game)
    normalized_x = (abs(player_loc[0] - center[0]) - 0) / (game.width - 0)
    normalized_y = (abs(player_loc[1] - center[1]) - 0) / (game.height - 0)
    return normalized_x + normalized_y


def score_ensemble(game, player):
    return score_center(game, player) * score_toward_center(game, player)


def score_center(game, player):
    possible_directions = ((1, 2), (1, -2), (2, 1), (2, -1),
                           (-1, 2), (-1, -2), (-2, 1), (-2, -1))
    bias = 0.0
    x, y = get_center_point(game)
    points_outside = []
    for dx, dy in possible_directions:
        loc = (x + dx, y + dy)
        if within_borders(game, loc[0], loc[1]):
            points_outside.append(loc)
    player_loc = game.get_player_location(player)

    # Reward center point and the centroid around it
    if player_loc == (x, y):
        bias = 1.5
    elif player_loc in points_outside:
        bias = 0.5

    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves - opp_moves) + bias


def score_basic_eval(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Returns a score based on how many legal moves
    there are for the player and the opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves - opp_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

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

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        # initialize best move
        best_move = (-1, -1)
        if game.move_count < 2:
            return random.choice(legal_moves)

        if not legal_moves:
            return best_move

        if self.iterative:
            depth = 1
        else:
            depth = self.search_depth

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            new_move = best_move
            while True:
                if self.method == 'minimax':
                    _, new_move = self.minimax(game, depth)
                elif self.method == 'alphabeta':
                    _, new_move = self.alphabeta(game, depth)

                best_move = new_move
                if not self.iterative:
                    break
                depth += 1

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move, possible_moves = None, game.get_legal_moves()
        if depth <= 0 or not possible_moves:
            return self.score(game, self), None

        best_val = float("-inf") if maximizing_player else float("inf")
        for move in possible_moves:
            next_state = game.forecast_move(move)
            if maximizing_player:
                value, _ = self.minimax(next_state, depth - 1, False)
                if value > best_val:
                    best_val, best_move = value, move
            else:
                value, _ = self.minimax(next_state, depth - 1, True)
                if value < best_val:
                    best_val, best_move = value, move
        return best_val, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

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

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        best_move, possible_moves = None, game.get_legal_moves()
        if depth <= 0 or not possible_moves:
            return self.score(game, self), None

        best_val = float("-inf") if maximizing_player else float("inf")
        for move in possible_moves:
            next_state = game.forecast_move(move)
            if maximizing_player:
                value, _ = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                alpha = max(alpha, value)
                if value > best_val:
                    best_val, best_move = value, move
                if alpha >= beta:
                    break
            else:
                value, _ = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                beta = min(beta, value)
                if value < best_val:
                    best_val, best_move = value, move
                if alpha >= beta:
                    break
        return best_val, best_move

import torch
import random
import typing
import sys


class Minesweeper:
    """Backend for minesweeper AI. This class contains a pytorch tensor
    representation of the gamestate in order to facilitate fast interaction with
    the ML libraries.
    """

    def __init__(self) -> None:
        """Create a minesweeper game object that can create new game states for
        us.
        """

        # setting device on GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_game_state(
        self,
        x: int,
        y: int,
        mines: typing.Optional[int] = None,
        mine_rate: float = 0.20625,
    ) -> None:
        """Create a Minesweeper game from a set of parameters

        Parameters
        ----------
        x : int
            Horizontal size of gameboard
        y : int
            Vertical size of gameboard
        mines : typing.Optional[int], optional
            Number of mines that will be placed on the board. Defining this
            feature will override any value in the mine_rate argument, by
            default None
        mine_rate : float, optional
            The share of tiles that should be mines, by default 0.20625

        Raises
        ------
        ValueError
            If the board shape is illegal or if the number of mines exceeds the number of tiles
        ValueError
            _description_
        """
        # Is the game over?
        self.over = False
        # Bool for loss in a ?
        self.lost = False

        self.play_initiated = False

        # Perform some dimension checks
        if (x <= 0) or (y <= 0):
            raise ValueError(
                "Illegal shape of game board. X and Y must both be positive "
                + f"non-zero integers; X: {x} Y: {y}"
            )
        # Store our dimensions
        self.x = x
        self.y = y

        # Store the number of tiles
        self.num_tiles = x * y

        # If the user did not provide a number of mines, use the provided
        # assumed classic minesweeper expert mine density
        if mines is None:
            # Store the mine rate
            self.MINE_RATE = mine_rate
            # Calc the number of mines off the rate
            self.num_mines = int(self.num_tiles * self.MINE_RATE)
        else:
            # Store the number of mines
            self.num_mines = mines
            # Calc the mine rate off the number of mines
            self.MINE_RATE = self.num_mines / self.num_tiles

        # Make sure that we have a legal number of mines
        if self.num_mines > self.num_tiles:
            raise ValueError(
                "Number of mines cannot exceed the number of tiles on the "
                + "game board."
            )

        # Set the recursion limit to the number of tiles on the board. This is
        # becasue in worst case scenario, the user discoveres an empty board
        # with one click. The recursive discovery function must be able to
        # recurse for all tiles.
        # sys.setrecursionlimit(self.num_tiles*10)
        recursion_limit = 1000
        if self.num_tiles > recursion_limit:
            recursion_limit = self.num_tiles + 1
        sys.setrecursionlimit(recursion_limit)

        # A list for tracking updated tiles
        self.update_list: list[tuple[int, int]] = []

        # Build the data structures that hold the game state
        self.__build_game_state_tensors()

    def reinitialize_game_state(self) -> None:
        """Restart the game from the previously defined arguments"""
        self.initialize_game_state(self.x, self.y, mine_rate=self.MINE_RATE)

    def __build_game_state_tensors(self) -> None:
        """Build all of the tensors that contain the game state in the backend"""
        # All of the game-state tensors will be storing small integers. This
        # means that we can certainly save memory space by using 8bit integers.
        self.dtype = torch.float

        # Mask for if a given square has been discovered.
        self.discovery = torch.zeros([self.x, self.y], dtype=self.dtype).to(self.device)
        self.num_discovered = 0

        # Mask for if a given square has been flagged for a mine.
        self.flags = torch.zeros([self.x, self.y], dtype=self.dtype).to(self.device)
        self.num_flags = 0

        # Mask for if a given square contains a mine
        self.mines: torch.Tensor
        # Fill the mine mask
        self.__populate_mine_mask()

        # Mask for what number of mines borders this square.
        self.numbers = torch.zeros([self.x, self.y], dtype=self.dtype).to(self.device)
        self.__fill_number_mask()

        # Build a tensor that represents the front facing view of a board that a
        # player will essentially interact with; either directly or through a
        # GUI
        self.board: torch.Tensor
        self.__build_game_board()

        # Tensor that stores the changes that need to occur to the game board
        # next time we update it. Refered to here as the derivative since it
        # represents the marginal change to the game state that occurs from one
        # action.
        self.__clear_derivative()

    def __populate_mine_mask(self) -> None:
        """Determine the locations of and place the mines on the board given a
        board size and a desired number of mines.
        """

        # Do we have more mines? Or empty spaces on the board?
        if self.num_mines > self.num_tiles / 2:
            # If more mines
            num_places_to_modify = self.num_tiles - self.num_mines
            # More efficient to place holes than mines
            value_to_place = 0
        else:
            # If more empty space
            num_places_to_modify = self.num_mines
            # More efficient to place mines
            value_to_place = 1

        # Using the logic above, instantiate the mine mask with the proper
        # initial value
        self.mines = torch.full(
            [self.x, self.y],
            1 - value_to_place,
            dtype=self.dtype,
        ).to(self.device)

        # While we still have mines (or holes) to place
        while num_places_to_modify > 0:
            # Where should we put the mine (or hole)?
            x_coord = random.randint(0, self.x - 1)
            y_coord = random.randint(0, self.y - 1)

            # If there is not a mine (or hole) in that location
            if not self.mines[x_coord, y_coord] == value_to_place:
                # Place the mine (or hole)
                self.mines[x_coord, y_coord] = value_to_place
                # And note that we have placed a mine (or hole)
                num_places_to_modify -= 1

            # Else, we do nothing since we need to find another place for the
            # mine (you get it) that was not able to be placed.

    def __fill_number_mask(self) -> None:
        """Calculate the number of mines that surround each square on the board"""
        # For the entire grid
        for y in range(self.y):
            for x in range(self.x):
                # Is there a mine where we are?
                mine_here = self.mines[x, y] == 1

                if not mine_here:
                    mines_nearby: int = 0

                    # Where should we look around ourselves for mines
                    search_space = self.__get_legal_neighbors(x, y)

                    # For each legal search location
                    for location in search_space:
                        # Add the mine mask directly to the number of mines
                        # nearby
                        mines_nearby += self.mines[
                            location[0], location[1]
                        ]  # type:ignore

                    self.numbers[x, y] = mines_nearby

    def __build_game_board(self) -> None:
        """Build the game board from the existing tensors; Flag, Discovery,
        Mine, and Number. All of these tensors must be represented in one game
        board, so we need a code system that can uniquely define all of the
        states of the board.

        They are as follows:
            -5  : A flagged mine that was not a mine
            -4  : The mine that was hit (if any)
            -3  : A mine
            -2  : An undiscovered square
            -1  : A flagged square
            0-9 : A discovered square that contains a number (the number held)

        This tensor represents the game board as it will be seen by the player.
        Therefore, if for example, a mine has not been discovered, it will not
        appear on this tensor.
        """

        # Start with the flags as it is defined, but inverted
        self.board = -1 * self.flags.clone().detach()
        self.board = torch.where(self.flags == 0, -2, self.board)

        # Where the board is discovered, replace those values with the number
        self.board = torch.where(self.discovery == 1, self.numbers, self.board)

    def update_game_board(self) -> None:
        """At the current step, with all accumulated changes to the board, perform
        the appropriate changes. In practice this means that we will add the
        delta or derivative tensor to the game board.
        """
        # print(self.derivative.transpose(0, 1))
        self.board += self.derivative
        self.__clear_derivative()
        # print(self.board.transpose(0, 1))

        # When we update the game board, check to see if we have won the game
        if self.num_tiles - self.num_discovered == self.num_mines:
            self.over = True
            self.board = torch.where(self.mines == 1, -1, self.board)

        if self.lost:
            self.board = torch.where(
                (self.mines == 0) & (self.flags == 1), -5, self.board
            )

    def __get_legal_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """For a given location on the board, what are all of the tiles that are
        considered neighbors of this X, Y coordinate?

        Parameters
        ----------
        x : int
            X coordinate of the tile to check
        y : int
            Y coordinate of the tile to check

        Returns
        -------
        list[tuple[int, int]]
            A list containing all of the other tiles that are considered
            neighbors of this tile
        """
        # The neighbor set is a 3x3 grid with the center at x, y. We use these
        # offsets to iterate over the offsets to x and y in order to explore
        # that space.
        offsets = [-1, 0, 1]

        # Where we will place the neighbors as we find them
        legal_neighbors: list[tuple[int, int]] = []

        # Iterate over the same offsets for both coordinates
        for y_offset in offsets:
            for x_offset in offsets:
                # The current loc to examine
                x_curr = x + x_offset
                y_curr = y + y_offset

                if (
                    # We have broken our X bounds
                    ((x_curr < 0) or (x_curr >= self.x))
                    # We have broken our Y bounds
                    or ((y_curr < 0) or (y_curr >= self.y))
                    # We are examining ourselves
                    or ((x_offset == 0) & (y_offset == 0))
                ):
                    continue

                # If this square is allowed to be checked for a mine, add to the
                # list.
                legal_neighbors.append((x_curr, y_curr))

        return legal_neighbors

    def get_board_dimension(self) -> tuple[int, int]:
        """Return the size of the board

        Returns
        -------
        tuple[int, int]
            The board size in (x, y) format
        """
        return (self.x, self.y)

    def discover_tile(self, x: int, y: int) -> bool:
        """Given a location of a tile, flag that location as discovered on the
        game board

        Parameters
        ----------
        x : int
            X coordinate of the tile to be discoverd
        y : int
            Y coordinate of the tile to be discoverd

        Returns
        -------
        bool
            True if the tile was successfully discovered, False if for any reason
            that tile could not be discovered
        """
        if self.over:
            return False

        self.clear_update_list()

        # Perform the logic of the update
        result = self.__discover_tile_backend(x, y)

        # Update the game board using the built derivative from this step
        self.update_game_board()

        return result

    def __discover_tile_backend(self, x: int, y: int) -> bool:
        """Private backend for discovering tiles. This method performs all of the
        logic the tile discovery, but none of the updates to the game board.

        Parameters
        ----------
        x : int
            X coordinate of the tile to be discoverd
        y : int
            Y coordinate of the tile to be discoverd

        Returns
        -------
        bool
            True if the tile was successfully discovered, False if for any reason
            that tile could not be discovered
        """

        # First check if the is flagged
        flagged = self.flags[x, y]
        if flagged:
            return False

        # If the first move of the game
        if not self.play_initiated:
            # If the player did not hit a zero, or if they hit a mine on their
            # first move
            if (not self.numbers[x, y] == 0) or (self.mines[x, y]):
                # Re-initialize the game state with the flags as they were
                current_flags = self.flags
                self.reinitialize_game_state()
                self.flags = current_flags

                # Manually rebuild the game board to make sure that the flags
                # are tracked correctly
                self.__build_game_board()

                # Try again in that spot
                return self.__discover_tile_backend(x, y)
            else:
                # Else, the game starts
                self.play_initiated = True

        # If the tile is already discovered
        discovered = self.discovery[x, y]
        if discovered:
            return False

        self.discovery[x, y] = 1

        if self.mines[x, y]:
            self.over = True
            self.lost = True

            # Expose the mines that were not flagged
            self.derivative -= self.mines - torch.where(self.mines == 0, 0, self.flags)

            # An undiscovered (and unflagged) square that is clicked with a mine
            # in it needs to be set to the special -4 flag on the game board
            # [-3 (for mine) - 1 = -4]
            self.derivative[x, y] -= 1

        else:
            # +2 for discovering the square, then add the number in that square
            self.derivative[x, y] += 2 + self.numbers[x, y]

            if self.numbers[x, y] == 0:
                for neighbor in self.__get_legal_neighbors(x, y):
                    self.__discover_tile_backend(neighbor[0], neighbor[1])

        self.num_discovered += 1
        self.update_list.append((x, y))
        return True

    def test_number_tile(self, x: int, y: int) -> bool:
        """Given the location of a tile, the user is stating that all of the
        unflagged neighbors of that tile are mineless. This method tests that
        hypothesis and attempts to discover all of the tiles that neighbor the
        passed tile.

        Parameters
        ----------
        x : int
            X coordinate of the tile to be tested
        y : int
            Y coordinate of the tile to be tested

        Returns
        -------
        bool
            True if the user was correct and all of the unflagged spaces around
            the passed tile have been successfully discovered, False otherwise
        """

        if self.over or not self.discovery[x, y]:
            return False

        self.clear_update_list()

        # Run the logical backend
        result = self.__test_number_tile_backend(x, y)

        # Update the board based on the steps in the backend
        self.update_game_board()

        return result

    def __test_number_tile_backend(self, x: int, y: int) -> bool:
        """Perform the logical backend of the updates for number tile testing. Two
        criteria must be met in order for this method to return True:

        1) The passed tile must have a number of flags in its legal neighbor set
        equal to the number shown on its face

        2) There must be no mines in the other undiscoverd squares in the legal
        neighbor set

        Parameters
        ----------
        x : int
            X coordinate of the tile to be tested
        y : int
            Y coordinate of the tile to be tested

        Returns
        -------
        bool
            True if the tiles around this were successfully discoverd and the
            game is not lost, False for any other reason
        """
        # If the game is or the tile is not already discovered, then this is an
        # illegal move

        # Which neighbors are we checking in?
        neighbors = self.__get_legal_neighbors(x, y)
        # What is our current number?
        number = self.numbers[x, y]

        # Find all the flags in the number set
        number_flags = 0
        for neighbor in neighbors:
            number_flags += self.flags[neighbor[0], neighbor[1]]

        # If that number of flags matches the target
        if number_flags == number:
            for neighbor in neighbors:
                self.__discover_tile_backend(neighbor[0], neighbor[1])

            return True

        # Otherwise invalid
        return False

    def flag_tile(self, x: int, y: int) -> bool:
        """Method for placing or removing a flag on the tile specified by the user

        Parameters
        ----------
        x : int
            X coordinate of the tile to be flagged or unflagged
        y : int
            Y coordinate of the tile to be flagged or unflagged

        Returns
        -------
        bool
            True if the flagging operation was successful, False otherwise
        """

        # If the game is over, flagging is illegal
        if self.over:
            return False

        self.clear_update_list()

        # Perform the backend updates for the flagging operation
        result = self.__flag_tile_backend(x, y)

        self.update_game_board()

        return result

    def __flag_tile_backend(self, x: int, y: int) -> bool:
        """Backend method for flagging or unflagging a tile on the gameboard

        Parameters
        ----------
        x : int
            X coordinate of the tile to have a flag operation
        y : int
            Y coordinate of the tile to have a flag operation

        Returns
        -------
        bool
            True if the tile was successfully operated on, False otherwise
        """
        
        # If the square has been discovered, then we can't flag it
        if self.discovery[x, y]:
            return False

        # If the square is already flagged
        if self.flags[x, y]:
            # Unflag it
            self.flags[x, y] = 0
            self.derivative[x, y] -= 1
            self.num_flags -= 1
        else:
            # Flag it
            self.flags[x, y] = 1
            self.derivative[x, y] += 1
            self.num_flags += 1

        self.update_list.append((x, y))
        return True

    def __clear_derivative(self) -> None:
        """_summary_"""
        self.derivative = torch.zeros([self.x, self.y], dtype=self.dtype).to(
            self.device
        )

    def clear_update_list(self) -> None:
        self.update_list = []

    # def get_update_list(self) -> list[tuple[int, int]]:
    #     return self.update_list


if __name__ == "__main__":
    minesweeper = Minesweeper()

    minesweeper.initialize_game_state(x=5, y=5)

    print(minesweeper.numbers)
    print()
    print(minesweeper.mines)
    print()
    print(minesweeper.board)

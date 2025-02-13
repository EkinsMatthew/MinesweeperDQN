import torch
import random
import typing
import sys
import pygame
import math


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
            If the board shape is illegal or if the number of mines exceeds the 
            number of tiles
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
        """At the current step, with all accumulated changes to the board, 
        perform the appropriate changes. In practice this means that we will add 
        the delta or derivative tensor to the game board.
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
            True if the tile was successfully discovered, False if for any 
            reason that tile could not be discovered
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
        """Private backend for discovering tiles. This method performs all of 
        the logic the tile discovery, but none of the updates to the game board.

        Parameters
        ----------
        x : int
            X coordinate of the tile to be discoverd
        y : int
            Y coordinate of the tile to be discoverd

        Returns
        -------
        bool
            True if the tile was successfully discovered, False if for any 
            reason that tile could not be discovered
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
        """Perform the logical backend of the updates for number tile testing. 
        Two criteria must be met in order for this method to return True:

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
        """Method for placing or removing a flag on the tile specified by the 
        user

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


class MinesweeperGUI:
    def __init__(
        self,
        game: Minesweeper,
        zoom_factor: float,
        FPS: int = 60,
        tile_set_number: int = 0,
    ) -> None:
        pygame.init()

        self.game = game

        self.scale_factor = zoom_factor

        self.FPS = FPS

        assets_folder = "../assets"

        self.tile_image = pygame.image.load(f"{assets_folder}/tiles.png")
        self.tile_offset = 48 * tile_set_number

        self.smile_image = pygame.image.load(f"{assets_folder}/smiles.png")
        self.smile_frame = pygame.image.load(f"{assets_folder}/smile_frame.png")

        self.number_image = pygame.image.load(f"{assets_folder}/numbers.png")
        self.number_frame = pygame.image.load(f"{assets_folder}/number_frame.png")

        self.corner_image = pygame.image.load(f"{assets_folder}/corners.png")

        self.update_set: list[tuple[int, int]] = []
        for y in range(self.game.y):
            for x in range(self.game.x):
                self.update_set.append((x, y))

        self.__initialize_window()

        self.__start_clock()

    def __initialize_window(self):

        self.TILE_SIZE = 16

        self.field_size = (self.game.x * self.TILE_SIZE, self.game.y * self.TILE_SIZE)

        self.field_x_buffers = (11, 11)
        self.field_y_buffers = (53, 11)

        self.ui_size = (
            self.field_x_buffers[0] + self.field_size[0] + self.field_x_buffers[1],
            self.field_y_buffers[0] + self.field_size[1] + self.field_y_buffers[1],
        )

        # self.ui_x_buffers = (3, 3)
        # self.ui_y_buffers = (42, 3)
        self.ui_x_buffers = (0, 0)
        self.ui_y_buffers = (0, 0)

        self.x_buffers = tuple(
            x1 + x2 for x1, x2 in zip(self.field_x_buffers, self.ui_x_buffers)
        )
        self.y_buffers = tuple(
            y1 + y2 for y1, y2 in zip(self.field_y_buffers, self.ui_y_buffers)
        )

        # Calc the size of the window for the game in native coodinates
        self.window_size = (
            self.ui_x_buffers[0] + self.ui_size[0] + self.ui_x_buffers[1],
            self.ui_y_buffers[0] + self.ui_size[1] + self.ui_y_buffers[1],
        )

        # Convert the window size to real pixels based on the zoom factor
        self.window_size_real = tuple(x * self.scale_factor for x in self.window_size)

        # Create the screen that the game works off of
        self.screen = pygame.display.set_mode(self.window_size_real)

        self.__build_ui_frame()

        self.refresh()

    def __build_ui_frame(self) -> None:

        # Create the general UI elements
        self.ui_surface = pygame.Surface(self.ui_size)

        # The default color of the surface
        self.ui_surface.fill(pygame.Color("#444444"))

        frame_width = 2

        ui_frame = (
            (1, 1),
            tuple(x - 3 for x in self.ui_size),
        )
        info_frame = (
            (9, 9),
            (self.ui_size[0] - (9 + frame_width), 43),
        )
        field_frame = (
            (
                self.field_x_buffers[0] - frame_width,
                self.field_y_buffers[0] - frame_width,
            ),
            (
                self.field_x_buffers[0] + self.field_size[0],
                self.field_y_buffers[0] + self.field_size[1],
            ),
        )

        # Draw inner backgound
        MinesweeperGUI.__draw_rects(
            self.ui_surface,
            "#BDBDBD",
            [
                (
                    ui_frame[0][0] + frame_width,
                    ui_frame[0][1] + frame_width,
                    ui_frame[1][0] - ui_frame[0][0] - frame_width,
                    ui_frame[1][1] - ui_frame[0][1] - frame_width,
                )
            ],
        )

        self.__draw_frame(self.ui_surface, ui_frame, 2, "#FFFFFF", "#7B7B7B", 0)

        self.__draw_frame(self.ui_surface, info_frame, 2, "#7B7B7B", "#FFFFFF", 1)

        self.__draw_frame(self.ui_surface, field_frame, 2, "#7B7B7B", "#FFFFFF", 1)

        # Place number and smile frames

        info_frame_width = info_frame[1][0] - info_frame[0][0]

        number_frame_size = self.number_frame.get_size()
        mine_counter_loc = (0, 0)
        timer_loc = (0, 0)

        smile_frame_size = self.smile_frame.get_size()
        smile_frame_loc = (0, 0)

        min_full_feature_width = (
            # Two counters
            2 * number_frame_size[0]
            # One smiley
            + smile_frame_size[0]
            # 6 pixel buffer for three elements is 4 (fence post)
            + 6 * 4
        )

        if info_frame_width >= min_full_feature_width:
            mine_counter_loc = (
                info_frame[0][0] + 6 + frame_width,
                info_frame[0][1] + 4 + frame_width,
            )
            self.ui_surface.blit(self.number_frame, mine_counter_loc)

            timer_loc = (
                info_frame[1][0] - 6 - number_frame_size[0],
                info_frame[0][1] + 6,
            )
            self.ui_surface.blit(self.number_frame, timer_loc)

            smile_frame_loc = (
                info_frame[0][0] + (info_frame_width / 2) - (smile_frame_size[0] / 2),
                info_frame[0][1] + 5,
            )
            self.ui_surface.blit(self.smile_frame, smile_frame_loc)

            # Defining the timer location is conditonal on all having space for all three
            self.timer_loc = (
                timer_loc[0] + self.ui_x_buffers[0],
                timer_loc[1] + self.ui_y_buffers[0],
            )

        self.mine_counter_loc = (
            mine_counter_loc[0] + self.ui_x_buffers[0],
            mine_counter_loc[1] + self.ui_y_buffers[0],
        )

        self.smile_frame_loc = (
            smile_frame_loc[0] + self.ui_x_buffers[0],
            smile_frame_loc[1] + self.ui_y_buffers[0],
        )

        self.screen.blit(
            pygame.transform.scale_by(self.ui_surface, self.scale_factor),
            self.__native_to_drawn_coordinates(
                (self.ui_x_buffers[0], self.ui_y_buffers[0])
            ),
        )

    def __update_board(self):
        tile_map = pygame.Surface(
            size=(self.game.x * self.TILE_SIZE, self.game.y * self.TILE_SIZE)
        )

        for y in range(self.game.y):
            for x in range(self.game.x):
                art_coord = [
                    coord * self.TILE_SIZE
                    for coord in self.__get_tile_art_coordinate(x, y)
                ]

                tile_map.blit(
                    self.tile_image,
                    dest=(x * self.TILE_SIZE, y * self.TILE_SIZE),
                    area=pygame.Rect(
                        art_coord[0],
                        art_coord[1] + self.tile_offset,
                        art_coord[0] + self.TILE_SIZE,
                        art_coord[1] + self.TILE_SIZE + self.tile_offset,
                    ),
                )
        board = pygame.transform.scale_by(tile_map, self.scale_factor)

        self.screen.blit(
            source=board,
            dest=self.__native_to_drawn_coordinates(
                (self.x_buffers[0], self.y_buffers[0])
            ),
        )

    def __update_tiles(self, update_list: list[tuple[int, int]]):
        for tile_coord in update_list:
            x = tile_coord[0]
            y = tile_coord[1]

            tile = pygame.Surface(size=(self.TILE_SIZE, self.TILE_SIZE))

            art_coord = [
                coord * self.TILE_SIZE for coord in self.__get_tile_art_coordinate(x, y)
            ]

            tile.blit(
                source=self.tile_image,
                dest=(0, 0),
                area=pygame.Rect(
                    art_coord[0],
                    art_coord[1] + self.tile_offset,
                    art_coord[0] + self.TILE_SIZE,
                    art_coord[1] + self.TILE_SIZE + self.tile_offset,
                ),
            )

            tile = pygame.transform.scale_by(tile, self.scale_factor)

            self.screen.blit(
                source=tile,
                dest=self.__native_to_drawn_coordinates(
                    (
                        (x * self.TILE_SIZE) + self.x_buffers[0],
                        (y * self.TILE_SIZE) + self.y_buffers[0],
                    )
                ),
            )

    def __counter_from_number(self, n: int) -> pygame.Surface:

        numbers = MinesweeperGUI.__int_to_three_digit_display(n)

        number_tile_size = (13, 23)
        counter = pygame.Surface((number_tile_size[0] * 3, number_tile_size[1]))

        for i, number in enumerate(numbers):
            number_loc = (number % 6, number // 6)
            counter.blit(
                self.number_image,
                (i * number_tile_size[0], 0),
                (
                    number_loc[0] * number_tile_size[0],
                    number_loc[1] * number_tile_size[1],
                    number_loc[0] * number_tile_size[0] + number_tile_size[0],
                    number_loc[1] * number_tile_size[1] + number_tile_size[1],
                ),
            )
        return counter

    def __start_clock(self) -> None:
        # Our cock for frame rate and update
        self.clock = pygame.time.Clock()

    def tick(self) -> bool:
        action_this_tick = False

        # Process user inputs.
        for event in pygame.event.get():
            # Check for QUIT event
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            # Check for various key-presses
            else:
                mouse_presses = pygame.mouse.get_pressed(3)
                # The current location of the mouse
                mouse_loc = self.__drawn_to_native_coordinates(pygame.mouse.get_pos())

                normalized_coords = [
                    math.floor((mouse_loc[0] - self.x_buffers[0]) // self.TILE_SIZE),
                    math.floor((mouse_loc[1] - self.y_buffers[0]) // self.TILE_SIZE),
                ]

                action_on_board = (
                    (0 <= normalized_coords[0])
                    and (normalized_coords[0] < self.game.x)
                    and (0 <= normalized_coords[1])
                    and (normalized_coords[1] < self.game.y)
                )

                if action_on_board:
                    if mouse_presses[0]:
                        action_this_tick = self.game.discover_tile(
                            normalized_coords[0], normalized_coords[1]
                        )

                    if mouse_presses[2]:
                        action_this_tick = self.game.flag_tile(
                            normalized_coords[0], normalized_coords[1]
                        )

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        if self.game.over:
                            self.game.reinitialize_game_state()
                            action_this_tick = False
                            self.refresh()
                        elif action_on_board:
                            discovered = self.game.discovery[
                                normalized_coords[0], normalized_coords[1]
                            ]
                            if discovered:
                                action_this_tick = self.game.test_number_tile(
                                    normalized_coords[0], normalized_coords[1]
                                )
                            else:
                                action_this_tick = self.game.flag_tile(
                                    normalized_coords[0], normalized_coords[1]
                                )

                    if event.key == pygame.K_RETURN:
                        self.game.reinitialize_game_state()

                        action_this_tick = False
                        self.refresh()

        if action_this_tick:
            self.refresh()

        return action_this_tick

    def __native_to_drawn_coordinates(
        self, native_coordinate: tuple[int, int]
    ) -> tuple[int, int]:
        return tuple(x * self.scale_factor for x in native_coordinate)

    def __drawn_to_native_coordinates(
        self, drawn_coordinate: tuple[int, int]
    ) -> tuple[int, int]:
        return tuple(x // self.scale_factor for x in drawn_coordinate)

    def __draw_frame(
        self,
        surface: pygame.Surface,
        coordinates: tuple[tuple[int, int], tuple[int, int]],
        frame_width: int,
        upper_color: str,
        lower_color: str,
        corner_value: int,
    ) -> None:

        MinesweeperGUI.__draw_rects(
            surface,
            upper_color,
            [
                (
                    coordinates[0][0],
                    coordinates[0][1],
                    frame_width,
                    coordinates[1][1] - coordinates[0][1],
                ),
                (
                    coordinates[0][0],
                    coordinates[0][1],
                    coordinates[1][0] - coordinates[0][0],
                    frame_width,
                ),
            ],
        )
        MinesweeperGUI.__draw_rects(
            surface,
            lower_color,
            [
                (
                    coordinates[1][0],
                    coordinates[0][1] + frame_width,
                    frame_width,
                    coordinates[1][1] - coordinates[0][1],
                ),
                (
                    coordinates[0][0] + frame_width,
                    coordinates[1][1],
                    coordinates[1][0] - coordinates[0][0],
                    frame_width,
                ),
            ],
        )

        surface.blit(
            self.corner_image,
            (coordinates[1][0], coordinates[0][1]),
            area=(
                frame_width * corner_value,
                0,
                frame_width * (corner_value) + frame_width,
                frame_width,
            ),
        )
        surface.blit(
            self.corner_image,
            (coordinates[0][0], coordinates[1][1]),
            area=(
                frame_width * corner_value,
                0,
                frame_width * (corner_value) + frame_width,
                frame_width,
            ),
        )

        return

    def refresh(
        self,
        extra_context: typing.Optional[str] = None,
    ) -> None:
        if (len(self.game.update_list) == 0) | (self.game.over):
            self.__update_board()
        else:
            self.__update_tiles(self.game.update_list)
        # self.game.clear_update_list()

        # The current location of the mouse
        mouse_loc = pygame.mouse.get_pos()

        pygame.display.set_caption(
            "MinesweeperGUI"
            + f" // Mines: {self.game.num_mines - self.game.num_flags}"
            + f" // Over: {self.game.over}"
            + f" // Won: {self.game.over and not self.game.lost}"
            + f" // MouseLoc: {mouse_loc}"
        )

        self.screen.blit(
            pygame.transform.scale_by(
                self.__counter_from_number(self.game.num_mines - self.game.num_flags),
                self.scale_factor,
            ),
            self.__native_to_drawn_coordinates(
                tuple(x + 1 for x in self.mine_counter_loc)
            ),
        )

        # Update the display
        # if action_this_tick:
        # pygame.display.flip()
        pygame.display.update()

        # self.clock.tick(self.FPS)

    def __get_tile_art_coordinate(self, x: int, y: int) -> tuple[int, int]:
        # What does the board say is in this square?
        tile_value = int(self.game.board[x, y])

        # All non-numbers have special case number descriptors

        # Incorrectly flagged square
        if tile_value == -5:
            return (3, 2)
        # Lost mine
        if tile_value == -4:
            return (2, 2)
        # All other mines
        if tile_value == -3:
            return (1, 2)
        # Undiscovered square
        if tile_value == -2:
            return (4, 1)
        # Flagged square
        if tile_value == -1:
            return (0, 2)

        # All numbers can derive their coordinate from their value
        return (tile_value % 5, tile_value // 5)

    @staticmethod
    def __draw_rects(
        surface: pygame.Surface,
        color: str,
        rects: list[tuple[int, int, int, int]],
    ) -> None:
        for rect in rects:
            pygame.draw.rect(surface, pygame.Color(color), rect)

    @staticmethod
    def __int_to_three_digit_display(number: int) -> tuple[int, int, int]:
        
        # NOTE: In this notation, the single digit numbers represent themselves,
        # the negative symbol is represented by a 10, and the blank display is
        # represented by 11
        
        # Edges of the function
        if number > 999:
            return (9, 9, 9)
        if number < -99:
            return (11, 9, 9)
        if number == 0:
            return (11, 11, 0)
        
        # Solution that consumes the number right to left
        abs_number = abs(number)
        
        # Start with a blank display
        display = [11, 11, 11]

        # Iterate starting on the right
        i = 2
        while abs_number != 0:
            # The number in this spot
            display[i] = abs_number % 10
            # Consume the digit
            abs_number = abs_number // 10
            # Move once to the left
            i -= 1
        
        # Where to put the negative symbol if the number is below zero
        if number < 0:
            # In the middle spot for single digit negatives
            if number > -10:
                display[1] = 10
            # In the first slot for double digit negatives
            else:
                display[0] = 10
        
        return tuple(display)


def main():
    ms = Minesweeper()

    ms.initialize_game_state(9, 9, 10)

    # print(ms.numbers)
    # print(ms.mines)

    gui = MinesweeperGUI(
        ms,
        zoom_factor=3,
        FPS=1000,
        tile_set_number=2,
    )

    while True:
        gui.tick()


if __name__ == "__main__":
    main()

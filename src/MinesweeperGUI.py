import pygame
import math

from Minesweeper import Minesweeper

import typing


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

        self.TILE_SIZE = 16
        self.LEFT_BUFFER = 10
        self.RIGHT_BUFFER = 10
        self.TOP_BUFFER = 52
        self.BOTTOM_BUFFER = 10

        self.zoom_factor = zoom_factor

        self.tile_width = self.TILE_SIZE * self.zoom_factor
        self.left_offset = self.LEFT_BUFFER * self.zoom_factor
        self.right_offset = self.RIGHT_BUFFER * self.zoom_factor
        self.top_offset = self.TOP_BUFFER * self.zoom_factor
        self.bottom_offset = self.BOTTOM_BUFFER * self.zoom_factor

        self.FPS = FPS

        assets_folder = "../assets"

        self.tile_image = pygame.image.load(f"{assets_folder}/tiles.png")
        self.tile_offset = 48 * tile_set_number

        self.smile_image = pygame.image.load(f"{assets_folder}/smiles.png")
        self.smile_frame = pygame.image.load(f"{assets_folder}/smile_frame.png")

        self.number_image = pygame.image.load(f"{assets_folder}/numbers.png")
        self.number_frame = pygame.image.load(f"{assets_folder}/number_frame.png")

        self.update_set: list[tuple[int, int]] = []
        for y in range(self.game.y):
            for x in range(self.game.x):
                self.update_set.append((x, y))

        self.__initialize_window()

        self.__start_clock()

    def __initialize_window(self):
        self.window_size = (
            self.game.x * self.tile_width + self.left_offset + self.right_offset,
            self.game.y * self.tile_width + self.top_offset + self.bottom_offset,
        )

        self.screen = pygame.display.set_mode(self.window_size)

        self.refresh()

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
        board = pygame.transform.scale_by(tile_map, self.zoom_factor)

        self.screen.blit(source=board, dest=(self.left_offset, self.top_offset))

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

            tile = pygame.transform.scale_by(tile, self.zoom_factor)

            self.screen.blit(
                source=tile,
                dest=(
                    (x * self.tile_width) + self.left_offset,
                    (y * self.tile_width) + self.top_offset,
                ),
            )

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
                mouse_loc = pygame.mouse.get_pos()

                normalized_coords = [
                    math.floor((mouse_loc[0] - self.left_offset) / self.tile_width),
                    math.floor((mouse_loc[1] - self.top_offset) / self.tile_width),
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

        # Update the display
        # if action_this_tick:
        # pygame.display.flip()
        pygame.display.update()

        # self.clock.tick(self.FPS)

    def __get_tile_art_coordinate(self, x: int, y: int):
        art_value = int(self.game.board[x, y])

        # Incorrectly flagged square
        if art_value == -5:
            return (3, 2)
        # Lost mine
        if art_value == -4:
            return (2, 2)
        # All other mines
        if art_value == -3:
            return (1, 2)
        if art_value == -2:
            return (4, 1)
        if art_value == -1:
            return (0, 2)

        return (art_value % 5, int(art_value / 5))

    # def get_tile_art_coordinate(self, x, y):
    #     discover = self.game.discovery[x, y]
    #     flag = self.game.flags[x, y]
    #     mine = self.game.mines[x, y]

    #     if not discover:
    #         if flag:
    #             if self.game.lost:
    #                 if not mine:
    #                     return (3, 2)
    #             return (0, 2)
    #         if mine & self.game.lost:
    #             return (1, 2)
    #         return (4, 1)
    #     else:
    #         if mine:
    #             return (2, 2)
    #         else:
    #             number: int = self.game.numbers[x, y]  # type:ignore
    #             return (number % 5, int(number / 5))


def main():
    ms = Minesweeper()

    ms.initialize_game_state(9, 9, 10)

    # print(ms.numbers)
    # print(ms.mines)

    gui = MinesweeperGUI(
        ms,
        zoom_factor=5,
        FPS=1000,
        tile_set_number=2,
    )

    while True:
        gui.tick()


if __name__ == "__main__":
    main()

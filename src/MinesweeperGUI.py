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

        self.ui_x_buffers = (3, 3)
        self.ui_y_buffers = (42, 3)
        # self.ui_x_buffers = (0, 0)
        # self.ui_y_buffers = (0, 0)

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
            tuple(x + 1 for x in (0, 0)),
            tuple(x - 3 for x in self.ui_size),
        )
        info_frame = (
            tuple(x + 9 for x in (0, 0)),
            (self.ui_size[0] - 11, 43),
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

    @staticmethod
    def __draw_rects(
        surface: pygame.Surface,
        color: str,
        rects: list[tuple[int, int, int, int]],
    ) -> None:
        for rect in rects:
            pygame.draw.rect(surface, pygame.Color(color), rect)


def main():
    ms = Minesweeper()

    ms.initialize_game_state(30, 16, 99)

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

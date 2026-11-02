"""Pure-Python H01 board and seeded move sampling helpers.

The Isaac environment owns physical poses and contact truth.  This module only
describes the deterministic Xiangqi starting layout and the task's random move
contract so tests and planner clients can use exactly the same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


H01_ROWS = 10
H01_COLS = 9
H01_REGION_ROWS = 2
H01_REGION_COLS = 3


class H01Stage(IntEnum):
    READY = 0
    FIRST_GRASPED = 1
    FIRST_WAYPOINT = 2
    FIRST_PLACED = 3
    FIRST_RELEASED = 4
    SECOND_GRASPED = 5
    SECOND_WAYPOINT = 6
    SECOND_PLACED = 7
    RELEASED = 8
    COMPLETE = 9

    # Compatibility aliases for planner clients that used the original
    # one-piece H01 stage names.
    SOURCE_GRASPED = FIRST_GRASPED
    RAISED_WAYPOINT = FIRST_WAYPOINT
    TARGET_PLACED = FIRST_PLACED


class H01MoveMode(IntEnum):
    """Whether the sampled episode performs or retracts a Xiangqi move."""

    FORWARD = 0
    UNDO = 1


class H01TableLayout(IntEnum):
    """Initial/final relationship between the selected piece and red zone."""

    EMPTY = 0
    INSIDE = 1
    TOP_OF_PILE = 2
    OUTSIDE = 3


@dataclass(frozen=True)
class H01Piece:
    """A piece in the standard Xiangqi starting position."""

    name: str
    side: str
    kind: str
    row: int
    col: int

    @property
    def square(self) -> tuple[int, int]:
        return self.row, self.col


@dataclass(frozen=True)
class H01MoveSpec:
    """Sampled source/target move exposed to policies and recording."""

    source: tuple[int, int]
    target: tuple[int, int]
    source_piece: str
    target_piece: str | None
    capture: bool
    target_region: int
    mode: H01MoveMode = H01MoveMode.FORWARD
    table_layout: H01TableLayout = H01TableLayout.EMPTY

    @property
    def undo(self) -> bool:
        return self.mode == H01MoveMode.UNDO

    @property
    def requires_second_move(self) -> bool:
        return bool(self.capture)

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.source[0],
                self.source[1],
                self.target[0],
                self.target[1],
                float(self.target_region),
                float(self.capture),
                float(self.mode),
                float(self.table_layout),
            ],
            dtype=np.float64,
        )


def standard_pieces() -> tuple[H01Piece, ...]:
    """Return the 32-piece standard Xiangqi layout.

    Red is the near/operator side (rows 6-9); black is the far side (rows
    0-3).  Names are stable and map one-to-one to MJCF body names.
    """

    back = ("chariot", "horse", "elephant", "advisor", "general", "advisor", "elephant", "horse", "chariot")
    pieces: list[H01Piece] = []
    for side, rows in (("black", (0, 2, 3)), ("red", (9, 7, 6))):
        back_row, cannon_row, soldier_row = rows
        for col, kind in enumerate(back):
            pieces.append(H01Piece(f"h01_{side}_{kind}_{col}", side, kind, back_row, col))
        for col in (1, 7):
            pieces.append(H01Piece(f"h01_{side}_cannon_{col}", side, "cannon", cannon_row, col))
        for col in (0, 2, 4, 6, 8):
            pieces.append(H01Piece(f"h01_{side}_soldier_{col}", side, "soldier", soldier_row, col))
    return tuple(pieces)


class H01BoardSampler:
    """Seeded board-wide sampler for forward and capture-undo episodes."""

    def __init__(self, rng: Any) -> None:
        self.rng = rng
        self.pieces = standard_pieces()
        self.by_square = {piece.square: piece for piece in self.pieces}

    def _index(self, upper: int) -> int:
        """Draw an integer from either RandomState or Generator."""

        draw = (
            self.rng.integers(0, upper)
            if hasattr(self.rng, "integers")
            else self.rng.randint(0, upper)
        )
        return int(draw)

    @staticmethod
    def region_for_square(row: int, col: int) -> int:
        if not (0 <= row < H01_ROWS and 0 <= col < H01_COLS):
            raise ValueError(f"invalid Xiangqi square {(row, col)}")
        row_bin = min(row // H01_REGION_ROWS, 4)
        col_bin = min(col // H01_REGION_COLS, 2)
        return row_bin * H01_REGION_COLS + col_bin

    def sample(self) -> H01MoveSpec:
        # Keep empty forward moves, captures, and undo-capture episodes equally
        # reachable.  Undo starts from the physical result of a capture: the
        # moving piece occupies the victim's square and the victim is on the
        # table, so the operator first retracts the mover and then restores the
        # victim.
        variant = self._index(3)
        mode = H01MoveMode.UNDO if variant == 2 else H01MoveMode.FORWARD
        require_capture = variant in {1, 2}
        source_piece = self.pieces[self._index(len(self.pieces))]
        source = source_piece.square
        # Choose the target region first, then a square in that region. This
        # keeps all board areas reachable over repeated seeded episodes.
        target_region = self._index(15)
        region_squares = [
            (row, col)
            for row in range(H01_ROWS)
            for col in range(H01_COLS)
            if self.region_for_square(row, col) == target_region
            and (row, col) != source
            and (
                (
                    require_capture
                    and (row, col) in self.by_square
                    and self.by_square[(row, col)].side != source_piece.side
                )
                or (not require_capture and (row, col) not in self.by_square)
            )
        ]
        if not region_squares:
            # A source can occupy the only eligible square in a tiny region;
            # fail over to a uniformly sampled eligible square on the board.
            region_squares = [
                (row, col)
                for row in range(H01_ROWS)
                for col in range(H01_COLS)
                if (row, col) != source
                and (
                    (
                        require_capture
                        and (row, col) in self.by_square
                        and self.by_square[(row, col)].side != source_piece.side
                    )
                    or (not require_capture and (row, col) not in self.by_square)
                )
            ]
            target_region = self.region_for_square(*region_squares[0])
        target = region_squares[self._index(len(region_squares))]
        target_piece = self.by_square.get(target)
        if mode == H01MoveMode.UNDO:
            table_layout = (
                H01TableLayout.INSIDE,
                H01TableLayout.TOP_OF_PILE,
                H01TableLayout.OUTSIDE,
            )[self._index(3)]
        elif target_piece is not None and self._index(2):
            table_layout = H01TableLayout.TOP_OF_PILE
        else:
            table_layout = H01TableLayout.EMPTY
        return H01MoveSpec(
            source=source,
            target=target,
            source_piece=source_piece.name,
            target_piece=target_piece.name if target_piece is not None else None,
            capture=target_piece is not None,
            target_region=self.region_for_square(*target),
            mode=mode,
            table_layout=table_layout,
        )


__all__ = [
    "H01BoardSampler",
    "H01_COLS",
    "H01MoveSpec",
    "H01MoveMode",
    "H01Piece",
    "H01_ROWS",
    "H01Stage",
    "H01TableLayout",
    "standard_pieces",
]

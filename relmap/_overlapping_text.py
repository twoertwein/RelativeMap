import math
from collections.abc import Callable, Sequence
from functools import cache
from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from shapely import Polygon, STRtree, box, get_coordinates, set_coordinates


def symmetrify(
    function: Callable[[Polygon, Polygon, bool], np.ndarray],
) -> Callable[[Polygon, Polygon, bool], np.ndarray]:
    """Wrapper to help cache `minimal_translation_vector`."""

    def symmetric(
        box_a: Polygon, box_b: Polygon, second_smallest: bool = False
    ) -> np.ndarray:
        reverse = hash(box_a) > hash(box_b)
        if reverse:
            box_a, box_b = box_b, box_a
        vectors = function(box_a, box_b, second_smallest)

        if reverse:
            return -vectors
        return vectors

    return symmetric


@symmetrify
@cache
def minimal_translation_vector(
    box_a: Polygon, box_b: Polygon, second_smallest: bool = False
) -> np.ndarray:
    """Find a minimal vector to move a box so that it doesn't overlap with another box anymore.

    Args:
        box_a:
            The moveable rotated rectangle.
        box_b:
            The fixed rotated rectangle.
        second_smallest:
            Find the slightly less minimal vector.

    Returns:
        Vectors by which to move box_a.
    """
    # get bounding box points (the last point is a repetition of the first point)
    box_a = get_coordinates(box_a)[:-1]
    box_b = get_coordinates(box_b)[:-1]
    # only need the first two edges, the other two are repetitions
    edges = np.concatenate([np.diff(box_a[:3], axis=0), np.diff(box_b[:3], axis=0)])

    # normals of edges are the potentials axes along which to move box_a
    axes = edges[:, [1, 0]] * [-1, 1] / np.linalg.norm(edges, axis=1, keepdims=True)

    # calculate overlapping area
    projections_a = box_a @ axes.T
    projections_b = box_b @ axes.T

    max_a = projections_a.max(axis=0)
    min_b = projections_b.min(axis=0)
    if (max_a < min_b).any():
        # does not overlap
        return np.empty((0, 2))
    min_a = projections_a.min(axis=0)
    max_b = projections_b.max(axis=0)
    if (max_b < min_a).any():
        # does not overlap
        return np.empty((0, 2))
    # not enclosed
    overlaps = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    fully_contained = np.logical_or(
        np.logical_and(max_a > max_b, min_a < min_b),
        np.logical_and(max_a < max_b, min_a > min_b),
    )
    overlaps[fully_contained] = np.minimum(max_a - min_b, max_b - min_a)[
        fully_contained
    ]

    # determine all vectors that have the same minimal overlap
    keep = overlaps - overlaps.min() < 1e-6
    if second_smallest and not keep.all():
        overlaps = overlaps[~keep]
        axes = axes[~keep]
        keep = overlaps - overlaps.min() < 1e-6
    overlaps = overlaps[keep]
    axes = axes[keep]

    # swap sign
    direction = box_a.mean(axis=0) - box_b.mean(axis=0)
    axes[axes @ direction < 0] *= -1

    # remove duplicate axes
    keep = ~np.triu(np.abs(axes[:, None] - axes[None]).sum(axis=-1) < 1e-6, k=1).any(
        axis=0
    )

    # don't alter the ndarray as it is cached
    axes = axes[keep] * overlaps[keep, None]
    axes.flags.writeable = False
    return axes


def gready_shifts(
    moveable_boxes: list[Polygon], fixed_boxes: list[Polygon], max_iterations: int = 150
) -> np.ndarray:
    """Shifts movable rotated rectangles to minimize overlaps.

    Note:
        All boxes have to be rotated bounding rectangles.
        This is just an iterative heuristic.

    Args:
        moveable_boxes:
            Initial location of moveable boxes.
        fixed_boxes:
            Locations of fixed boxes.
        max_iterations:
            The maximum number of steps.

    Returns:
        The shifts of the moveable boxes.
    """
    shifts = np.zeros((len(moveable_boxes), 2))
    fixed_tree = STRtree(fixed_boxes)

    # fallback to the second smallest translation vector for boxes that seem to oscelate between locations
    tried_second_smallest_n_iter_ago = np.zeros(len(moveable_boxes), dtype=int)

    for _ in range(max_iterations):
        shifts_norm = np.linalg.norm(shifts, axis=1, keepdims=True).clip(min=1e-9)
        tried_second_smallest_n_iter_ago += 1

        # move boxes: setting coordinates is ~3x faster than shapely.affine.translate
        current_boxes = [
            set_coordinates(box, get_coordinates(box) + shift)
            for box, shift in zip(moveable_boxes, shifts, strict=True)
        ]
        all_boxes = current_boxes + fixed_boxes

        # Use STRtree to efficiently find potentially overlapping boxes
        tree = STRtree(current_boxes)

        # For each movable polygon, find what it's overlapping with (vectorized query is much faster)
        moveable_intersections = tree.query(current_boxes, predicate="intersects")
        moveable_intersections = moveable_intersections[
            :, moveable_intersections[0] != moveable_intersections[1]
        ]
        fixed_intersections = fixed_tree.query(current_boxes, predicate="intersects")
        fixed_intersections[1] += len(moveable_boxes)
        intersections = np.concatenate(
            [moveable_intersections.T, fixed_intersections.T]
        )
        # group_by in numpy
        intersections = intersections[intersections[:, 0].argsort()]
        moveable_box_indices, unique_indices = np.unique(
            intersections[:, 0], return_index=True
        )
        intersecting_with = np.split(intersections[:, 1], unique_indices[1:])
        for ibox, boxes_indices in zip(
            moveable_box_indices, intersecting_with, strict=True
        ):
            box = current_boxes[ibox]
            mtvs = []
            second_smallest = tried_second_smallest_n_iter_ago[ibox] == 0

            for overlapping_box_index in boxes_indices:
                mtv = minimal_translation_vector(
                    box, all_boxes[overlapping_box_index], second_smallest
                )
                if overlapping_box_index < len(moveable_boxes):
                    # the other moveable box will also be moved
                    mtv = mtv / 2
                mtvs.append(mtv)

            mtvs = np.concatenate(mtvs)
            mtvs_norm = np.linalg.norm(mtvs, axis=1, keepdims=True)
            keep = mtvs_norm[:, 0] > 1e-8
            mtvs = mtvs[keep]
            mtvs_norm = mtvs_norm[keep]
            if mtvs.shape[0] == 0:
                continue
            if mtvs.shape[0] == 1:
                # try to escape oscelating
                if (
                    tried_second_smallest_n_iter_ago[ibox] > 10
                    and (shifts[ibox] + mtvs[0])
                    @ shifts[ibox]
                    / shifts_norm[ibox]
                    / np.linalg.norm(shifts[ibox] + mtvs)
                    < -0.5
                ):
                    tried_second_smallest_n_iter_ago[ibox] = -1
                shifts[ibox] += mtvs[0]
                continue

            # move in the most common direction
            directions = mtvs / mtvs_norm
            votes = (directions @ directions.T).sum(axis=0)
            peaks = votes - votes.max() > -1e-8
            mtvs = mtvs[peaks]
            mtvs_norm = mtvs_norm[peaks]
            if mtvs.shape[0] == 1:
                shifts[ibox] += mtvs[0]
                continue
            # if we haven't moved, prefer the largest move
            if shifts_norm[ibox] < 1e-8:
                shifts[ibox] += mtvs[mtvs_norm.argmax()]
                continue
            # remove directions that undo all of the existing walking
            dot_similarity = (shifts[ibox] + mtvs) @ shifts[ibox].T
            keep = dot_similarity >= 0
            if keep.any():
                mtvs = mtvs[keep]
                dot_similarity = dot_similarity[keep]
            # but prefer new directions (this also helps not too move too far)
            shifts[ibox] += mtvs[np.argmin(dot_similarity)]

    return shifts


def artists_to_boxes(
    artists: Sequence[Artist], *, decompose: bool, figure: plt.Figure, axes: plt.Axes
) -> list[Polygon]:
    """Converts artists to rotated bounding boxes.

    Args:
        artists:
            List of artists.
        decompose:
            Decompose artists into smaller artists.
        figure:
            Matplotlib figure containing the artists.
        axes:
            Matplotlib axes containing the artists.

    Returns:
        List of rotated bounding boxes in display coordinate systems.
    """
    points_to_pixels = figure.get_dpi() / 72.0

    boxes = []
    for artist in artists:
        match artist:
            case PathCollection() if decompose:
                offsets = artist.get_offsets()
                sizes = artist.get_sizes()
                if sizes.size == 0:
                    sizes = np.array([plt.rcParams["lines.markersize"] ** 2])
                if sizes.size == 1:
                    sizes = np.repeat(sizes, len(offsets))

                boxes.extend(
                    _point_to_box(offset, size, axes, points_to_pixels)
                    for offset, size in zip(offsets, sizes, strict=True)
                )

            case Line2D() if decompose:
                points = artist.get_xydata()
                linewidth = artist.get_linewidth()
                boxes.extend(
                    _line_to_box(start, end, linewidth, axes, points_to_pixels)
                    for start, end in pairwise(points)
                    if not np.all(start == end)
                )

            case Text():
                boxes.append(_text_to_box(artist, figure.canvas.get_renderer()))  # ty: ignore

            case _:
                raise ValueError(f"Artist of type {type(artist)} is not supported.")
    return boxes


def _point_to_box(
    coords: np.ndarray, size: float, axes: plt.Axes, points_to_pixels: float
) -> Polygon:
    # convert to display coordinates
    x, y = axes.transData.transform(coords)
    radius = (math.sqrt(size) * points_to_pixels) / 2.0
    return box(x - radius, y - radius, x + radius, y + radius)


def _line_to_box(
    start: np.ndarray,
    end: np.ndarray,
    linewidth: float,
    axes: plt.Axes,
    points_to_pixels: float,
) -> Polygon:
    # convert to display coordinates
    start = axes.transData.transform(start)
    end = axes.transData.transform(end)
    # Linewidth is in points. Convert to pixels.
    linewidth = linewidth * points_to_pixels

    vector = end - start
    direction = vector / np.linalg.norm(vector)
    # Perpendicular vector for creating the rectangle width
    normal = np.array([-direction[1], direction[0]]) * linewidth / 2.0
    return Polygon([start + normal, end + normal, end - normal, start - normal])


def _text_to_box(text: Text, renderer: Any) -> Polygon:
    # un-rotate the text to determine its with
    rotation = text.get_rotation()
    text.set_rotation(0)
    box = text.get_window_extent(renderer)
    width, height = box.width, box.height
    text.set_rotation(rotation)

    # determine the center
    box = text.get_window_extent(renderer)
    center = np.array([box.x0 + box.width / 2, box.y0 + box.height / 2])

    # Define corners of the un-rotated box around the origin.
    # build bounding box: corner @ rotation + center
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    rotation = math.radians(rotation)
    cos_rotation = math.cos(rotation)
    sin_rotation = math.sin(rotation)
    rotation = np.array([[cos_rotation, sin_rotation], [-sin_rotation, cos_rotation]])
    return Polygon(corners @ rotation + center)


def shift_artist(artist: Text, shift: np.ndarray, axes: plt.Axes) -> None:
    """Moves a matplotlib artist by a delta in display coordinates.

    Args:
        artist:
            The matplotlib artist to move. Only tested for Text artists.
        shift:
            x and y shift in display coordinates.
        axes:
            The matplotlib axes containing the artist.
    """
    position = axes.transData.transform(artist.get_position())
    new_position = axes.transData.inverted().transform(position + shift)
    artist.set_position(new_position)  # ty: ignore


def minimize_overlaps(
    figure: Figure,
    axes: plt.Axes,
    moveable_artists: Sequence[Artist],
    fixed_artists: Sequence[Artist],
    iterations: int,
) -> None:
    """Adjusts the positions of moveable_artists to minimize overlaps.

    Args:
        figure:
            The matplotlib figure containing the artists.
        axes:
            The matplotlib axes containing the artists.
        moveable_artists:
            A sequence of non-composable artists to be moved.
        fixed_artists:
            A sequence of static obstacle artists.
        iterations:
            Run the heuristic for n iterations.
    """
    figure.canvas.draw()

    fixed_boxes = artists_to_boxes(
        fixed_artists, decompose=True, figure=figure, axes=axes
    )
    moveable_boxes = artists_to_boxes(
        moveable_artists, decompose=False, figure=figure, axes=axes
    )

    shifts = gready_shifts(moveable_boxes, fixed_boxes, max_iterations=iterations)
    for artist, shift in zip(moveable_artists, shifts, strict=True):
        shift_artist(artist, shift=shift, axes=axes)

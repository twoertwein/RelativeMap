"""Create a map from (relative) node constraints."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from cvxopt import blas, matrix, solvers, sparse, spmatrix

from relmap import _sensitivity

if TYPE_CHECKING:
    from matplotlib.pyplot import Figure

_HashableT = TypeVar("_HashableT", bound=Hashable)


class RelativeMap:
    r"""Create a map from node and edge constraints.

    Note:
        Depending on the constraints, there might be multiple solutions.

    Examples:
        >>> from relmap import RelativeMap
        >>> r'''
                       C (?, ?)
                      / \ 
        angle: 30    /   \ angle: 330
        distance: ? /     \ distance: ?
                   /       \ 
           (?, ?) A---------B (?, ?)
                 angle: 90 (west)
                 distance: 2
        '''
        >>> map = RelativeMap(
        ...     {
        ...         ("A", "B"): {"angle": 90, "min_distance": 2, "max_distance": 2},
        ...         ("A", "C"): {"angle": 30},
        ...         ("B", "C"): {"angle": 330},
        ...     },
        ...     angle_unit="degree+clockwise",
        ... )
        >>> map.x  # inferred locations
        <3x2 matrix, tc='d'>
        >>> map.node_to_index  # mapping from node to index in map.x
        {'C': 0, 'B': 1, 'A': 2}
        >>> map.stableish_edges  # edges that are likely stable
        frozenset({('C', 'A'), ('B', 'A'), ('C', 'B')})
        >>> map.plot()  # plot map with matplotlib
        <Figure size 1500x1500 with 1 Axes>
    """

    def __init__(
        self,
        edge_constraints: Mapping[
            tuple[_HashableT, _HashableT],
            dict[Literal["angle", "min_distance", "max_distance"], float],
        ],
        *,
        node_constraints: Mapping[_HashableT, tuple[float, float]] | None = None,
        min_edge_distance: float = 0.01,
        angle_unit: Literal["degree", "radian", "degree+clockwise"] = "degree",
        stable_distance: float = 0.1,
        **cvxopt_options: Any,
    ) -> None:
        """Find node locations that best fit the constraints.

        Args:
            edge_constraints:
                Constraints between two nodes `(from_node, to_node)`. Possible constraints are:
                    "angle": the direction relative to north to get from `from_node` to `to_node`.
                    "min_distance": the minimal distance between the two nodes (requires `"angle"` to be specified as well`)
                    "max_distance": the maximal distance between the two nodes
                If constraints for `(a, b)` are provident, constraints for `(b, a)` don't need to be provided.
            node_constraints:
                The absolute position of nodes. These positions are assumed to be error-free. If you know the rough location of a node, create a dummy node with an exact location and use the "max_distance" constraint between the dummy node and the actual node.
            min_edge_distance:
                The minimum distance of edges that have an "angle" constraint. This is needed to prevent node locations from collapsing to one point.
            angle_unit:
                The unit of the angle constraints. Options are: "degree" (default, counter-clockwise), "degree+clockwise", "radian" (counter-clockwise).
            stable_distance:
                Edges included in `stableish_edges` can likely move less than this distance when one of the two nodes is fixed.
            cvxopt_options:
                Keyword arguments added to `cvxopt.solvers.options`.
        """
        edge_constraints = dict(deepcopy(edge_constraints))
        node_constraints = node_constraints or {}
        for (from_node, to_node), constraints in tuple(edge_constraints.items()):
            # remove constraints between two located nodes
            if from_node in node_constraints and to_node in node_constraints:
                edge_constraints.pop((from_node, to_node))

            # remove distances that are not positive
            for distance in ("min_distance", "max_distance"):
                if constraints.get(distance, 1) <= 0:
                    constraints.pop(distance)
            if "angle" in constraints:
                # always need a minimum distance for angle constraints
                constraints["min_distance"] = max(
                    min_edge_distance, constraints.get("min_distance", 0.0)
                )
            elif "min_distance" in constraints:
                raise ValueError(
                    "'min_distance' is only supported when an 'angle' is specified"
                )
            if not constraints:
                edge_constraints.pop((from_node, to_node))
                continue

            # raise if min_distance is larger than max_distance
            if constraints.get("min_distance", 0.0) > constraints.get(
                "max_distance", float("inf")
            ):
                raise ValueError(
                    f"Minimum distance is larger then maximum distance for {(from_node, to_node)}"
                )

            # merge a->b and b->a and then delete one of them
            reverse_constraints = edge_constraints.get((to_node, from_node), {})
            if not reverse_constraints:
                continue
            for constraint_type, constraint_value in constraints.items():
                if constraint_type == "angle":
                    constraint_value *= -1
                reverse_constraints[constraint_type] = constraint_value
            edge_constraints.pop((from_node, to_node))

        # raise if a located node has no edge constraint (irrelevant for the optimization)
        nodes = {node for pair in edge_constraints for node in pair}
        for node in node_constraints:
            if node not in nodes:
                raise ValueError(f"Located node '{node}' has no edge constraints")

        # convert angles to counter-clockwise radians
        if "degree" in angle_unit:
            clockwise = -1 if "clockwise" in angle_unit else 1
            edge_constraints = {
                edge: constraints
                | (
                    {"angle": math.radians(constraints["angle"] * clockwise)}
                    if "angle" in constraints
                    else {}
                )
                for edge, constraints in edge_constraints.items()
            }

        self._edge_constraints = edge_constraints
        self._node_constraints = node_constraints
        self._angle_unit = angle_unit
        self._stable_distance = stable_distance

        self._nodes = tuple(nodes)
        self._node_to_index = {node: inode for inode, node in enumerate(self._nodes)}

        solvers.options |= {"show_progress": False} | cvxopt_options
        self._solve()

    @property
    def x(self) -> matrix:
        """N x 2 matrix of node locations that satisfy the constraints."""
        return matrix(self._x)

    @property
    def y(self) -> float:
        """The objective value of the optimization problem.

        Lower is better, ideally 0.0. It roughly can be interpreted as the squared sum of
        Euclidean distances so that all nodes fulfil the angle constraints.
        """
        return self._y

    @property
    def node_to_index(self) -> Mapping[Hashable, int]:
        """Map node names to indices in `x`."""
        return self._node_to_index

    @property
    def stableish_edges(self) -> frozenset[tuple[Hashable, Hashable]]:
        """Set of edges (node pairs) that are likely stable.

        If you are at one of the two nodes, you can likely trust the angle and
        distance of the edge to get to the other node. Technically: If one of the
        nodes is fixed, the other node can likely not move more than `stable_distance`
        (and the reverse).

        To reduce the risk of false positives (claiming the edge is stable but it is not),
        reduce the value of `stable_distance`.
        """
        return self._stableish_edges

    def __repr__(self) -> str:
        """Print a quick summary of the object."""
        return f"{self.__class__.__name__}\n x: {self.x!r}\n stableish_edges: {self.stableish_edges}"

    def _solve(self) -> None:
        # directed distances (linear constraints)
        directed_edges = [
            edge
            for edge, constraints in self._edge_constraints.items()
            if "angle" in constraints
        ]
        # objective: Ax=0 (later transformed into the quadratic objective)
        # constraints: Gx<=h
        a_objective, b_objective, g_constraints, h_constraints, keep_variables = (
            self._directed_distances(directed_edges)
        )

        # undirected max distances (cone constraints)
        undirected_edges = [
            edge
            for edge, constraints in self._edge_constraints.items()
            if "angle" not in constraints
        ]
        g_cone_constraints, h_cone_constraints = self._undirected_distances(
            undirected_edges, a_objective.size[1]
        )
        dims = {
            "l": g_constraints.size[0],
            "q": [cone.size[0] for cone in g_cone_constraints],
            "s": [],
        }
        g_constraints = sparse([g_constraints, *g_cone_constraints])
        h_constraints = matrix([h_constraints, *h_cone_constraints])
        a_objective = a_objective[:, keep_variables]
        g_constraints = g_constraints[:, keep_variables]

        # solve problem
        solution, a_objective, g_constraints, h_constraints = (
            self._solve_with_fixed_nodes(
                a_objective, b_objective, g_constraints, h_constraints, dims
            )
        )

        # find edges that are likely stable
        self._stableish_edges = self._find_stableish_edges(
            2 * a_objective.T * a_objective,
            g_constraints,
            h_constraints,
            dims["q"],
            solution,
        )

    def _directed_distances(
        self, edges: Sequence[tuple[Hashable, Hashable]]
    ) -> tuple[spmatrix, matrix, spmatrix, matrix, list[int]]:
        # distance constraints when we also have an angle
        n_variables = len(self._nodes) * 2 + len(edges)
        directions_x = matrix(
            [-math.sin(self._edge_constraints[edge]["angle"]) for edge in edges]
        )
        directions_y = matrix(
            [math.cos(self._edge_constraints[edge]["angle"]) for edge in edges]
        )
        from_to_indices = 2 * matrix(
            [
                [self._node_to_index[from_node], self._node_to_index[to_node]]
                for from_node, to_node in edges
            ]
        )
        start_of_distance_variables = len(self._nodes) * 2
        lambda_indices = list(range(start_of_distance_variables, n_variables))

        if not edges:
            return (
                spmatrix([], [], [], size=(0, n_variables)),
                matrix(0.0, size=(0, 1)),
                spmatrix([], [], [], size=(0, n_variables)),
                matrix(0.0, size=(0, 1)),
                list(range(n_variables)),
            )

        # objective
        #  xB - xA - λ dx = 0
        #  yB - yA - λ dy = 0
        a_values = (
            [1] * len(edges)
            + [-1] * len(edges)
            + list(-directions_x)
            + [1] * len(edges)
            + [-1] * len(edges)
            + list(-directions_y)
        )
        a_rows = (
            list(range(len(edges))) * 3 + list(range(len(edges), len(edges) * 2)) * 3
        )
        a_columns = (
            list(from_to_indices[1, :])
            + list(from_to_indices[0, :])
            + lambda_indices
            + list(from_to_indices[1, :] + 1)
            + list(from_to_indices[0, :] + 1)
            + lambda_indices
        )
        a_least_squares = spmatrix(
            a_values, a_rows, a_columns, (max(a_rows) + 1, n_variables)
        )

        # constraints
        #  λ <= d_max
        #  -λ <= -d_min
        distances_min = [
            -self._edge_constraints[edge]["min_distance"] for edge in edges
        ]
        # more complicated for d_max as it is not always known
        distances_indices_max = [
            (max_distance, start_of_distance_variables + iedge)
            for iedge, edge in enumerate(edges)
            if (max_distance := self._edge_constraints[edge].get("max_distance", 0)) > 0
        ]
        distances_max = []
        indices_max = []
        if distances_indices_max:
            distances_max, indices_max = zip(*distances_indices_max, strict=True)
            distances_max = list(distances_max)
            indices_max = list(indices_max)
        g_values = [1] * len(distances_max) + [-1] * len(distances_min)
        g_rows = list(range(len(g_values)))
        g_columns = indices_max + list(range(start_of_distance_variables, n_variables))
        g_constraints = spmatrix(
            g_values, g_rows, g_columns, (max(g_rows) + 1, n_variables)
        )
        h_constraints = matrix(distances_max + distances_min)

        # for exact distance: replace variable with constant
        equal_rows_distances_columns = [
            (iconstraint, distance_max, idistance)
            for iconstraint, (distance_max, idistance) in enumerate(
                zip(distances_max, indices_max, strict=True)
            )
            if abs(
                distance_max + distances_min[idistance - start_of_distance_variables]
            )
            < 1e-8
        ]
        if equal_rows_distances_columns:
            # if any distance_min = distance_max
            rows, distances, columns = list(
                zip(*equal_rows_distances_columns, strict=True)
            )
            distances = spmatrix(
                distances,
                columns,
                [0] * len(distances),
                size=(a_least_squares.size[1], 1),
            )
            b_least_square = -a_least_squares * distances
            # which columns to keep in a/g, applied later (after we also have the cone constraints)
            keep_variables = [
                variable
                for variable in range(a_least_squares.size[1])
                if variable not in columns
            ]
            # remove rows in g/h, applied now
            keep_constraints = list(
                set(range(g_constraints.size[0]))
                .difference(rows)
                .difference(
                    {
                        variable - start_of_distance_variables + len(distances_max)
                        for variable in columns
                    }
                )
            )
            g_constraints = g_constraints[keep_constraints, :]
            h_constraints = h_constraints[keep_constraints]
        else:
            b_least_square = matrix(0.0, size=(a_least_squares.size[0], 1))
            keep_variables = list(range(a_least_squares.size[1]))

        return (
            a_least_squares,
            b_least_square,
            g_constraints,
            h_constraints,
            keep_variables,
        )

    def _undirected_distances(
        self, edges: Sequence[tuple[Hashable, Hashable]], n_variables: int
    ) -> tuple[spmatrix, matrix]:
        # use second-order cone constraints to limit the maximum distance between nodes
        g_cone_constraints = [
            spmatrix(
                [1, -1, 1, -1],
                # first row is empty: don't use linear part
                [1, 1, 2, 2],
                [
                    self._node_to_index[from_node] * 2,
                    self._node_to_index[to_node] * 2,
                    self._node_to_index[from_node] * 2 + 1,
                    self._node_to_index[to_node] * 2 + 1,
                ],
                (3, n_variables),
            )
            for from_node, to_node in edges
        ]
        h_cone_constraints = [
            matrix([self._edge_constraints[edge]["max_distance"], 0.0, 0.0])
            for edge in edges
        ]
        return g_cone_constraints, h_cone_constraints

    def plot(
        self,
        *,
        exclude_edges: Sequence[tuple[Hashable, Hashable]] = (),
        distance_unit: str = "m",
        subplots_kwargs: dict[str, Any] | None = None,
        node_text_kwargs: dict[str, Any] | None = None,
        edge_text_kwargs: dict[str, Any] | None = None,
    ) -> Figure:
        """Plot nodes and stable'ish edges.

        Args:
            exclude_edges:
                Sequence of edges that should not be plotted.
            distance_unit:
                Distance unit is printed for stable'ish edges.
            subplots_kwargs:
                Keywords forwarded to `matplotlib.pyplot.subplots`.
            node_text_kwargs:
                Keywords forwarded to `matplotlib.pyplot.text` for node names.
            edge_text_kwargs:
                Keywords forwarded to `matplotlib.pyplot.text` for edge description.
        """
        import matplotlib.pyplot as plt

        # default arguments
        subplots_kwargs = subplots_kwargs or {}
        node_text_kwargs = node_text_kwargs or {}
        edge_text_kwargs = edge_text_kwargs or {}

        figure, axis = plt.subplots(**({"figsize": (15, 15)} | subplots_kwargs))
        axis.set_aspect("equal")
        axis.set_axis_off()

        # plot nodes and their names
        axis.scatter(self._x[:, 0], self._x[:, 1], c="blue")
        for i, txt in enumerate(self._nodes):
            axis.text(
                self._x[i, 0],
                self._x[i, 1],
                txt,
                ha="left",
                va="bottom",
                **node_text_kwargs,
            )

        # determine which edges to plot
        # all edges between nodes that seem stable'ish
        edges = self._stableish_edges
        # all edges that have user-provided angle constraints
        edges = edges | {
            (from_node, to_node)
            for (from_node, to_node), constraints in self._edge_constraints.items()
            if "angle" in constraints and (to_node, from_node) not in edges
        }
        # remove user-requested edges
        edges = (
            edges
            - set(exclude_edges)
            - {(to_node, from_node) for from_node, to_node in exclude_edges}
        )

        # plot the measured directions
        for from_node, to_node in edges:
            from_index = self._node_to_index[from_node]
            to_index = self._node_to_index[to_node]

            # plot line
            axis.plot(
                self._x[[from_index, to_index], 0],
                self._x[[from_index, to_index], 1],
                "blue",
            )

            # and add angles
            nodes = self._x[[from_index, to_index], :]
            direction = nodes[1, :] - nodes[0, :]
            if direction[0] < 0:
                direction = -direction
            angle_left = math.atan2(direction[1], direction[0])
            angle_right = angle_left + math.pi

            axis.text(
                (nodes[1, 0] + nodes[0, 0]) / 2,
                (nodes[1, 1] + nodes[0, 1]) / 2,
                f"{_print_angle(angle_left, self._angle_unit)}→   ←{_print_angle(angle_right, self._angle_unit)}",
                ha="center",
                va="bottom",
                rotation=math.degrees(angle_left),
                rotation_mode="anchor",
                **edge_text_kwargs,
            )

            # add distance only if the edge is stable'ish
            if (from_node, to_node) not in self._stableish_edges and (
                to_node,
                from_node,
            ) not in self._stableish_edges:
                continue

            distance = blas.nrm2(direction)
            axis.text(
                (nodes[1, 0] + nodes[0, 0]) / 2,
                (nodes[1, 1] + nodes[0, 1]) / 2,
                f"{round(distance)}{distance_unit}",
                ha="center",
                va="top",
                rotation=math.degrees(angle_left),
                rotation_mode="anchor",
                **edge_text_kwargs,
            )

        # TODO: refine text positions to avoid overlapping elements
        return figure

    def _find_stableish_edges(
        self,
        p_objective: spmatrix,
        g_constraints: spmatrix,
        h_constraints: matrix,
        second_order_cone_dims: Sequence[int],
        solution: matrix,
        close_enough: float = 3.0,
    ) -> frozenset[tuple[Hashable, Hashable]]:
        """Find edges that are likely stable: they move very little."""
        # subset of nodes used for the analysis: nodes with node constraints are later
        # merged with the stable edges
        node_indices = [
            self._node_to_index[node]
            for node in self._nodes
            if node not in self._node_constraints
        ]

        stableish_nodes, stableish_edges = _sensitivity.find_stableish_nodes_and_edges(
            p_objective,
            g_constraints,
            h_constraints,
            spmatrix([], [], [], size=(0, p_objective.size[1])),
            solution,
            # compare all nodes with all nodes
            [
                [node_index * 2, node_index * 2 + 1]
                for node_index in range(len(node_indices))
            ],
            self._stable_distance,
            second_order_cone_dims,
        )
        if self._node_constraints:
            # need to remap the free nodes to the full list of nodes
            mapping = dict(enumerate(node_indices))
            stableish_nodes = {mapping[node] for node in stableish_nodes}
            stableish_edges = tuple(
                (mapping[node_from], mapping[node_to])
                for node_from, node_to in stableish_edges
            )

            # if there are _node_constraints add them and combine them with stablish nodes
            stableish_edges = stableish_edges + tuple(
                combinations(
                    stableish_nodes
                    | {self._node_to_index[node] for node in self._node_constraints},
                    r=2,
                )
            )
        return frozenset(
            (self._nodes[from_index], self._nodes[to_index])
            for from_index, to_index in stableish_edges
        )

    def _solve_with_fixed_nodes(
        self,
        a_objective: spmatrix,
        b_objective: matrix,
        g_constraints: spmatrix,
        h_constraints: matrix,
        dims: dict[str, int | list[int]],
    ) -> tuple[matrix, spmatrix, spmatrix, matrix]:
        """Solves the problem while replacing located nodes with constants.

        Setting node constraints through Ax=b is not as stable as replacing all occurrences with constants.
        """
        # at least one located node (this is later removed to not impact the edge analysis)
        fake_location = not self._node_constraints
        if fake_location:
            self._node_constraints = {self._nodes[0]: (1.0, 1.0)}

        located_node_indices = [
            index
            for node in self._node_constraints
            for index in (
                self._node_to_index[node] * 2,
                self._node_to_index[node] * 2 + 1,
            )
        ]
        locations = matrix(
            [
                location
                for locations in self._node_constraints.values()
                for location in locations
            ]
        )
        keep = [
            index
            for index in range(a_objective.size[1])
            if index not in located_node_indices
        ]
        b_objective_fixed = (
            b_objective - a_objective[:, located_node_indices] * locations
        )
        a_objective_fixed = a_objective[:, keep]
        h_constraints_fixed = (
            h_constraints - g_constraints[:, located_node_indices] * locations
        )
        g_constraints_fixed = g_constraints[:, keep]

        # prepopulate the solution with known locations
        x = matrix(0.0, (len(self._nodes) * 2, 1))
        x[located_node_indices] = locations
        keep = keep[: 2 * (len(self._nodes) - len(self._node_constraints))]

        # find a solution
        result = solvers.coneqp(
            # convert ||AX-b||^2 into the standard format for quadratic solvers
            P=2 * a_objective_fixed.T * a_objective_fixed,
            q=-2 * a_objective_fixed.T * b_objective_fixed,
            G=g_constraints_fixed,
            h=h_constraints_fixed,
            dims=dims,
        )
        if result["status"] != "optimal":
            error = ValueError("Failed to find an optimal solution")
            error.add_note(f"{result}")
            raise error

        # the optimization value: add the constant term so that the minimum is at 0
        self._y = result["primal objective"] + next(
            iter(b_objective_fixed.T * b_objective_fixed)
        )

        # re-map optimization results to node locations (exclude distances)
        x[keep] = result["x"][: len(keep)]
        full_solution = matrix([x, result["x"][len(keep) :]])
        x.size = (2, len(self._nodes))
        self._x = x.T

        if fake_location:
            self._node_constraints = {}
        else:
            a_objective = a_objective_fixed
            g_constraints = g_constraints_fixed
            h_constraints = h_constraints_fixed
            full_solution = result["x"]
        return full_solution, a_objective, g_constraints, h_constraints


def _print_angle(radian: float, unit: str) -> str:
    """Print the angle using the correct convention."""
    angle = radian - math.pi / 2
    if "clockwise" in unit:
        angle = -angle
    angle = (angle + 2 * math.pi) % (2 * math.pi)
    if "degree" in unit:
        angle = round(math.degrees(angle))
    return f"{angle}"

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pytest
from cvxopt import matrix, solvers, spmatrix
from scipy.linalg import null_space as null_space_scipy
from shapely import Polygon
from shapely.affinity import translate

from relmap import RelativeMap, _overlapping_text, _sensitivity


def assert_directions(
    actual_directions: np.ndarray,
    expected_angles: Sequence[float],
    abs_tol: float = 1e-6,
) -> None:
    actual_radians = (
        np.arctan2(actual_directions[:, 1], actual_directions[:, 0]) - np.pi / 2
    )
    expected_radians = np.deg2rad(expected_angles)

    assert np.rad2deg(
        np.arctan2(
            np.sin(actual_radians - expected_radians),
            np.cos(actual_radians - expected_radians),
        )
    ) == pytest.approx(0, abs=abs_tol)


def get_locations(solver: RelativeMap, nodes: Iterable[Hashable]) -> np.ndarray:
    assert solver.y >= -1e-8
    return np.array([list(solver.x[solver.node_to_index[node], :]) for node in nodes])


@pytest.mark.parametrize("angle", range(-90, 450, 5))
@pytest.mark.parametrize("add_distance", (False, True))
def test_two_nodes(angle: float, add_distance: bool) -> None:
    locations = get_locations(
        RelativeMap(
            {  # ty: ignore
                (0, 1): {"angle": angle}
                | ({"min_distance": 1.0, "max_distance": 2.0} if add_distance else {})
            }
        ),
        [0, 1],
    )
    direction = locations[[1]] - locations[[0]]
    assert_directions(direction, [angle])
    if add_distance:
        assert 1 <= np.linalg.norm(direction, 2) <= 2


@pytest.mark.parametrize("n_nodes", range(2, 32))
@pytest.mark.parametrize("round_nodes", (False, True))
@pytest.mark.parametrize("located_percentage", (0.0, 0.25))
@pytest.mark.parametrize("distance_percentage", (0.0, 0.5, 1.0))
@pytest.mark.parametrize(
    "percent_direction_constraints, undirected_distances",
    ((0.25, False), (0.5, False), (1.0, False), (0.25, True), (0.5, True)),
)
def test_random_input(
    n_nodes: int,
    percent_direction_constraints: float,
    round_nodes: bool,
    located_percentage: float,
    distance_percentage: float,
    undirected_distances: bool,
) -> None:
    # sample n_nodes point and give the solver a percentage of all possible constraints
    rng = np.random.default_rng(seed=n_nodes)
    nodes = rng.uniform(-10, 10, size=(n_nodes, 2))

    # round nodes to increase the chance of axis-aligned angles
    if round_nodes:
        nodes = nodes.round().astype(int)
    nodes = np.unique(nodes, axis=0)

    # calculate all angles
    directions = nodes[None] - nodes[:, None]
    angles = np.rad2deg(np.arctan2(directions[:, :, 1], directions[:, :, 0])) - 90

    # determine located nodes
    located_indices = np.nonzero(
        rng.uniform(0, 1, size=nodes.shape[0]) < located_percentage
    )[0]

    # determine which one to send to solver
    from_indices, to_indices = np.triu_indices(nodes.shape[0], k=1)
    mask = rng.uniform(0, 1, size=from_indices.size) <= percent_direction_constraints
    from_indices = from_indices[mask]
    to_indices = to_indices[mask]

    # remove directions between located nodes (the solver does that anyways
    # but we might end up with a located node that has no directions)
    keep = [
        not (from_index in located_indices and to_index in located_indices)
        for from_index, to_index in zip(from_indices, to_indices, strict=True)
    ]
    from_indices = from_indices[keep]
    to_indices = to_indices[keep]

    # not all nodes might be represented by the selected constraints: remove other nodes and remap indices
    represented_nodes = np.unique(np.concatenate([from_indices, to_indices]))
    nodes = nodes[represented_nodes]
    angles = angles[np.ix_(represented_nodes, represented_nodes)]
    remapping = dict(zip(represented_nodes, range(nodes.shape[0]), strict=True))
    from_indices = [remapping[index] for index in from_indices]
    to_indices = [remapping[index] for index in to_indices]
    located_indices = [
        remapping[index] for index in located_indices if index in remapping
    ]
    if not from_indices:
        return

    edge_constraints = {
        (from_index, to_index): {"angle": angles[from_index, to_index].item()}
        for from_index, to_index in zip(from_indices, to_indices, strict=True)
    }
    located_nodes = {
        node: (nodes[node, 0].item(), nodes[node, 1].item()) for node in located_indices
    }
    # add distance constraints to x% of directions
    for from_node, to_node in rng.choice(
        list(edge_constraints), int(len(edge_constraints) * distance_percentage)
    ).tolist():
        distance = np.linalg.norm(nodes[from_node] - nodes[to_node]).item()
        delta = 0.5
        # exact constraint in 25% of the cases
        if rng.uniform(0, 1) < 0.25:
            delta = 0.0
        edge_constraints[from_node, to_node] |= {
            "min_distance": distance - delta,
            "max_distance": distance + delta,
        }
    # add max_distance to all edges that have no angles
    if undirected_distances:
        for from_node in range(nodes.shape[0]):
            for to_node in range(from_node + 1, nodes.shape[0]):
                if (from_node, to_node) in edge_constraints:
                    continue
                max_distance = 1.1 * np.linalg.norm(nodes[from_node] - nodes[to_node])
                edge_constraints[from_node, to_node] = {"max_distance": max_distance}

    locations = get_locations(
        RelativeMap(edge_constraints, node_constraints=located_nodes),
        range(nodes.shape[0]),
    )

    assert locations[located_indices, :] == pytest.approx(nodes[located_indices])
    assert_directions(
        -locations[from_indices, :] + locations[to_indices, :],
        angles[from_indices, to_indices],
        abs_tol=0.5,
    )
    for (from_node, to_node), constraints in edge_constraints.items():
        if "max_constraint" not in constraints:
            continue
        distance = np.linalg.norm(locations[from_node, :] - locations[to_node, :])
        assert (
            constraints.get("min_distance", float(".inf"))
            <= distance
            <= constraints.get("max_distance", float("inf"))
        )


def test_crusty() -> None:
    # not a real test, an example
    min_factor = 0.9
    max_factor = 1.1
    RelativeMap(
        {  # ty: ignore
            ("South Dock", "Boxing Ring"): {
                "angle": 90,
                "min_distance": 26 * min_factor,
                "max_distance": 26 * max_factor,
            },
            ("South Dock", "Bus"): {
                "angle": 25,
                "min_distance": 39 * min_factor,
                "max_distance": 39 * max_factor,
            },
            ("South Dock", "Plane"): {"angle": 340},
            ("Bus", "Boxing Ring"): {"angle": 165},
            ("SAT Dish", "Boxing Ring"): {
                "angle": 220,
                "min_distance": 35 * min_factor,
                "max_distance": 35 * max_factor,
            },
            ("SAT Dish", "Bus"): {
                "angle": 270,
                "min_distance": 27 * min_factor,
                "max_distance": 27 * max_factor,
            },
            ("Bus", "Plane"): {
                "angle": 310,
                "min_distance": 30 * min_factor,
                "max_distance": 30 * max_factor,
            },
            ("Plane", "TBird"): {"angle": 325},
            ("TBird", "Trojan"): {"angle": 335},
            ("Plane", "Richardson"): {"angle": 30},
            ("Trojan", "Richardson"): {
                "angle": 90,
                "min_distance": 70 * min_factor,
                "max_distance": 70 * max_factor,
            },
            ("Trojan", "Chris Craft"): {
                "angle": 25,
                "min_distance": 88 * min_factor,
                "max_distance": 88 * max_factor,
            },
            ("Richardson", "Chris Craft"): {
                "angle": 340,
                "min_distance": 85 * min_factor,
                "max_distance": 85 * max_factor,
            },
            ("Plane", "Trojan"): {"max_distance": 79 * max_factor},
        },
        angle_unit="degree+clockwise",
        stable_distance=1.0,
    ).plot().savefig("crustys.pdf", bbox_inches="tight")


def test_null_space() -> None:
    rng = np.random.default_rng(seed=0)
    for size in ((10, 5), (10, 15)):
        a_matrix = rng.normal(size=size)
        expected = null_space_scipy(a_matrix)
        actual = _sensitivity.null_space(matrix(a_matrix))
        np.testing.assert_allclose(np.array(actual), expected)


def test_stableish_edges() -> None:
    # variables: x0, x1, x2, x3
    # objective: x0^2
    p_objective = spmatrix([1], [0], [0], size=(4, 4))
    q_objective = matrix(0.0, (4, 1))
    # constraint: x >= 0
    g_constraints = spmatrix([-1] * 4, list(range(4)), list(range(4)))
    h_constraints = matrix(0.0, size=(4, 1))
    # constraint: x1 + x2 + x3 == 1
    a_equal_constraints = matrix(1.0, size=(1, 4))
    b_equal_constraints = matrix([1.0])

    # find a solution
    solvers.options["show_progress"] = False
    solution = solvers.qp(
        p_objective,
        q_objective,
        g_constraints,
        h_constraints,
        a_equal_constraints,
        b_equal_constraints,
    )

    # test each variable by itself: there should be no stable edge
    stableish_nodes, stableish_edges = _sensitivity.find_stableish_nodes_and_edges(
        p_objective,
        g_constraints,
        h_constraints,
        a_equal_constraints,
        solution["x"],
        variable_groups=([0], [1], [2], [3]),
        max_distance=0.1,
    )
    assert stableish_nodes == {0}
    assert not stableish_edges

    # when fixing (x1, x2), (x2, x3) is fixed
    stableish_nodes, stableish_edges = _sensitivity.find_stableish_nodes_and_edges(
        p_objective,
        g_constraints,
        h_constraints,
        a_equal_constraints,
        solution["x"],
        variable_groups=([0], [1, 2], [2, 3], [3]),
        max_distance=0.1,
    )
    assert stableish_nodes == {0}
    assert stableish_edges == ((1, 2),)

    # all individual variables are unique for a large distance
    stableish_nodes, stableish_edges = _sensitivity.find_stableish_nodes_and_edges(
        p_objective,
        g_constraints,
        h_constraints,
        a_equal_constraints,
        solution["x"],
        variable_groups=([0], [1], [2], [3]),
        max_distance=1,
    )
    assert stableish_nodes == {0, 1, 2, 3}
    assert stableish_edges == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def test_cone_stableish_edges_stable() -> None:
    # 1 should be at (0, 1): all nodes are uniquely determined
    solver = RelativeMap(
        {(0, 1): {"max_distance": 1}, (1, 2): {"max_distance": 1}},  # ty: ignore
        node_constraints={0: (0, 0), 2: (0, 2)},
    )
    assert len(solver.stableish_edges) == 3


def test_cone_stableish_edges_unstable() -> None:
    # 1 could be anywhere: no stable edges
    solver = RelativeMap({(0, 1): {"max_distance": 1}}, node_constraints={0: (0, 0)})  # ty: ignore
    assert not solver.stableish_edges


def test_minimal_translation_vectors() -> None:
    # generate boxes with no overlap, overlap, rotation, and fully enclosed
    boxes = [
        np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]),
        np.array([[1, 0], [0, -1], [-1, 0], [0, 1]]),
    ]
    boxes += [box * 2 for box in boxes] + [box + 1 for box in boxes]
    boxes = list(map(Polygon, boxes))
    for box_a in boxes:
        for box_b in boxes:
            vectors = _overlapping_text.minimal_translation_vector(box_a, box_b, False)
            for vector in vectors:
                if np.linalg.norm(vector) < 1e-6:
                    continue
                assert translate(box_a, *vector).intersection(box_b).area < 1e-6
                assert translate(box_a, *vector * 0.99).intersection(box_b).area > 1e-6

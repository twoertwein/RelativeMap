import math
from collections.abc import Mapping, Sequence
from itertools import accumulate, pairwise

from cvxopt import lapack, matrix, spmatrix


def find_stableish_edges(
    p_objective: spmatrix,
    g_constraints: spmatrix,
    h_constraints: matrix,
    a_equal_constraints: spmatrix,
    solution: matrix,
    variable_groups: Sequence[Sequence[int]],
    dims: Mapping[str, int | Sequence[int]],
    max_distance: float,
    tolerance: float = 1e-7,
) -> tuple[tuple[int, int]]:
    """Find pair of nodes (edges) whose relationship is likely to be stable in a quadratic problem.

    Note:
        A "node" is a group of variables.
        A node pair is "stable", when one node can't move (much) given the other
        nodes is fixed (and the reverse).
        This function should not be used for concave problems as it analyzes the
        null space around the current solution.
        Even for positive-semi-definite P, this function might incorrectly indicate
        that an edge is stable as it only tests the basis directions for movement of
        each nodes and not all infinitely directions spanned by the null space.

    Args:
        p_objective:
            A positive-semi-definitive matrix.
        g_constraints:
            Linear inequality constraints matrix.
        h_constraints:
            The upper bound of the inequality constraints.
        a_equal_constraints:
            Equality constraints matrix.
        solution:
            The solution found by the solver.
        variable_groups:
            Sequence of variable sequences. A variable sequence defines a "node".
        dims:
            Used to determine the cone constraints. Supports only cone constraints with c=0.
        max_distance:
            The maximum movement to be allowed for the edge to be stable'ish.
        tolerance:
            Used to determine if a value is close enough to zero.

    Returns:
        A list of two-element tuples that represent stable edges. Tuple values are indices into `variable_groups`.
    """
    assert not dims.get("s", []), "Not supported"

    # split linear and cone constraints
    (
        g_constraints,
        h_constraints,
        g_cone_constraints,
        max_norm_cone_constraints,
        g_x_h_cone_constraints,
    ) = _split_linear_and_cone_constraints(g_constraints, h_constraints, solution, dims)

    # assume everything can move, unless we find a direction with enough movement
    stationary = [
        {j for j in range(len(variable_groups)) if i != j}
        for i in range(len(variable_groups))
    ]

    # active constraints and the objective function define the directions
    slacks = h_constraints - g_constraints * solution
    active_constraints = [
        index for index, slack in enumerate(slacks) if slack < tolerance
    ]
    free_null_space = null_space(
        matrix([p_objective, a_equal_constraints, g_constraints[active_constraints, :]])
    )

    # inactive constraints define the distances along the constraints
    inactive_indices = [
        index for index, slack in enumerate(slacks) if slack >= tolerance
    ]
    constraints_inactive = g_constraints[inactive_indices, :]
    distance_to_inactive_constraint = slacks[inactive_indices]

    for ifixed, fixed_variables in enumerate(variable_groups):
        if not stationary[ifixed]:
            continue

        # fix the current variables: this is the same as the null space of the original
        # matrix with two rows appended to fix the variables but that would be far less efficient
        fixed_null_space = free_null_space * null_space(
            free_null_space[fixed_variables, :]
        )
        if fixed_null_space.size[1] == 0:
            # no movement possible
            continue

        # We can't check all direction that are induced by the null space. Instead of using only the
        # basis vectors, we also add a few interpolations of the basis vectors to reduce the chance
        # of incorrectly inferring that an edge is stable
        fixed_null_space = matrix(
            [[fixed_null_space]]
            + [
                [(fixed_null_space[:, ibasis] + fixed_null_space[:, jbasis]) / 2]
                for ibasis in range(fixed_null_space.size[1])
                for jbasis in range(ibasis + 1, fixed_null_space.size[1])
            ]
        )

        # > 0: moving towards an inactive constraint
        # = 0: moving parallel to an inactive constraint
        # < 0: moving away from an inactive constraint
        speeds_towards_constraints = constraints_inactive * fixed_null_space
        speeds_towards_cone_constraints = [
            g_cone_constraint * fixed_null_space
            for g_cone_constraint in g_cone_constraints
        ]

        # check for each variable group, if it can move
        for ivariables in stationary[ifixed].copy():
            variables = variable_groups[ivariables]
            direction_norms = (
                matrix(1.0, size=(1, len(variables)))
                * (fixed_null_space[variables, :] ** 2)
            ) ** 0.5

            # check how much we can move along each direction
            for idirection, direction_norm in enumerate(direction_norms):
                # Skip the basis vector if there is no movement for the current variables
                if direction_norm < tolerance:
                    continue

                speed_towards_constraints = speeds_towards_constraints[:, idirection]
                # maximum distance to reach the first inactive constraint
                max_alpha = min(
                    (
                        distance_to_inactive_constraint[index] / speed
                        for index, speed in enumerate(speed_towards_constraints)
                        # can ignore constraints from which we are moving away or are moving
                        # parallel to: can move infinetally
                        if speed > tolerance
                    ),
                    default=float("inf"),
                )

                # check cone constraints - only when distance is larger
                if max_alpha * direction_norm >= max_distance:
                    max_alpha = min(
                        [
                            find_max_step_cone_constraint(
                                g_x_h_cone_constraint,
                                speed_towards_cone_constraints[:, idirection],
                                max_norm_cone_constraint,
                                tolerance,
                            )
                            for g_x_h_cone_constraint, max_norm_cone_constraint, speed_towards_cone_constraints in zip(
                                g_x_h_cone_constraints,
                                max_norm_cone_constraints,
                                speeds_towards_cone_constraints,
                                strict=True,
                            )
                        ],
                        default=max_alpha,
                    )

                    if max_alpha * direction_norm >= max_distance:
                        stationary[ifixed].remove(ivariables)
                        stationary[ivariables].remove(ifixed)
                        break

    return tuple(
        (from_group, to_group)
        for from_group, to_groups in enumerate(stationary)
        for to_group in to_groups
        if from_group < to_group
    )


def null_space(a_matrix: matrix) -> matrix:
    """Computes the null space similarly to scipy.

    Args:
        a_matrix:
            The input matrix A. This matrix will be mutated!

    Returns:
        A matrix whose columns form an orthonormal basis of the null space of A.
    """
    rows, columns = a_matrix.size

    # We need the singular values (S) and the right singular vectors (Vt, jobvt="A").
    # We don't need the left singular vectors (U, jobu="N")
    singular_values = matrix(0.0, (min(rows, columns), 1))
    singular_vectors = matrix(0.0, (columns, columns))
    lapack.gesvd(a_matrix, singular_values, jobu="N", jobvt="A", Vt=singular_vectors)

    # SciPy's default tolerance calculation
    max_singular_value = max(singular_values, default=0)
    np_finfo_float_eps = 2.220446049250313e-16
    tolerance = max_singular_value * max(rows, columns) * np_finfo_float_eps

    # The rank is the number of singular values greater than the tolerance
    rank = sum(value > tolerance for value in singular_values)

    # The null space is spanned by the last (n - rank) right singular vectors.
    return singular_vectors[rank:, :].T


def find_max_step_cone_constraint(
    g_x_h: matrix, g_direction: matrix, max_norm: float, tolerance: float
) -> float:
    """Calculates the maximum step size for a specific second-order cone constraint.

    The constraint is of the form: ||G*(x + step*direction) + h||_2 <= max_norm.

    Args:
        g_x_h:
            The matrix G in the constraint.
        g_direction:
            Speed of going towards the constraint.
        max_norm:
            The maximal norm.
        tolerance:
            Used to determine if a value is close enough to zero.

    Returns:
        The maximum non-negative step size.
    """
    # Calculate the coefficientsfor the quadratic formula
    a = (g_direction.T * g_direction)[0]
    b = (2 * g_x_h.T * g_direction)[0]
    c = (g_x_h.T * g_x_h)[0] - max_norm**2

    # Check if the initial point is feasible (with a small tolerance)
    if c > tolerance:
        return 0.0

    # Case when a is zero: ||G*dir|| is zero, so the step has no impact on the norm.
    if a < tolerance:
        return float("inf")

    # The largest root of the quadratic equation is the maximum step
    return (-b + math.sqrt(max(b**2 - 4 * a * c, 0.0))) / (2 * a)


def _split_linear_and_cone_constraints(
    g_constraints: spmatrix,
    h_constraints: matrix,
    solution: matrix,
    dims: Mapping[str, int | Sequence[int]],
) -> tuple[spmatrix, matrix, list[spmatrix], list[float], list[matrix]]:
    # split linear and cone constraints
    n_linear_constraints = dims.get("l", g_constraints.size[0])
    cone_dimensions = dims.get("q", [])
    constraint_ends = list(accumulate([n_linear_constraints, *cone_dimensions]))
    g_cone_constraints = [
        g_constraints[start:end, :] for start, end in pairwise(constraint_ends)
    ]

    # remove linear part of cone constraints
    assert all(
        cone_constraint[0, :].V.size[0] == 0 for cone_constraint in g_cone_constraints
    ), "Do not support linear terms in cone constraints"
    g_cone_constraints = [
        g_cone_constraint[1:, :] for g_cone_constraint in g_cone_constraints
    ]

    # split norm maximum from h
    h_cone_constraints = [
        h_constraints[start:end] for start, end in pairwise(constraint_ends)
    ]
    max_norm_cone_constraints = [
        h_cone_constraint[0] for h_cone_constraint in h_cone_constraints
    ]

    # pre-compute expression for _find_max_step_norm
    g_x_h_cone_constraints = [
        g_cone_constraint * solution + h_cone_constraint[1:]
        for g_cone_constraint, h_cone_constraint in zip(
            g_cone_constraints, h_cone_constraints, strict=True
        )
    ]

    # linear constraints
    g_constraints = g_constraints[:n_linear_constraints, :]
    h_constraints = h_constraints[:n_linear_constraints]

    return (
        g_constraints,
        h_constraints,
        g_cone_constraints,
        max_norm_cone_constraints,
        g_x_h_cone_constraints,
    )

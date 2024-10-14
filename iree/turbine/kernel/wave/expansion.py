# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import torch.fx as fx
from typing import Any, TypeAlias
from functools import partial

from .constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from ..ops.wave_ops import *
from .._support.indexing import IndexingContext, IndexSequence
from ...support.logging import get_logger
from .._support.tracing import CapturedTrace
from .utils import get_mma_dimensional_mapping, specialize_index_sequence
from ..lang.global_symbols import *

logger = get_logger("turbine.wave.expansion")
# This represents a mapping of a node + indexing + res_idx(output index for op with multiple results)
# of node into the dimensions to the corresponding expanded node in these specific dimensions.
# An example for a record in this map is (read_0_0_0, ((M,0),(N,0),(K,1), 0) -> read_0_0_1.
ExpandedNodeMap: TypeAlias = dict[
    tuple[CustomOp, tuple[tuple[IndexSymbol, int], int, ...]], CustomOp
]


def already_expanded_op(node: CustomOp, dims: dict[IndexSymbol, int]) -> bool:
    return hasattr(node.fx_node, "expanded_dims") and (
        filter_and_zero_unselected_dims(dims, node.indexing_dims)
        == node.fx_node.expanded_dims  # type: ignore
    )


def expansion_needed(
    dims: dict[IndexSymbol, int], selection: Sequence[IndexSymbol]
) -> bool:
    """Check if any of the dimensions in the selection are non-zero."""
    return any(dim[1] != 0 and dim[0] in selection for dim in dims.items())


def filter_and_zero_unselected_dims(
    dims: dict[IndexSymbol, int], selection: Sequence[IndexSymbol]
) -> dict[IndexSymbol, int]:
    """
    Filters dimensions based on selection and sets unselected dimensions' values to zero.
    """
    return {dim: val if dim in selection else 0 for dim, val in dims.items()}


def get_dim_combinations(
    all_dims: dict[IndexSymbol, int],
    selection: Sequence[IndexSymbol],
):
    """
    Returns all combinations of sizes for the selected dimensions.
    Other dimensions are clamped to 0.
    """
    adjusted_dimension_sizes = [
        list(range(all_dims[dim])) if dim in selection else [0] for dim in all_dims
    ]
    return itertools.product(*adjusted_dimension_sizes)


def get_indexed_dims(
    all_dims: dict[IndexSymbol, int], nodeOrDims: CustomOp | Sequence[IndexSymbol]
) -> tuple[tuple[IndexSymbol, int], ...]:
    """
    Generates a tuple of (key, value) pairs from the provided dimensions.
    If given a CustomOp instance, it uses its indexing_dims attribute.
    """
    if isinstance(nodeOrDims, CustomOp):
        nodeOrDims = nodeOrDims.indexing_dims
    # Flatten dims for node with multiple values or expanded Reduction.
    if all(isinstance(el, Sequence) for el in nodeOrDims):
        flattened_dims = list(itertools.chain.from_iterable(nodeOrDims))
        flatten_dims_set = dict.fromkeys(flattened_dims)
        nodeOrDims = list(flatten_dims_set)
    return tuple((key, all_dims[key]) for key in nodeOrDims if key in all_dims)


def get_last(node_list: fx.graph._node_list) -> fx.Node:  # type: ignore
    """Get the last element of the fx node_list structure"""
    return next(iter(reversed(node_list)))  # type: ignore


def is_expandable(arg: Any) -> bool:
    """Check if an argument is expandable."""
    if isinstance(arg, list):
        return all(is_expandable(a) for a in arg)
    # Placeholder nodes are only expanded if they are a reduction init arg
    if isinstance(arg, Placeholder) and not isinstance(arg, IterArg):
        return False
    return isinstance(arg, CustomOp)


def is_contiguous_dim(
    dim: IndexSymbol, symbolic_shape: list[IndexSymbol], vector_shapes: list[int]
) -> bool:
    """
    Checks if the given dimension is stored contiguously in memory. This happens if
    the dimension is the last one in the symbolic shape or all dimensions after it
    are unit dimensions.
    """
    is_innermost_dim = dim == symbolic_shape[-1]
    dim_index = symbolic_shape.index(dim)
    static_shape = [vector_shapes[dim] for dim in symbolic_shape]
    all_unit_dims = all(dim == 1 for dim in static_shape[dim_index + 1 :])
    return is_innermost_dim or all_unit_dims


def compute_stride(
    symbolic_shape: tuple[IndexSymbol, ...],
    vector_shapes: dict[IndexSymbol, int],
    target_dim: IndexSymbol,
) -> int:
    """
    Compute the stride for a given dimension based on the vector shapes.
    The stride is the product of the vector shapes of all dimensions that are
    not the given dimension.
    """
    stride = 1
    for dim in reversed(symbolic_shape):
        if dim == target_dim:
            break
        assert dim in vector_shapes, f"Dimension {dim} not found in vector shapes"
        stride *= vector_shapes[dim]

    try:
        stride = int(stride)
    except Exception as e:
        logger.error(e)
    return stride


def set_node_index(
    constraints: Sequence[Constraint],
    mma_index: dict[IndexSymbol, int],
    mma_slices: dict[IndexSymbol, list[fx.Node]],
    dim_tile_size: dict[IndexSymbol, int],
    custom: CustomOp,
    dim_scaling: dict[IndexSymbol, int],
):
    """
    Set the index of the node based on the user constraints. In certain
    operators (like read, write), there is only a single index associated
    with the node (the index to read from, the index to write to). But for
    other operators like mma, each operand reads from a different index.

    Rather than maintain operand specific indices for operators, we maintain
    dimension specific indices for each operator. So for an mma operator that
    has a signature of (MxK, NxK) -> MxN, we maintain only 3 mappings for
    dimensions M, N and K, but allow each mapping to be piecewise conditioned
    on the operand.
    """
    hardware_constraint = [c for c in constraints if isinstance(c, HardwareConstraint)]
    workgroup_constraints = {
        c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    }
    other_constraints = [
        c for c in constraints if not isinstance(c, HardwareConstraint)
    ]
    # Apply hardware constraint first since it dictates the stride and size.
    sorted_constraints = hardware_constraint + other_constraints

    index = {}
    temp_test = "init"
    for dim in custom.indexing_dims:
        dim_test = "init"
        index_seq = None
        for constraint in sorted_constraints:
            cont_test = "init"
            mma_check = isinstance(constraint, HardwareConstraint) and dim in mma_index

            vector_check = (
                isinstance(constraint, HardwareConstraint)
                and constraint.vector_shapes is not None
                and hasattr(custom, "elements_per_thread")
            )

            constraint_check = (
                not isinstance(constraint, HardwareConstraint) and dim == constraint.dim
            )

            if (not (mma_check or vector_check)) and (not constraint_check):
                continue

            if isinstance(constraint, HardwareConstraint):

                # The semantics of elements_per_thread are that it represents the number of
                # elements that are loaded contiguously from memory.
                elements_per_thread = getattr(custom, "elements_per_thread", None)
                constraint_index, elements_per_thread, stride = (
                    (
                        workgroup_constraints[dim].workgroup_dim,
                        (
                            1
                            if not is_contiguous_dim(
                                dim,
                                custom.indexing_dims,
                                constraint.vector_shapes,
                            )
                            else elements_per_thread
                        ),
                        compute_stride(
                            custom.indexing_dims, constraint.vector_shapes, dim
                        ),
                    )
                    if constraint.vector_shapes is not None
                    else (mma_index[dim], elements_per_thread, None)
                )
                index_seq = constraint.apply(
                    constraint_index, dim, elements_per_thread, stride
                )
                if mma_index:
                    # Newly expanded op may not be found in mma_slices.
                    pre_expanded_src = getattr(custom.fx_node, "expanded_src", custom)
                    index_seq = specialize_index_sequence(
                        index_seq, mma_slices, pre_expanded_src
                    )

            else:
                if index_seq is None:
                    index_seq = constraint.apply()
                else:
                    index_seq.start += constraint.apply().start

        if index_seq is not None:
            if dim in dim_scaling and dim in dim_tile_size:
                index_seq.start += dim_scaling[dim] * dim_tile_size[dim]
            index.update({dim: index_seq})
        else:
            index.update({dim: IndexSequence(0, 1, 1)})

    setattr(custom.fx_node, "index", index)


def expand_graph(
    trace: CapturedTrace,
    constraints_or_scaling: Sequence[Constraint] | dict[IndexSymbol, int],
):
    """
    Create a graph that represents the expanded version of the wave function.
    The expansion is done in the dimensions specified by the constraints.
    """
    if isinstance(constraints_or_scaling, dict):
        dim_scaling = constraints_or_scaling
        node_index_setter = lambda *args: None
    else:
        mma_index, mma_slices = get_mma_dimensional_mapping(trace)
        dim_scaling, dim_tile_size = get_dim_scaling(constraints_or_scaling, mma_index)
        node_index_setter = partial(
            set_node_index, constraints_or_scaling, mma_index, mma_slices, dim_tile_size
        )

    # Start from the back and expand in the corresponding indexing dimensions of a node
    # Then proceed to the operands
    leaf_nodes: list[Type[CustomOp]] = [Write]

    # Some graphs may not have a write node, so we need to add the leaf nodes present in the
    # graph, excluding output nodes.
    all_fx_nodes_reversed = list(reversed(trace.get_root_graph().nodes))
    has_write = any(
        isinstance(get_custom(fx_node), Write) for fx_node in all_fx_nodes_reversed
    )
    if not has_write:
        for node in (get_custom(fx_node) for fx_node in all_fx_nodes_reversed):
            if isinstance(node, Output):
                continue
            leaf_nodes.append(node.__class__)
            break

    expansion_context: ExpandedNodeMap = {}
    for node in (get_custom(fx_node) for fx_node in all_fx_nodes_reversed):

        # Expansion begins at the leaf nodes
        if node.__class__ not in leaf_nodes:
            continue

        for dim_combination in get_dim_combinations(dim_scaling, node.indexing_dims):
            expand_dims = {
                dim: val for dim, val in zip(dim_scaling.keys(), dim_combination)
            }
            logger.debug(f"Starting expansion at leaf:{node} in dims:{expand_dims}")
            _expand_node(
                node,
                trace,
                expand_dims,
                dim_scaling,
                node_index_setter,
                expansion_context,
            )


def _expand_node(
    node: CustomOp | list[CustomOp],
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    node_index_setter: Callable[[CustomOp, dict[IndexSymbol, int]], None],
    context: ExpandedNodeMap,
    res_idx: int = 0,
) -> CustomOp:
    """Expand a single node or list of nodes in specific dimensions and recursively proceed to its inputs."""
    if isinstance(node, list):
        expanded_nodes = []
        for elem in node:
            expanded_nodes.append(
                _expand_node(
                    elem,
                    trace,
                    dim_query,
                    dim_scaling,
                    node_index_setter,
                    context,
                    res_idx,
                ).fx_node
            )
        return expanded_nodes
    # If we expanded a node in the same dimensions before, we can reuse it
    if (node, get_indexed_dims(dim_query, node), res_idx) in context:
        logger.debug(f"Already expanded node: {node} in {dim_query}")
        return context[(node, get_indexed_dims(dim_query, node), res_idx)]
    elif isinstance(node, Reduction):
        return _expand_reduction(
            node, trace, dim_query, dim_scaling, node_index_setter, context, res_idx
        )
    elif isinstance(node, Getitem):
        res_idx = node.res_idx
    elif isinstance(node, GetResult) and not isinstance(node, Getitem):
        # The presence of a GetResult node indicates that the reduction has already
        # been expanded. Simply return the corresponding node.
        reduction = get_custom(node.value)
        return context[(reduction, get_indexed_dims(dim_query, reduction), res_idx)]
    elif isinstance(node, Allocate):
        # Allocate nodes are not expanded.
        return node

    # Handles case where we are trying to expand a "new_node" that
    # came from an expansion of the original node. To prevent re-expansion
    # of the "new_node" in the same dims.
    if already_expanded_op(node, dim_query):
        return node

    # Filter out the dimensions that are not indexed by the node
    restricted_dims = filter_and_zero_unselected_dims(dim_query, node.indexing_dims)
    logger.debug(f"Expanding node: {node} in {restricted_dims}")

    # For iter args, we want to insert
    if not hasattr(_expand_node, "last_expanded_iter_arg"):
        _expand_node.last_expanded_iter_arg = None

    # Clone the node for the new expansion. The original node is reused for the
    # case of all dimensions being zero.
    if expansion_needed(restricted_dims, node.indexing_dims):
        new_node = node.copy(
            anchor=(
                _expand_node.last_expanded_iter_arg
                if isinstance(node, IterArg)
                else None
            )
        )
    else:
        new_node = node
        logger.debug(f"did not clone node: {node} in {restricted_dims}")

    if isinstance(node, IterArg):
        _expand_node.last_expanded_iter_arg = new_node.fx_node

    new_node.fx_node.expanded_dims = restricted_dims
    new_node.fx_node.expanded_src = node
    new_node.fx_node.name = get_expanded_name(node, restricted_dims)
    node_index_setter(new_node, restricted_dims)

    constraints = node_index_setter.args[0]

    # Proceed with expansion of the arguments
    for i, arg in node.node_args.items():
        if is_expandable(arg):
            new_arg = _expand_node(
                arg,
                trace,
                restricted_dims,
                dim_scaling,
                node_index_setter,
                context,
                res_idx,
            )
            new_node.update_arg(i, new_arg)

    new_node.post_expansion(constraints)

    context[(node, get_indexed_dims(restricted_dims, node), res_idx)] = new_node
    # Uncomment to fix:
    # context[(new_node, get_indexed_dims(restricted_dims, node), res_idx)] = new_node
    return new_node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    node_index_setter: Callable[[CustomOp, dict[IndexSymbol, int]], None],
    context: ExpandedNodeMap,
    res_idx: int = 0,
) -> CustomOp:
    """Expand a reduction in a specific dimension and recursively proceed to its inputs."""
    # Determine the dimensions to expand the reduction from the indexing of its users
    users = reduction.users
    expand_dims: list[IndexSymbol] = []
    for user in users:
        for indexing_dim in user.indexing_dims:
            if indexing_dim not in expand_dims:
                expand_dims.append(indexing_dim)
    logger.debug(f"expanding reduction in dims: {expand_dims}")

    # Get the output node of the reduction
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    local_mma_index, local_mma_slices = get_mma_dimensional_mapping(reduction_subgraph)
    local_index_setter = partial(
        node_index_setter.func,
        node_index_setter.args[0],
        local_mma_index,
        local_mma_slices,
        node_index_setter.args[3],
    )
    output = get_custom(get_last(reduction_subgraph.nodes))
    if not isinstance(output, Output):
        raise ValueError(
            "fx.Graph malformed: The last node of a subgraph must be an output node"
        )

    new_output_args = []
    new_init_args = []
    for dim_vals in get_dim_combinations(dim_scaling, expand_dims):
        return_vals = output.return_vals[0]
        dims = {dim: val for dim, val in zip(dim_scaling.keys(), dim_vals)}
        if not isinstance(return_vals, Sequence):
            return_vals = [return_vals]
        for arg_idx, arg in enumerate(return_vals):
            arg = get_custom(arg)
            # Add GetResult nodes for the corresponding dimensions
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, len(new_output_args))
            new_node.add_to_graph(reduction.graph)
            new_node.fx_node.name = get_expanded_name(new_node, dims)
            context[
                (reduction, get_indexed_dims(dims, expand_dims), arg_idx)
            ] = new_node

            # Proceed with expansion inside the reduction
            expanded_output = _expand_node(
                arg, trace, dims, dim_scaling, local_index_setter, context, 0
            )
            if expanded_output in new_output_args:
                continue
            new_output_args.append(expanded_output)

        # Proceed with expansion outside the reduction
        for init_arg in reduction.init_args:
            expanded_init_arg = _expand_node(
                get_custom(init_arg),
                trace,
                dims,
                dim_scaling,
                local_index_setter,
                context,
                0,
            )
            if expanded_init_arg in new_init_args:
                continue
            new_init_args.append(expanded_init_arg)

    # Update init_args and return values
    reduction.update_arg(
        "init_args", [new_init_arg.fx_node for new_init_arg in new_init_args]
    )
    output.update_arg("return_vals", [node.fx_node for node in new_output_args])
    _handle_reduction_dim(
        reduction,
        output,
        trace,
        dim_scaling,
        local_index_setter,
        context,
        0,
    )
    # Even though we expanded the reduction in multiple dimensions, we only return
    # the node corresponding to the original query
    return context[(reduction, get_indexed_dims(dim_query, expand_dims), res_idx)]


def get_expanded_name(node: CustomOp, dims: dict[IndexSymbol, int]) -> str:
    """Returns the name of a node with the dimensions appended."""

    separated = node.fx_node.name.split("_")
    node_name = separated[0]
    if isinstance(node, Read) or isinstance(node, Write):
        if get_custom(node.memory).type.address_space == SHARED_ADDRESS_SPACE:
            node_name = node_name + "_shared"
    # Special case for get_result op
    if node_name == "get":
        node_name = node_name + separated[1]
    for val in dims.values():
        node_name += f"_{val}"
    return node_name


def _contains(elem, container):
    if container is None:
        return False

    return elem in container


def get_dim_scaling(
    constraints: Sequence[Constraint], mma_indices: dict[IndexSymbol, int]
) -> tuple[dict[IndexSymbol, int]]:
    """Get the number of expansions for the dimensions based on the constraints."""
    dim_scaling: dict[IndexSymbol, int] = {}
    dim_tile_size: dict[IndexSymbol, int] = {}
    hardware_constraints: list[HardwareConstraint] = [
        constraint
        for constraint in constraints
        if isinstance(constraint, HardwareConstraint)
    ]
    if len(hardware_constraints) != 1:
        raise ValueError("Exactly one hardware constraint must be provided")

    idxc = IndexingContext.current()
    for constraint in constraints:
        if isinstance(constraint, WorkgroupConstraint) or isinstance(
            constraint, TilingConstraint
        ):
            hw_cons = hardware_constraints[0]
            tile_size = idxc.get_static_value(constraint.tile_size)
            if not (
                _contains(constraint.dim, mma_indices)
                or _contains(constraint.dim, hw_cons.vector_shapes)
            ):
                raise ValueError(
                    f"Attempting to determine vector shape for unmapped dimension {constraint.dim}"
                )

            if mma_indices:
                vector_size = hardware_constraints[0].mma_matrix_shapes[
                    mma_indices[constraint.dim]
                ]
            else:
                vector_size = hardware_constraints[0].vector_shapes[constraint.dim]

            wave_count = 1
            if isinstance(constraint, WorkgroupConstraint):
                wave_count = hardware_constraints[0].waves_per_block[
                    constraint.workgroup_dim
                ]
            if tile_size is None or wave_count is None or vector_size is None:
                raise ValueError(
                    "Tile size, wave count and vector size must be statically known"
                )
            if (
                tile_size % wave_count != 0
                or (tile_size / wave_count) % vector_size != 0
            ):
                raise ValueError(
                    "Tile size must be divisible by wave count and vector size"
                )
            dim_scaling[constraint.dim] = tile_size // wave_count // vector_size
            dim_tile_size[constraint.dim] = vector_size
    return (dim_scaling, dim_tile_size)


def _handle_reduction_dim(
    reduction: Reduction,
    output: Output,
    trace: CapturedTrace,
    dim_scaling: dict[IndexSymbol, int],
    node_index_setter: Callable[[CustomOp, dict[IndexSymbol, int]], None],
    context: ExpandedNodeMap,
    res_idx: int,
):
    # Iterate through ops that requires expansion in reduction dims
    # and expand starting there and propagate up.
    reduction_root_ops: list[CustomOp] = []
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    for node in (get_custom(fx_node) for fx_node in reduction_subgraph.nodes):
        if isinstance(node, MMA) and reduction.axis in node.indexing_dims:
            reduction_root_ops.append(node)
        elif isinstance(node, ReduceOp) and reduction.axis == node.dim:
            reduction_root_ops.append(node)

    # TODO: Add support for case where we process MMA before returning to IterArg.
    def get_output_index(custom: CustomOp):
        output_users = [
            get_custom(user)
            for user in custom.fx_node.users
            if isinstance(get_custom(user), Output)
        ]
        if len(output_users) != 1:
            raise NotImplementedError(
                "NYI: Currently only handle direct and 1:1 MMA -> Output case."
            )
        return output_users[0].return_vals[0].index(custom.fx_node)

    new_outputs = list(reduction.outputs(trace.get_subgraph(reduction.subgraph_name)))
    new_reductions = []
    expanded_reductions = set()
    for reduction_root_op in reduction_root_ops:
        op_output_index = get_output_index(reduction_root_op)
        if isinstance(reduction_root_op, MMA):
            op_acc_name = "acc"
        elif isinstance(reduction_root_op, ReduceOp):
            op_acc_name = "init"
        else:
            raise NotImplementedError(
                f"NYI: Expanding reduction dim for {type(reduction_root_op)} is not supported yet."
            )
        if dim_scaling[reduction.axis] <= 1:
            continue
        dims = reduction_root_op.fx_node.expanded_dims.copy()
        last_expanded_reduction_op = reduction_root_op
        for scale_idx in range(1, dim_scaling[reduction.axis]):
            dims[reduction.axis] = scale_idx
            new_node = reduction_root_op.copy(anchor=last_expanded_reduction_op.fx_node)

            for arg_idx, node_arg in enumerate(new_node.fx_node.args):
                if not isinstance(node_arg, fx.Node):
                    continue
                if node_arg == getattr(new_node, op_acc_name):
                    continue
                new_node_arg = _expand_node(
                    get_custom(node_arg),
                    trace,
                    dims,
                    dim_scaling,
                    node_index_setter,
                    context,
                    res_idx,
                )
                # This expansion always happens, user should never be reused
                assert new_node_arg != node_arg
                new_node.update_arg(arg_idx, new_node_arg)

            # Update MMA_{t} to accumulate on MMA_{t-1}, and then save
            # current MMA_{t} to outputs for use in next loop.
            new_node.update_arg(op_acc_name, last_expanded_reduction_op)
            new_node.fx_node.name = get_expanded_name(reduction_root_op, dims)
            last_expanded_reduction_op = new_node
            expanded_reductions.add(last_expanded_reduction_op)

        # Post processing
        init_dims = reduction_root_op.fx_node.expanded_dims
        context[
            (reduction_root_op, get_indexed_dims(init_dims, reduction_root_op), res_idx)
        ] = last_expanded_reduction_op
        new_outputs[op_output_index] = last_expanded_reduction_op.fx_node
        new_reductions.append(last_expanded_reduction_op)
        # for scale_idx in range(0, dim_scaling[reduction.axis]):
        #     dims[reduction.axis] = scale_idx
        #     context[(reduction_root_op, get_indexed_dims(dims, reduction_root_op), res_idx)] = last_expanded_reduction_op
        # import pdb; pdb.set_trace()
        # If I uncomment below, for some reason max_0_0_3_0 gets updated and removed to max_0_0_0_0
        # for root_user in list(reduction_root_op.fx_node.users):
        #     if get_custom(root_user) in expanded_reductions or isinstance(root_user, Output):
        #         continue
        #     root_user.replace_input_with(reduction_root_op.fx_node, last_expanded_reduction_op.fx_node)
        # TODO: Figure out/check out after every iter here what is happening. graph seems to already be broken right after.
    output.update_arg("return_vals", new_outputs)
    for old_reduc, new_reduc in zip(reduction_root_ops, new_reductions):
        for root_user in list(old_reduc.fx_node.users):
            if get_custom(root_user) in expanded_reductions or isinstance(
                get_custom(root_user), Output
            ):
                continue
            root_user.replace_input_with(old_reduc.fx_node, new_reduc.fx_node)

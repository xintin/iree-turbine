# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexingContext, IndexSequence, IndexSymbol, IndexExpr
from ..ops.wave_ops import (
    get_custom,
    Add,
    Maximum,
    ReduceOp,
    ShuffleOp,
    CustomOp,
    Extract,
    Reduction,
)

from .utils import DCE, subs_idxc
import torch.fx as fx
import math
from typing import Callable
import sympy

TKW_COMBINER = {"sum": Add, "max": Maximum}


def get_graph_node(custom: CustomOp, graph: fx.Graph):
    custom.add_to_graph(graph)
    custom = custom.fx_node
    return custom


def emit_local_reduction(
    binary_fn: Callable, src: list[fx.Node], graph: fx.Graph, local_reduction_size: int
) -> fx.Node:
    init = None
    for i in range(len(src)):
        for j in range(local_reduction_size):
            if init is None:
                init = get_graph_node(Extract(src[i], [j]), graph)
                continue
            cur_slice = get_graph_node(Extract(src[i], [j]), graph)
            init = get_graph_node(binary_fn(init, cur_slice), graph)
    return init


def emit_global_reduction(
    binary_fn: Callable,
    src: fx.Node,
    graph: fx.Graph,
    subgroup_size: int,
    cluster_size: int,
    cluster_stride: int,
) -> fx.Node:
    init = src
    max_offset = cluster_size * cluster_stride
    cur_offset = cluster_stride
    while cur_offset < max_offset:
        shuffle_val = ShuffleOp(init, int(cur_offset), subgroup_size)
        shuffle_node = get_graph_node(shuffle_val, graph)
        init = get_graph_node(binary_fn(init, shuffle_node), graph)
        cur_offset *= 2
    return init


def decompose_reduce_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
    index_map: dict[IndexSymbol, int],
):
    """
    The lowering for multi_reduction is done in two steps:
      1. Local Reduce: Each thread reduces all elements carried by it along
         the reduction dimensions.
      2. Thread Reduce: Each thread reduces result of step 1 across threads
         by doing a butterfly shuffle.
      3. Accumulator Reduce: Each thread reduces it's intermediate reduced
         results with the accumulator it holds.
    """
    # Get reducte nodes.
    reduce_nodes = trace.walk(lambda node: isinstance(get_custom(node), ReduceOp))
    if not reduce_nodes:
        return

    # Setup constraints
    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )
    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, TilingConstraint) or isinstance(c, WorkgroupConstraint)
    }
    subgroup_size = hardware_constraint.threads_per_wave
    for node in reduce_nodes:
        custom = get_custom(node)
        with custom.graph.inserting_before(custom.fx_node):
            reduction_src, reduction_acc, reduction_dim = node.args
            binary_fn = TKW_COMBINER[custom.tkw_op_name]
            if reduction_dim is None:
                raise ValueError(
                    "No reduction dim specified, please specify a reduction dim."
                )
            if not isinstance(reduction_src, list):
                reduction_src = [reduction_src]

            # Local Reduce
            if (
                reduction_dim
                is not get_custom(reduction_src[0]).type.symbolic_shape[-1]
            ):
                raise NotImplementedError(
                    "Only implemented reduction on fastest dimension."
                )

            get_thread_shape = lambda index: max(
                subs_idxc(x.size) for x in index.values()
            )
            local_reduction_size = get_thread_shape(get_custom(reduction_src[0]).index)
            local_reduction = emit_local_reduction(
                binary_fn, reduction_src, custom.graph, local_reduction_size
            )

            # Global Reduce
            vector_size = reduction_src[0].index[reduction_dim].size
            if not isinstance(vector_size, (sympy.Integer, int)):
                raise NotImplementedError(
                    "Cannot handle non integer stride for index triplet."
                )
            vector_size = int(vector_size)
            cluster_size = 64
            cluster_stride = 1
            if vector_size != 1:
                cluster_stride = subgroup_size / vector_size
                cluster_size = subgroup_size / cluster_stride
            if cluster_size * cluster_stride > subgroup_size:
                raise ValueError(
                    "ReduceOp is illformed as cluster specified is > threads per wave."
                )
            global_reduction = emit_global_reduction(
                binary_fn,
                local_reduction,
                custom.graph,
                subgroup_size,
                cluster_size,
                cluster_stride,
            )

            # Local Accumulator Reduce
            final_reduction = global_reduction
            if reduction_acc is not None:
                final_reduction = get_graph_node(
                    binary_fn(reduction_acc, global_reduction), custom.graph
                )

            # Replace all uses with global reduction
            custom.replace_all_uses_with(final_reduction)

    DCE(trace)

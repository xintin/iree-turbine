# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
import functools
from typing import Any, Callable, ClassVar, Optional, List, Type, Dict

import torch.fx as fx

from ...compiler.ir import (
    Attribute,
    DenseElementsAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    IrType,
    OpResult,
    Value,
    VectorType,
    arith_d,
    vector_d,
    memref_d,
    ShapedType,
    MemRefType,
    amdgpu_d,
)

from ...compiler.utils import strides_from_symbolic_shape
from ...compiler.builder import IRProxyValue
from ...compiler.vector_codegen import (
    cast_kernel_buffer,
    cast_py_literal,
    cast_vector,
)

from ...ops.wave_ops import (
    get_custom,
    read,
    write,
)

from ..utils import safe_subs, subs_idxc, find_index_bounds

from ..._support.indexing import IndexingContext, IndexExpr, IndexSequence, index_symbol
from ...lang.wave_types import IndexMapping
from ...lang.global_symbols import *

from .emitter import (
    WaveEmitter,
    handle_op,
    get_type_or_element_type,
    add_emitter_subs,
    gen_sympy_index,
    get_constant_attr,
)


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr]
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def _split_index(src: IndexExpr | int) -> tuple[IndexExpr, IndexExpr]:
    """
    Split index expr into thread-dependent and thread-independent parts
    """
    subs_wg = {WORKGROUP_0: 0, WORKGROUP_1: 0, WORKGROUP_2: 0}
    subs_th = {THREAD_0: 0, THREAD_1: 0, THREAD_2: 0}
    # Replace all wg symbols with 0s to get thread-dependent index.
    # All dynamic values will also be part of thread-index.
    thread_dependend_index = safe_subs(src, subs_wg)

    # Compute thread-independent index as `orig_index - thread_dependend_index`
    # All thread symbols should cancel-out in the result, but to be sure
    # replace all thread symbols by 0 in the result.
    # We cannot just replace all thread symbols without the subtraction as
    # any constant or dynamic values will end up in both expressions.
    thread_indepdndent_index = sympy.simplify(
        safe_subs(src - thread_dependend_index, subs_th)
    )
    return thread_indepdndent_index, thread_dependend_index


def _build_start_indices(
    emitter: WaveEmitter,
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
    dynamic_values: dict[IndexExpr, Any] = {},
) -> tuple[list[OpResult], list[OpResult], list[OpResult]]:
    start_indices = _get_start_indices(src_indices)
    split_indices = [_split_index(i) for i in start_indices]
    subs = add_emitter_subs(emitter, dynamic_values)
    indices = [gen_sympy_index(subs, i) for i in start_indices]
    indices_wg = [gen_sympy_index(subs, i[0]) for i in split_indices]
    indices_th = [gen_sympy_index(subs, i[1]) for i in split_indices]

    return indices, indices_wg, indices_th


def _get_fastest_index(indices: dict[IndexExpr, IndexSequence]):
    """
    This function takes in indices of a Node, extract their sizes
    into a list, and then try do an argmax on it. In the case where
    there are multipled max_vals we pick the fastest/most minor one.
    """

    index_sizes = [subs_idxc(i.size) for i in indices.values()]
    # Find the maximum value
    max_size = max(index_sizes)
    # Find the fastest/most minor index of the maximum value.
    return max(i for i, size in enumerate(index_sizes) if size == max_size)


def _compute_offset(indices: list[IndexExpr], strides: list[IndexExpr]) -> IndexExpr:
    return sum(i * s for i, s in zip(indices, strides))


def _get_symbolic_shape(node: fx.Node) -> tuple[IndexExpr]:
    return get_custom(node).type.symbolic_shape


def _build_mask(
    emitter: WaveEmitter, index: Dict[IndexExpr, IndexExpr], elements_per_thread: int
) -> Optional[OpResult]:
    bounds = find_index_bounds(emitter.constraints, index)
    if bounds is None:
        return None

    idxc = IndexingContext.current()
    fastest_dim = _get_fastest_index(index)
    last_dim = list(index)[fastest_dim]
    new_index = {k: _get_start_index(v) for k, v in index.items()}

    new_index[last_dim] = new_index[last_dim] + idxc.iota(elements_per_thread)

    mask_expr = functools.reduce(
        lambda a, b: sympy.And(a, b), (new_index[dim] < dim for dim in bounds)
    )
    mask = gen_sympy_index(add_emitter_subs(emitter), mask_expr)

    mask_vec_type = VectorType.get([elements_per_thread], IntegerType.get_signless(1))
    if mask.type != mask_vec_type:
        mask = vector_d.splat(mask_vec_type, mask)

    return mask


def _get_splat_const(vec_type: IrType, value: Any) -> Value:
    splat = DenseElementsAttr.get_splat(
        vec_type, get_constant_attr(value, vec_type.element_type)
    )
    return arith_d.constant(vec_type, splat)


def _constant_mask(vec_type: IrType) -> Value:
    return _get_splat_const(vec_type, 1)


def _construct_gather_scatter_indices(
    emitter: WaveEmitter,
    symbolc_shape: tuple[IndexExpr],
    index: tuple[IndexExpr],
    mapping: IndexMapping,
    elements_per_thread: int,
    is_read: bool,
    dynamic_vals: tuple[Any, ...],
    is_contiguous: bool,
) -> tuple[list[OpResult], list[OpResult], list[OpResult], OpResult, OpResult]:
    # Apply symbolc_shape order to indices, e.g. if original mapping is
    # {M: iter(0), N: iter(1)} and symbolc_shape is (N, M), result will
    # be (iter(1), iter(0))
    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        index_mapping = mapping.map_input_indices(symbolc_shape)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolc_shape)

    idxc = IndexingContext.current()
    index_mapping = tuple(i.subs(idxc.subs) for i in index_mapping)

    iters = mapping.iters

    # As we only support identity input/output mapping for now, we can directly
    # substitute iterators with corresponding expanded index.
    subs = [
        (sym, expr.start) for sym, expr in zip(iters.keys(), index.values())
    ] + list(idxc.subs.items())

    # Contruct input/output index, substituting iterators in input mapping with
    # expanded index.
    result_index = {key: m.subs(subs) for key, m in zip(symbolc_shape, index_mapping)}

    mask = _build_mask(emitter, index, elements_per_thread)
    if mask is None:
        mask_vec_type = VectorType.get(
            [elements_per_thread], IntegerType.get_signless(1)
        )
        mask = _constant_mask(mask_vec_type)

    def extract0(src):
        static_pos = [0] * src.type.rank
        return vector_d.extract(src, static_position=static_pos, dynamic_position=[])

    dynamic_vals_map_start = {
        sym: extract0(val)
        for sym, val in zip(mapping.dynamic_val_indices.keys(), dynamic_vals)
    }
    if is_contiguous:
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, result_index, dynamic_vals_map_start
        )
        return start_indices, start_indices_wg, start_indices_th, None, mask

    start_indices = _get_start_indices(result_index)
    start_indices_orig = _get_start_indices(index)
    fastest_dim = _get_fastest_index(index)
    need_dynamic_offsets = False
    for val in dynamic_vals:
        shape = val.type.shape
        assert shape in (
            [1],
            [elements_per_thread],
        ), f"Dynamic val shape must be {[1]} or {[elements_per_thread]} but got {shape}"
        if shape[0] > 1:
            need_dynamic_offsets = True

    offsets = []
    strides = strides_from_symbolic_shape(idxc, symbolc_shape, allow_mixed_shapes=True)
    start_indices_offset = _compute_offset(start_indices, strides)
    for i in range(elements_per_thread):
        # Update fastest dim, i.e. in case of identity mapping it will
        # be equivalent to just vector.load
        subs = [(sym, idx) for sym, idx in zip(iters.keys(), start_indices_orig)]
        subs[fastest_dim] = (subs[fastest_dim][0], start_indices_orig[fastest_dim] + i)
        indices = [i.subs(subs) for i in index_mapping]

        # First, we build indices as if resulting gather/scatter `start_indices`
        # are 0 as mapping expression may depend on absolute value of index
        # (e.g. `index % 32`). Then we adjust for the non-0 `start_indices` by
        # subtracting computed previously linear `start_indices_offset`. For
        # simple cases like transpose, the resulting expression should fold into
        # simple constant while more complex expressions may requires actual
        # arith ops on dynamic values.
        offset = _compute_offset(indices, strides) - start_indices_offset
        offset = subs_idxc(offset)

        if offset.is_number:
            # If resulted offset sympy expr is convertible to int constant it
            # will be directly encoded into `arith.constant`.
            # For non-constant expressions, we will generate a real sequence of
            # arith ops and then `vector.insertelement` them into offsets vec.
            offset = int(offset)
        else:
            need_dynamic_offsets = True
            break

        offsets.append(offset)

    offsets_vec_type = VectorType.get([elements_per_thread], IndexType.get())
    if need_dynamic_offsets:
        # In case we need dynamic `offsets_vec`, set all `start_indices` to 0
        # and encode entire index info in `offsets_vec`.
        result_index = {key: 0 for key in symbolc_shape}
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, result_index, dynamic_vals_map_start
        )
        subs = [(sym, idx) for sym, idx in zip(iters.keys(), start_indices_orig)]
        # Last item in `subs` corresponds to last item in `start_indices_orig`
        # which is fastest changing dim.
        # Replacing last element with `idxc.iota(elements_per_thread)` will
        # generate vectorized index code, each element in it corresponding to
        # individual vector element index.
        subs[-1] = (
            subs[-1][0],
            start_indices_orig[-1] + idxc.iota(elements_per_thread),
        )
        dynamic_vals_map = {
            sym: val
            for sym, val in zip(mapping.dynamic_val_indices.keys(), dynamic_vals)
        }
        indices = [i.subs(subs) for i in index_mapping]
        offsets_vec = gen_sympy_index(
            add_emitter_subs(emitter, dynamic_vals_map),
            _compute_offset(indices, strides),
        )
    else:
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, result_index, dynamic_vals_map_start
        )
        if offsets == list(range(elements_per_thread)):
            return start_indices, start_indices_wg, start_indices_th, None, mask

        offsets = [IntegerAttr.get(IndexType.get(), off) for off in offsets]
        offsets_vec = arith_d.ConstantOp(
            offsets_vec_type, DenseElementsAttr.get(offsets, offsets_vec_type)
        )

    return start_indices, start_indices_wg, start_indices_th, offsets_vec, mask


def _get_max_buffer_size(elem_type: IrType) -> int:
    """
    Return max memref size suitable for buffer ops.

    Buffer ops offsets are i32, return maximum memref size in elements.
    """
    return ((1 << 31) - 1) // (elem_type.width // 8)


def _linearize_memref(
    mem: Value,
    offsets_wg: tuple[Value | int],
    offsets_th: tuple[Value | int],
    strides: tuple[Value],
) -> tuple[Value, Value]:
    """
    Convert n-D memref into 1-D memref, suitable for buffer ops.

    Apply offsets to the memref and convert result to 1-D. Resulting memref size
    is set to `max_buffer_size - 1` so buffer access to the last element will be
    no-op.
    """
    memref_type = mem.type
    offset = None
    offset_th = None
    overflow_flags = arith_d.IntegerOverflowFlags.nsw
    for ind_wg, ind_th, stride in zip(offsets_wg, offsets_th, strides):
        if isinstance(ind_wg, int):
            ind_wg = arith_d.constant(IndexType.get(), ind_wg)

        if isinstance(ind_th, int):
            ind_th = arith_d.constant(IndexType.get(), ind_th)

        off_wg = arith_d.muli(ind_wg, stride, overflow_flags=overflow_flags)
        if offset is None:
            offset = off_wg
        else:
            offset = arith_d.addi(offset, off_wg, overflow_flags=overflow_flags)

        off_th = arith_d.muli(ind_th, stride, overflow_flags=overflow_flags)
        if offset_th is None:
            offset_th = off_th
        else:
            offset_th = arith_d.addi(offset_th, off_th, overflow_flags=overflow_flags)

    size_full = arith_d.constant(
        IndexType.get(), _get_max_buffer_size(memref_type.element_type) - 1
    )

    dyn_val = ShapedType.get_dynamic_size()
    res_shape = [dyn_val]
    element_type = memref_type.element_type
    memory_space = memref_type.memory_space
    resut_type = MemRefType.get(
        res_shape,
        element_type,
        layout=Attribute.parse("strided<[1], offset: ?>"),
        memory_space=memory_space,
    )
    return (
        memref_d.reinterpret_cast(
            resut_type,
            mem,
            offsets=[offset],
            sizes=[size_full],
            strides=[],
            static_offsets=[dyn_val],
            static_sizes=[dyn_val],
            static_strides=[1],
        ),
        offset_th,
    )


def _create_vec_read(
    emitter: WaveEmitter,
    symbolic_shape: tuple[IndexExpr, ...],
    mem: Value,
    vector_type: IrType,
    start_indices: tuple[Value],
    start_indices_wg: tuple[Value],
    start_indices_th: tuple[Value],
    elements_per_thread: int,
    mask: Optional[Value],
    offsets_vec: Optional[Value],
) -> Value:
    if mask is None and offsets_vec is None:
        return vector_d.load(vector_type, mem, start_indices)

    # Only use buffer ops if it's gather/scatter on global mem.
    use_buffer_ops = offsets_vec is not None and mem.type.memory_space is None

    element_type = vector_type.element_type
    if offsets_vec is None:
        offsets_vec_type = VectorType.get(vector_type.shape, IndexType.get())
        vals = [IntegerAttr.get(IndexType.get(), v) for v in range(elements_per_thread)]
        offsets_vec = arith_d.constant(
            offsets_vec_type, DenseElementsAttr.get(vals, offsets_vec_type)
        )

    zero = get_constant_attr(0, element_type)
    zero = arith_d.constant(element_type, zero)

    strides = strides_from_symbolic_shape(
        IndexingContext.current(), symbolic_shape, allow_mixed_shapes=True
    )
    buffer_ops_enabled = emitter.params.get("use_buffer_load_ops", False)
    has_int_strides = all(isinstance(s, int) for s in strides)
    if buffer_ops_enabled and has_int_strides and use_buffer_ops:
        result = vector_d.splat(vector_type, zero)

        strides = [gen_sympy_index(add_emitter_subs(emitter), s) for s in strides]
        data, offset_th = _linearize_memref(
            mem, start_indices_wg, start_indices_th, strides
        )
        offset_th = vector_d.splat(offsets_vec.type, offset_th)
        offsets_vec = arith_d.addi(offsets_vec, offset_th)
        if mask is not None:
            i32 = IntegerType.get_signless(32)
            i32vec = VectorType.get([elements_per_thread], i32)
            offsets_vec = arith_d.index_cast(i32vec, offsets_vec)
            oob_idx = _get_max_buffer_size(element_type)
            oob_idx = arith_d.constant(i32, oob_idx)
            oob_idx = vector_d.splat(offsets_vec.type, oob_idx)
            offsets_vec = arith_d.select(mask, offsets_vec, oob_idx)

        for i in range(elements_per_thread):
            offset = vector_d.extract(
                offsets_vec, static_position=[i], dynamic_position=[]
            )
            if mask is None:
                elem = memref_d.load(element_type, data, indices=[offset])
            else:
                elem = amdgpu_d.raw_buffer_load(element_type, data, indices=[offset])

            result = vector_d.insert(
                elem, result, static_position=[i], dynamic_position=[]
            )

        return result

    else:
        passthru = vector_d.splat(vector_type, zero)

        if mask is None:
            mask_vec_type = VectorType.get(
                [elements_per_thread], IntegerType.get_signless(1)
            )
            mask = _constant_mask(mask_vec_type)

        return vector_d.gather(
            vector_type, mem, start_indices, offsets_vec, mask, passthru
        )


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, mapping, dyn_vals, _ = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    input_shape = _get_symbolic_shape(memory)
    elements_per_thread = cast_py_literal(emitter, elements_per_thread)
    if get_custom(node).has_identity_mapping():
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, index
        )
        mask = _build_mask(
            emitter,
            index,
            elements_per_thread,
        )
        result = _create_vec_read(
            emitter,
            input_shape,
            kb_src,
            vector_type,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            mask,
            offsets_vec=None,
        )
    else:
        dyn_vals = tuple(
            cast_vector(emitter, reg, element_type=IndexType.get()) for reg in dyn_vals
        )
        (
            start_indices,
            start_indices_wg,
            start_indices_th,
            offsets_vec,
            mask,
        ) = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolc_shape=input_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=elements_per_thread,
            is_read=True,
            dynamic_vals=dyn_vals,
            is_contiguous=get_custom(node).is_contiguous_vec(),
        )
        result = _create_vec_read(
            emitter,
            input_shape,
            kb_src,
            vector_type,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            mask,
            offsets_vec,
        )

    emitter.bind_node_proxy(node, IRProxyValue(result))


def _create_vec_write(
    emitter: WaveEmitter,
    symbolic_shape: tuple[IndexExpr, ...],
    mem: Value,
    value: Value,
    start_indices: tuple[Value],
    start_indices_wg: tuple[Value],
    start_indices_th: tuple[Value],
    elements_per_thread: int,
    mask: Optional[Value],
    offsets_vec: Optional[Value],
):
    if mask is None and offsets_vec is None:
        vector_d.store(value, mem, start_indices)
        return

    # Only use buffer ops if it's gather/scatter on global mem.
    use_buffer_ops = offsets_vec is not None and mem.type.memory_space is None

    vector_type = value.type
    element_type = vector_type.element_type
    if offsets_vec is None:
        offsets_vec_type = VectorType.get(vector_type.shape, IndexType.get())
        vals = [IntegerAttr.get(IndexType.get(), v) for v in range(elements_per_thread)]
        offsets_vec = arith_d.constant(
            offsets_vec_type, DenseElementsAttr.get(vals, offsets_vec_type)
        )

    strides = strides_from_symbolic_shape(
        IndexingContext.current(), symbolic_shape, allow_mixed_shapes=True
    )
    buffer_ops_enabled = emitter.params.get("use_buffer_store_ops", False)
    has_int_strides = all(isinstance(s, int) for s in strides)
    if buffer_ops_enabled and has_int_strides and use_buffer_ops:
        strides = [gen_sympy_index(add_emitter_subs(emitter), s) for s in strides]
        data, offset_th = _linearize_memref(
            mem, start_indices_wg, start_indices_th, strides
        )
        offset_th = vector_d.splat(offsets_vec.type, offset_th)
        offsets_vec = arith_d.addi(offsets_vec, offset_th)
        if mask is not None:
            i32 = IntegerType.get_signless(32)
            i32vec = VectorType.get([elements_per_thread], i32)
            offsets_vec = arith_d.index_cast(i32vec, offsets_vec)
            oob_idx = _get_max_buffer_size(element_type)
            oob_idx = arith_d.constant(i32, oob_idx)
            oob_idx = vector_d.splat(offsets_vec.type, oob_idx)
            offsets_vec = arith_d.select(mask, offsets_vec, oob_idx)

        for i in range(elements_per_thread):
            offset = vector_d.extract(
                offsets_vec, static_position=[i], dynamic_position=[]
            )
            elem = vector_d.extract(value, static_position=[i], dynamic_position=[])
            if mask is None:
                memref_d.store(elem, data, indices=[offset])
            else:
                amdgpu_d.raw_buffer_store(elem, data, indices=[offset])

    else:
        if mask is None:
            mask_vec_type = VectorType.get(
                [elements_per_thread], IntegerType.get_signless(1)
            )
            mask = _constant_mask(mask_vec_type)

        vector_d.scatter(mem, start_indices, offsets_vec, mask, value)


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    try:
        register, memory, elements_per_thread, mapping, dyn_vals = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # memory has no IR node yet.
    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)
    insert_vector = cast_vector(emitter, register, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    vector_shape = cast_py_literal(emitter, (elements_per_thread,))

    # TODO: Support elements_per_thread size mismatch and broadcasting

    assert (
        tuple(insert_type.shape) == vector_shape
    ), f"Shape doesn't match: {tuple(insert_type.shape)} and {(vector_shape)}"

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected write to have index attr.")

    index = node.index

    input_shape = _get_symbolic_shape(register)
    output_shape = _get_symbolic_shape(memory)
    elements_per_thread = cast_py_literal(emitter, elements_per_thread)
    if get_custom(node).has_identity_mapping():
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, index
        )
        mask = _build_mask(emitter, index, elements_per_thread)
        _create_vec_write(
            emitter,
            output_shape,
            kb_dest,
            insert_vector,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            mask,
            offsets_vec=None,
        )
    else:
        assert (
            input_shape == mapping.input_shape
        ), "non-identity input mapping is not supported yet"

        dyn_vals = tuple(
            cast_vector(emitter, reg, element_type=IndexType.get()) for reg in dyn_vals
        )
        (
            start_indices,
            start_indices_wg,
            start_indices_th,
            offsets_vec,
            mask,
        ) = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolc_shape=output_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=elements_per_thread,
            is_read=False,
            dynamic_vals=dyn_vals,
            is_contiguous=get_custom(node).is_contiguous_vec(),
        )

        _create_vec_write(
            emitter,
            output_shape,
            kb_dest,
            insert_vector,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            mask,
            offsets_vec,
        )

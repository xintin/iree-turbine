import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
import torch
from numpy.testing import assert_allclose, assert_equal
import pytest
import os
import math
import sympy
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.indexing import IndexSymbol


def test_nested_reduction_gemm():
    shape = (128, 128, 32)

    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = IndexSymbol("K1")
    K2 = IndexSymbol("K2")
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = IndexSymbol("BLOCK_K1")
    BLOCK_K2 = IndexSymbol("BLOCK_K2")
    # Q: [M, K1] -> [seq_len, ]
    # K: [K2, K1] ->
    # V: [K2, N] -> [b[num_head * batch_seq], seq_len, head_dim]
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.TilingConstraint(K1, K1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    ]

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[N, M, tkl.f32](0.0)
        init_sum = tkl.Register[M, tkl.f32](0.0)
        init_max = tkl.Register[M, tkl.f32](-1e6)
        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_sum: tkl.Register[M, tkl.f32],
            acc: tkl.Register[N, M, tkl.f32],
        ) -> (
            tkl.Register[M, tkl.f32],
            tkl.Register[M, tkl.f32],
            tkl.Register[N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[K2, M, tkl.f32](0.0)
            # acc: tkw.Register[N, M, tkl.f32]
            @tkw.reduction(K1, init_args=[imm_reg])
            def inner_loop(
                inner_acc: tkl.Register[K2, M, tkl.f32]
            ) -> tkl.Register[K2, M, tkl.f32]:
                # a_reg: tkw.Register[M, K, tkl.f16]
                q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                # b_reg: tkw.Register[N, K, tkl.f16]
                k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                # acc: tkw.Register[N, M, tkl.f32]
                inner_acc = tkw.mma(k_reg, q_reg, inner_acc)
                return inner_acc

            x_j = tkw.transpose(inner_loop, dims=[M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.trunc(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        M: shape[0],
        N: shape[1],
        K1: 16,
        K2: shape[2],
    }
    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}
    with tk.gen.TestLaunchContext(
        hyperparams, canonicalize=True, run=True, run_config=config
    ):
        m_dim = shape[0]
        k1_dim = 16
        k2_dim = shape[2]
        n_dim = shape[1]
        torch.manual_seed(0)
        q = torch.randn(m_dim, k1_dim, dtype=torch.float16)
        k = torch.randn(k2_dim, k1_dim, dtype=torch.float16)
        v = torch.randn(k2_dim, n_dim, dtype=torch.float16)
        c = torch.zeros(m_dim, n_dim, dtype=torch.float32)
        d = torch.zeros(m_dim, n_dim, dtype=torch.float32)
        e = torch.zeros(m_dim, n_dim, dtype=torch.float32)
        f = torch.zeros(m_dim, n_dim, dtype=torch.float32)
        # TODO: Fix the max/sum is indeed transposed from the output.
        # TODO: Align Reference kernel and scaled_dot_product_attention
        # To try simple chain matmul, uncomment here:
        log2e = 1.44269504089
        dk_sqrt = 0.25
        mb = base_attention(q * dk_sqrt * log2e, k, v.T, c)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)

        # with open("attention_all_scale.mlir", "w") as f:
        #     f.write(str(mb.module_op))

        assert_allclose(ref, c, atol=3e-2)
        print("SUCCESS")


if __name__ == "__main__":
    # test_reduce_non_iv_acc()
    # test_gemm()
    test_nested_reduction_gemm()
    # test_single_gemm()
    # test_partial_reduce_elemwise()

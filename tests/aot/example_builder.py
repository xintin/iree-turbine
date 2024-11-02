# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

from iree.build import *
from iree.turbine.aot.build_actions import *
from iree.turbine.aot import FxProgramsBuilder


def export_simple_model() -> FxProgramsBuilder:
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example_args = (torch.randn(32, 64), torch.randn(32, 128))

    # Create a dynamic batch size
    batch = torch.export.Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    fxb = FxProgramsBuilder(M())

    @fxb.export_program(args=example_args, dynamic_shapes=dynamic_shapes)
    def dynamic_batch(module: M, x1, x2):
        return module.forward(x1, x2)

    return fxb


@entrypoint(description="Builds an awesome pipeline")
def pipe():
    results = []
    for i in range(3):
        turbine_generate(
            name=f"import_stage{i}",
            generator=export_simple_model,
        )
        results.extend(
            compile(
                name=f"stage{i}",
                source=f"import_stage{i}.mlir",
            )
        )
    return results


if __name__ == "__main__":
    iree_build_main()

#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @base_ref_gemm {
    stream.executable.export public @base_ref_gemm workgroups() -> (index, index, index) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      stream.return %c2, %c2, %c1 : index, index, index
    }
    builtin.module {
      func.func @base_ref_gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c18 = arith.constant 18 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c16_i32 = arith.constant 16 : i32
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_1 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %alloc = memref.alloc() : memref<64x36xf16, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x16xf16, strided<[16, 1], offset: ?>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<32x16xf16, strided<[16, 1], offset: ?>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x32xf16, strided<[32, 1], offset: ?>>
        %3 = arith.divsi %thread_id_x, %c64 : index
        %4 = arith.muli %3, %c32 : index
        %5 = arith.muli %workgroup_id_0, %c64 : index
        %6 = arith.remsi %thread_id_x, %c16 : index
        %7 = arith.addi %6, %5 : index
        %8 = arith.addi %7, %4 : index
        %9 = arith.remsi %thread_id_x, %c64 : index
        %10 = arith.divsi %9, %c16 : index
        %11 = arith.muli %10, %c4 : index
        %12 = vector.load %0[%8, %11] : memref<128x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %13 = arith.addi %8, %c16 : index
        %14 = vector.load %0[%13, %11] : memref<128x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %alloc_2 = memref.alloc() : memref<64x20xf16, #gpu.address_space<workgroup>>
        %15 = arith.addi %6, %4 : index
        vector.store %12, %alloc_2[%15, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %16 = arith.addi %15, %c16 : index
        vector.store %14, %alloc_2[%16, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        amdgpu.lds_barrier
        %17 = vector.load %alloc_2[%15, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %18 = vector.load %alloc_2[%16, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %19 = vector.load %1[%6, %11] : memref<32x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %20 = arith.addi %6, %c16 : index
        %21 = vector.load %1[%20, %11] : memref<32x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %alloc_3 = memref.alloc() : memref<32x20xf16, #gpu.address_space<workgroup>>
        amdgpu.lds_barrier
        vector.store %19, %alloc_3[%6, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        vector.store %21, %alloc_3[%20, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        amdgpu.lds_barrier
        %22 = vector.load %alloc_3[%6, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %23 = vector.load %alloc_3[%20, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %24 = amdgpu.mfma %22 * %17 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %25 = amdgpu.mfma %23 * %18 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %26 = amdgpu.mfma %22 * %18 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %27 = amdgpu.mfma %23 * %17 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %28 = vector.extract_strided_slice %24 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %29 = vector.extract_strided_slice %24 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %30 = arith.maximumf %28, %29 : vector<1xf32>
        %31 = vector.extract_strided_slice %24 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %32 = arith.maximumf %30, %31 : vector<1xf32>
        %33 = vector.extract_strided_slice %24 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %34 = arith.maximumf %32, %33 : vector<1xf32>
        %35 = vector.extract_strided_slice %27 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %36 = arith.maximumf %34, %35 : vector<1xf32>
        %37 = vector.extract_strided_slice %27 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %38 = arith.maximumf %36, %37 : vector<1xf32>
        %39 = vector.extract_strided_slice %27 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %40 = arith.maximumf %38, %39 : vector<1xf32>
        %41 = vector.extract_strided_slice %27 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %42 = arith.maximumf %40, %41 : vector<1xf32>
        %43 = vector.extract %42[0] : f32 from vector<1xf32>
        %shuffleResult, %valid = gpu.shuffle  xor %43, %c16_i32, %c64_i32 : f32
        %44 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
        %45 = arith.maximumf %42, %44 : vector<1xf32>
        %46 = vector.extract %45[0] : f32 from vector<1xf32>
        %shuffleResult_4, %valid_5 = gpu.shuffle  xor %46, %c32_i32, %c64_i32 : f32
        %47 = vector.broadcast %shuffleResult_4 : f32 to vector<1xf32>
        %48 = arith.maximumf %45, %47 : vector<1xf32>
        %49 = arith.maximumf %48, %cst : vector<1xf32>
        %50 = vector.extract_strided_slice %26 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %51 = vector.extract_strided_slice %26 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %52 = arith.maximumf %50, %51 : vector<1xf32>
        %53 = vector.extract_strided_slice %26 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %54 = arith.maximumf %52, %53 : vector<1xf32>
        %55 = vector.extract_strided_slice %26 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %56 = arith.maximumf %54, %55 : vector<1xf32>
        %57 = vector.extract_strided_slice %25 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %58 = arith.maximumf %56, %57 : vector<1xf32>
        %59 = vector.extract_strided_slice %25 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %60 = arith.maximumf %58, %59 : vector<1xf32>
        %61 = vector.extract_strided_slice %25 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %62 = arith.maximumf %60, %61 : vector<1xf32>
        %63 = vector.extract_strided_slice %25 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %64 = arith.maximumf %62, %63 : vector<1xf32>
        %65 = vector.extract %64[0] : f32 from vector<1xf32>
        %shuffleResult_6, %valid_7 = gpu.shuffle  xor %65, %c16_i32, %c64_i32 : f32
        %66 = vector.broadcast %shuffleResult_6 : f32 to vector<1xf32>
        %67 = arith.maximumf %64, %66 : vector<1xf32>
        %68 = vector.extract %67[0] : f32 from vector<1xf32>
        %shuffleResult_8, %valid_9 = gpu.shuffle  xor %68, %c32_i32, %c64_i32 : f32
        %69 = vector.broadcast %shuffleResult_8 : f32 to vector<1xf32>
        %70 = arith.maximumf %67, %69 : vector<1xf32>
        %71 = arith.maximumf %70, %cst_0 : vector<1xf32>
        %72 = arith.subf %cst, %49 : vector<1xf32>
        %73 = arith.subf %cst_0, %71 : vector<1xf32>
        %74 = math.exp2 %72 : vector<1xf32>
        %75 = math.exp2 %73 : vector<1xf32>
        %76 = vector.extract %49[0] : f32 from vector<1xf32>
        %77 = vector.splat %76 : vector<4xf32>
        %78 = arith.subf %24, %77 : vector<4xf32>
        %79 = arith.subf %27, %77 : vector<4xf32>
        %80 = vector.extract %71[0] : f32 from vector<1xf32>
        %81 = vector.splat %80 : vector<4xf32>
        %82 = arith.subf %26, %81 : vector<4xf32>
        %83 = arith.subf %25, %81 : vector<4xf32>
        %84 = math.exp2 %78 : vector<4xf32>
        %85 = math.exp2 %79 : vector<4xf32>
        %86 = math.exp2 %82 : vector<4xf32>
        %87 = math.exp2 %83 : vector<4xf32>
        %88 = arith.truncf %84 : vector<4xf32> to vector<4xf16>
        %89 = arith.truncf %85 : vector<4xf32> to vector<4xf16>
        %90 = arith.truncf %86 : vector<4xf32> to vector<4xf16>
        %91 = arith.truncf %87 : vector<4xf32> to vector<4xf16>
        %92 = arith.muli %workgroup_id_1, %c64 : index
        %93 = arith.muli %thread_id_y, %c32 : index
        %94 = arith.divsi %thread_id_x, %c4 : index
        %95 = arith.addi %94, %93 : index
        %96 = arith.remsi %95, %c64 : index
        %97 = arith.addi %96, %92 : index
        %98 = arith.remsi %thread_id_x, %c4 : index
        %99 = arith.muli %98, %c8 : index
        %100 = vector.load %2[%97, %99] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<8xf16>
        amdgpu.lds_barrier
        vector.store %100, %alloc[%96, %99] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        %101 = arith.addi %6, %93 : index
        %102 = vector.load %alloc[%101, %11] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %103 = arith.addi %11, %c16 : index
        %104 = vector.load %alloc[%101, %103] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %105 = arith.addi %101, %c16 : index
        %106 = vector.load %alloc[%105, %11] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %107 = vector.load %alloc[%105, %103] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %108 = vector.extract %74[0] : f32 from vector<1xf32>
        %109 = vector.splat %108 : vector<4xf32>
        %110 = arith.mulf %109, %cst_1 : vector<4xf32>
        %111 = vector.extract %75[0] : f32 from vector<1xf32>
        %112 = vector.splat %111 : vector<4xf32>
        %113 = arith.mulf %112, %cst_1 : vector<4xf32>
        %114 = amdgpu.mfma %88 * %102 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %115 = amdgpu.mfma %89 * %104 + %114 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %116 = amdgpu.mfma %90 * %106 + %113 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %117 = amdgpu.mfma %91 * %107 + %116 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %118 = amdgpu.mfma %90 * %102 + %113 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %119 = amdgpu.mfma %91 * %104 + %118 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %120 = amdgpu.mfma %88 * %106 + %110 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %121 = amdgpu.mfma %89 * %107 + %120 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %122 = vector.extract_strided_slice %115 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %123 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<128x128xf32, strided<[128, 1], offset: ?>>
        %124 = arith.addi %5, %4 : index
        %125 = arith.addi %124, %11 : index
        %126 = arith.addi %6, %92 : index
        %127 = arith.addi %126, %93 : index
        vector.store %122, %123[%125, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %128 = vector.extract_strided_slice %115 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %129 = arith.addi %125, %c1 : index
        vector.store %128, %123[%129, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %130 = vector.extract_strided_slice %115 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %131 = arith.addi %125, %c2 : index
        vector.store %130, %123[%131, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %132 = vector.extract_strided_slice %115 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %133 = arith.addi %125, %c3 : index
        vector.store %132, %123[%133, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %134 = vector.extract_strided_slice %117 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %135 = arith.addi %125, %c16 : index
        %136 = arith.addi %127, %c16 : index
        vector.store %134, %123[%135, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %137 = vector.extract_strided_slice %117 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %138 = arith.addi %125, %c17 : index
        vector.store %137, %123[%138, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %139 = vector.extract_strided_slice %117 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %140 = arith.addi %125, %c18 : index
        vector.store %139, %123[%140, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %141 = vector.extract_strided_slice %117 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %142 = arith.addi %125, %c19 : index
        vector.store %141, %123[%142, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %143 = vector.extract_strided_slice %119 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %143, %123[%135, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %144 = vector.extract_strided_slice %119 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %144, %123[%138, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %145 = vector.extract_strided_slice %119 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %145, %123[%140, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %146 = vector.extract_strided_slice %119 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %146, %123[%142, %127] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %147 = vector.extract_strided_slice %121 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %147, %123[%125, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %148 = vector.extract_strided_slice %121 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %148, %123[%129, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %149 = vector.extract_strided_slice %121 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %149, %123[%131, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %150 = vector.extract_strided_slice %121 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %150, %123[%133, %136] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<128x16xf16>, %arg1: tensor<32x16xf16>, %arg2: tensor<128x32xf16>) -> tensor<128x128xf32> {
    %0 = flow.dispatch @base_ref_gemm::@base_ref_gemm(%arg0, %arg1, %arg2) : (tensor<128x16xf16>, tensor<32x16xf16>, tensor<128x32xf16>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}

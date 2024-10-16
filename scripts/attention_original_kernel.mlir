#translation = #iree_codegen.translation_info<None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      stream.return %c2, %c2, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding) attributes {translation_info = #translation} {
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
        %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_0 = arith.constant dense<-1.000000e+06> : vector<1xf32>
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
        %49 = arith.maximumf %48, %cst_0 : vector<1xf32>
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
        %72 = arith.subf %cst_0, %49 : vector<1xf32>
        %73 = arith.subf %cst_0, %71 : vector<1xf32>
        %74 = math.exp2 %72 : vector<1xf32>
        %75 = math.exp2 %73 : vector<1xf32>
        %76 = arith.mulf %74, %cst : vector<1xf32>
        %77 = arith.mulf %75, %cst : vector<1xf32>
        %78 = vector.extract %49[0] : f32 from vector<1xf32>
        %79 = vector.splat %78 : vector<4xf32>
        %80 = arith.subf %24, %79 : vector<4xf32>
        %81 = arith.subf %27, %79 : vector<4xf32>
        %82 = vector.extract %71[0] : f32 from vector<1xf32>
        %83 = vector.splat %82 : vector<4xf32>
        %84 = arith.subf %26, %83 : vector<4xf32>
        %85 = arith.subf %25, %83 : vector<4xf32>
        %86 = math.exp2 %80 : vector<4xf32>
        %87 = math.exp2 %81 : vector<4xf32>
        %88 = math.exp2 %84 : vector<4xf32>
        %89 = math.exp2 %85 : vector<4xf32>
        %90 = vector.extract_strided_slice %86 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %91 = vector.extract_strided_slice %86 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %92 = arith.addf %90, %91 : vector<1xf32>
        %93 = vector.extract_strided_slice %86 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %94 = arith.addf %92, %93 : vector<1xf32>
        %95 = vector.extract_strided_slice %86 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %96 = arith.addf %94, %95 : vector<1xf32>
        %97 = vector.extract_strided_slice %87 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %98 = arith.addf %96, %97 : vector<1xf32>
        %99 = vector.extract_strided_slice %87 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %100 = arith.addf %98, %99 : vector<1xf32>
        %101 = vector.extract_strided_slice %87 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %102 = arith.addf %100, %101 : vector<1xf32>
        %103 = vector.extract_strided_slice %87 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %104 = arith.addf %102, %103 : vector<1xf32>
        %105 = vector.extract %104[0] : f32 from vector<1xf32>
        %shuffleResult_10, %valid_11 = gpu.shuffle  xor %105, %c16_i32, %c64_i32 : f32
        %106 = vector.broadcast %shuffleResult_10 : f32 to vector<1xf32>
        %107 = arith.addf %104, %106 : vector<1xf32>
        %108 = vector.extract %107[0] : f32 from vector<1xf32>
        %shuffleResult_12, %valid_13 = gpu.shuffle  xor %108, %c32_i32, %c64_i32 : f32
        %109 = vector.broadcast %shuffleResult_12 : f32 to vector<1xf32>
        %110 = arith.addf %107, %109 : vector<1xf32>
        %111 = arith.addf %76, %110 : vector<1xf32>
        %112 = vector.extract_strided_slice %88 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %113 = vector.extract_strided_slice %88 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %114 = arith.addf %112, %113 : vector<1xf32>
        %115 = vector.extract_strided_slice %88 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %116 = arith.addf %114, %115 : vector<1xf32>
        %117 = vector.extract_strided_slice %88 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %118 = arith.addf %116, %117 : vector<1xf32>
        %119 = vector.extract_strided_slice %89 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %120 = arith.addf %118, %119 : vector<1xf32>
        %121 = vector.extract_strided_slice %89 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %122 = arith.addf %120, %121 : vector<1xf32>
        %123 = vector.extract_strided_slice %89 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %124 = arith.addf %122, %123 : vector<1xf32>
        %125 = vector.extract_strided_slice %89 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %126 = arith.addf %124, %125 : vector<1xf32>
        %127 = vector.extract %126[0] : f32 from vector<1xf32>
        %shuffleResult_14, %valid_15 = gpu.shuffle  xor %127, %c16_i32, %c64_i32 : f32
        %128 = vector.broadcast %shuffleResult_14 : f32 to vector<1xf32>
        %129 = arith.addf %126, %128 : vector<1xf32>
        %130 = vector.extract %129[0] : f32 from vector<1xf32>
        %shuffleResult_16, %valid_17 = gpu.shuffle  xor %130, %c32_i32, %c64_i32 : f32
        %131 = vector.broadcast %shuffleResult_16 : f32 to vector<1xf32>
        %132 = arith.addf %129, %131 : vector<1xf32>
        %133 = arith.addf %77, %132 : vector<1xf32>
        %134 = arith.truncf %86 : vector<4xf32> to vector<4xf16>
        %135 = arith.truncf %87 : vector<4xf32> to vector<4xf16>
        %136 = arith.truncf %88 : vector<4xf32> to vector<4xf16>
        %137 = arith.truncf %89 : vector<4xf32> to vector<4xf16>
        %138 = arith.muli %workgroup_id_1, %c64 : index
        %139 = arith.muli %thread_id_y, %c32 : index
        %140 = arith.divsi %thread_id_x, %c4 : index
        %141 = arith.addi %140, %139 : index
        %142 = arith.remsi %141, %c64 : index
        %143 = arith.addi %142, %138 : index
        %144 = arith.remsi %thread_id_x, %c4 : index
        %145 = arith.muli %144, %c8 : index
        %146 = vector.load %2[%143, %145] : memref<128x32xf16, strided<[32, 1], offset: ?>>, vector<8xf16>
        amdgpu.lds_barrier
        vector.store %146, %alloc[%142, %145] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        %147 = arith.addi %6, %139 : index
        %148 = vector.load %alloc[%147, %11] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %149 = arith.addi %11, %c16 : index
        %150 = vector.load %alloc[%147, %149] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %151 = arith.addi %147, %c16 : index
        %152 = vector.load %alloc[%151, %11] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %153 = vector.load %alloc[%151, %149] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %154 = vector.extract %74[0] : f32 from vector<1xf32>
        %155 = vector.splat %154 : vector<4xf32>
        %156 = arith.mulf %155, %cst_1 : vector<4xf32>
        %157 = vector.extract %75[0] : f32 from vector<1xf32>
        %158 = vector.splat %157 : vector<4xf32>
        %159 = arith.mulf %158, %cst_1 : vector<4xf32>
        %160 = amdgpu.mfma %134 * %148 + %156 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %161 = amdgpu.mfma %135 * %150 + %160 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %162 = amdgpu.mfma %136 * %152 + %159 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %163 = amdgpu.mfma %137 * %153 + %162 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %164 = amdgpu.mfma %136 * %148 + %159 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %165 = amdgpu.mfma %137 * %150 + %164 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %166 = amdgpu.mfma %134 * %152 + %156 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %167 = amdgpu.mfma %135 * %153 + %166 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %168 = vector.extract %111[0] : f32 from vector<1xf32>
        %169 = vector.splat %168 : vector<4xf32>
        %170 = arith.divf %161, %169 : vector<4xf32>
        %171 = vector.extract %133[0] : f32 from vector<1xf32>
        %172 = vector.splat %171 : vector<4xf32>
        %173 = arith.divf %163, %172 : vector<4xf32>
        %174 = arith.divf %165, %172 : vector<4xf32>
        %175 = arith.divf %167, %169 : vector<4xf32>
        %176 = vector.extract_strided_slice %170 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %177 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<128x128xf32, strided<[128, 1], offset: ?>>
        %178 = arith.addi %5, %4 : index
        %179 = arith.addi %178, %11 : index
        %180 = arith.addi %6, %138 : index
        %181 = arith.addi %180, %139 : index
        vector.store %176, %177[%179, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %182 = vector.extract_strided_slice %170 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %183 = arith.addi %179, %c1 : index
        vector.store %182, %177[%183, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %184 = vector.extract_strided_slice %170 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %185 = arith.addi %179, %c2 : index
        vector.store %184, %177[%185, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %186 = vector.extract_strided_slice %170 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %187 = arith.addi %179, %c3 : index
        vector.store %186, %177[%187, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %188 = vector.extract_strided_slice %173 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %189 = arith.addi %179, %c16 : index
        %190 = arith.addi %181, %c16 : index
        vector.store %188, %177[%189, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %191 = vector.extract_strided_slice %173 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %192 = arith.addi %179, %c17 : index
        vector.store %191, %177[%192, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %193 = vector.extract_strided_slice %173 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %194 = arith.addi %179, %c18 : index
        vector.store %193, %177[%194, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %195 = vector.extract_strided_slice %173 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %196 = arith.addi %179, %c19 : index
        vector.store %195, %177[%196, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %197 = vector.extract_strided_slice %174 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %197, %177[%189, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %198 = vector.extract_strided_slice %174 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %198, %177[%192, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %199 = vector.extract_strided_slice %174 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %199, %177[%194, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %200 = vector.extract_strided_slice %174 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %200, %177[%196, %181] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %201 = vector.extract_strided_slice %175 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %201, %177[%179, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %202 = vector.extract_strided_slice %175 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %202, %177[%183, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %203 = vector.extract_strided_slice %175 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %203, %177[%185, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %204 = vector.extract_strided_slice %175 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %204, %177[%187, %190] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<128x16xf16>, %arg1: tensor<32x16xf16>, %arg2: tensor<128x32xf16>) -> tensor<128x128xf32> {
    %0 = flow.dispatch @gemm::@gemm(%arg0, %arg1, %arg2) : (tensor<128x16xf16>, tensor<32x16xf16>, tensor<128x32xf16>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}

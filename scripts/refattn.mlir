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
        %c32_i32 = arith.constant 32 : i32
        %c16_i32 = arith.constant 16 : i32
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        %c2_i32 = arith.constant 2 : i32
        %c64_i32 = arith.constant 64 : i32
        %c1_i32 = arith.constant 1 : i32
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
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
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<256x16xf16, strided<[16, 1], offset: ?>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x256xf16, strided<[256, 1], offset: ?>>
        %3 = arith.divsi %thread_id_x, %c64 : index
        %4 = arith.muli %3, %c32 : index
        %5 = arith.muli %workgroup_id_0, %c64 : index
        %6 = arith.remsi %thread_id_x, %c16 : index
        %7 = arith.addi %6, %5 : index
        %8 = arith.addi %7, %4 : index
        %9 = arith.remsi %thread_id_x, %c64 : index
        %10 = arith.divsi %9, %c16 : index
        %11 = arith.muli %10, %c4 : index
        %12 = arith.addi %8, %c16 : index
        %13 = arith.addi %6, %4 : index
        %14 = arith.addi %13, %c16 : index
        %15 = arith.addi %6, %c16 : index
        %16 = arith.muli %workgroup_id_1, %c64 : index
        %17 = arith.muli %thread_id_y, %c32 : index
        %18 = arith.divsi %thread_id_x, %c4 : index
        %19 = arith.addi %18, %17 : index
        %20 = arith.remsi %19, %c64 : index
        %21 = arith.addi %20, %16 : index
        %22 = arith.remsi %thread_id_x, %c4 : index
        %23 = arith.muli %22, %c8 : index
        %24 = arith.addi %6, %17 : index
        %25 = arith.addi %11, %c16 : index
        %26 = arith.addi %24, %c16 : index
        %27:8 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %cst_0, %arg6 = %cst, %arg7 = %cst_1, %arg8 = %cst_1, %arg9 = %cst_0, %arg10 = %cst, %arg11 = %cst_1, %arg12 = %cst_1) -> (vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>) {
          %74 = vector.load %0[%8, %11] : memref<128x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
          %75 = vector.load %0[%12, %11] : memref<128x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
          %alloc_2 = memref.alloc() : memref<64x20xf16, #gpu.address_space<workgroup>>
          vector.store %74, %alloc_2[%13, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %75, %alloc_2[%14, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          %76 = vector.load %alloc_2[%13, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %77 = vector.load %alloc_2[%14, %11] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %78 = arith.muli %arg4, %c32 : index
          %79 = arith.addi %6, %78 : index
          %80 = vector.load %1[%79, %11] : memref<256x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
          %81 = arith.addi %79, %c16 : index
          %82 = vector.load %1[%81, %11] : memref<256x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
          %alloc_3 = memref.alloc() : memref<32x20xf16, #gpu.address_space<workgroup>>
          amdgpu.lds_barrier
          vector.store %80, %alloc_3[%6, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          vector.store %82, %alloc_3[%15, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          amdgpu.lds_barrier
          %83 = vector.load %alloc_3[%6, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %84 = vector.load %alloc_3[%15, %11] : memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %85 = amdgpu.mfma %83 * %76 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %86 = amdgpu.mfma %84 * %77 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %87 = amdgpu.mfma %83 * %77 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %88 = amdgpu.mfma %84 * %76 + %cst_1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %89 = vector.extract_strided_slice %85 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %90 = vector.extract_strided_slice %85 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %91 = arith.maximumf %89, %90 : vector<1xf32>
          %92 = vector.extract_strided_slice %85 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %93 = arith.maximumf %91, %92 : vector<1xf32>
          %94 = vector.extract_strided_slice %85 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %95 = arith.maximumf %93, %94 : vector<1xf32>
          %96 = vector.extract %95[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %96, %c1_i32, %c64_i32 : f32
          %97 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %98 = arith.maximumf %95, %97 : vector<1xf32>
          %99 = vector.extract %98[0] : f32 from vector<1xf32>
          %shuffleResult_4, %valid_5 = gpu.shuffle  xor %99, %c2_i32, %c64_i32 : f32
          %100 = vector.broadcast %shuffleResult_4 : f32 to vector<1xf32>
          %101 = arith.maximumf %98, %100 : vector<1xf32>
          %102 = vector.extract %101[0] : f32 from vector<1xf32>
          %shuffleResult_6, %valid_7 = gpu.shuffle  xor %102, %c4_i32, %c64_i32 : f32
          %103 = vector.broadcast %shuffleResult_6 : f32 to vector<1xf32>
          %104 = arith.maximumf %101, %103 : vector<1xf32>
          %105 = vector.extract %104[0] : f32 from vector<1xf32>
          %shuffleResult_8, %valid_9 = gpu.shuffle  xor %105, %c8_i32, %c64_i32 : f32
          %106 = vector.broadcast %shuffleResult_8 : f32 to vector<1xf32>
          %107 = arith.maximumf %104, %106 : vector<1xf32>
          %108 = vector.extract %107[0] : f32 from vector<1xf32>
          %shuffleResult_10, %valid_11 = gpu.shuffle  xor %108, %c16_i32, %c64_i32 : f32
          %109 = vector.broadcast %shuffleResult_10 : f32 to vector<1xf32>
          %110 = arith.maximumf %107, %109 : vector<1xf32>
          %111 = vector.extract %110[0] : f32 from vector<1xf32>
          %shuffleResult_12, %valid_13 = gpu.shuffle  xor %111, %c32_i32, %c64_i32 : f32
          %112 = vector.broadcast %shuffleResult_12 : f32 to vector<1xf32>
          %113 = arith.maximumf %110, %112 : vector<1xf32>
          %114 = arith.maximumf %arg5, %113 : vector<1xf32>
          %115 = vector.extract_strided_slice %88 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %116 = vector.extract_strided_slice %88 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %117 = arith.maximumf %115, %116 : vector<1xf32>
          %118 = vector.extract_strided_slice %88 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %119 = arith.maximumf %117, %118 : vector<1xf32>
          %120 = vector.extract_strided_slice %88 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %121 = arith.maximumf %119, %120 : vector<1xf32>
          %122 = vector.extract %121[0] : f32 from vector<1xf32>
          %shuffleResult_14, %valid_15 = gpu.shuffle  xor %122, %c1_i32, %c64_i32 : f32
          %123 = vector.broadcast %shuffleResult_14 : f32 to vector<1xf32>
          %124 = arith.maximumf %121, %123 : vector<1xf32>
          %125 = vector.extract %124[0] : f32 from vector<1xf32>
          %shuffleResult_16, %valid_17 = gpu.shuffle  xor %125, %c2_i32, %c64_i32 : f32
          %126 = vector.broadcast %shuffleResult_16 : f32 to vector<1xf32>
          %127 = arith.maximumf %124, %126 : vector<1xf32>
          %128 = vector.extract %127[0] : f32 from vector<1xf32>
          %shuffleResult_18, %valid_19 = gpu.shuffle  xor %128, %c4_i32, %c64_i32 : f32
          %129 = vector.broadcast %shuffleResult_18 : f32 to vector<1xf32>
          %130 = arith.maximumf %127, %129 : vector<1xf32>
          %131 = vector.extract %130[0] : f32 from vector<1xf32>
          %shuffleResult_20, %valid_21 = gpu.shuffle  xor %131, %c8_i32, %c64_i32 : f32
          %132 = vector.broadcast %shuffleResult_20 : f32 to vector<1xf32>
          %133 = arith.maximumf %130, %132 : vector<1xf32>
          %134 = vector.extract %133[0] : f32 from vector<1xf32>
          %shuffleResult_22, %valid_23 = gpu.shuffle  xor %134, %c16_i32, %c64_i32 : f32
          %135 = vector.broadcast %shuffleResult_22 : f32 to vector<1xf32>
          %136 = arith.maximumf %133, %135 : vector<1xf32>
          %137 = vector.extract %136[0] : f32 from vector<1xf32>
          %shuffleResult_24, %valid_25 = gpu.shuffle  xor %137, %c32_i32, %c64_i32 : f32
          %138 = vector.broadcast %shuffleResult_24 : f32 to vector<1xf32>
          %139 = arith.maximumf %136, %138 : vector<1xf32>
          %140 = arith.maximumf %114, %139 : vector<1xf32>
          %141 = vector.extract_strided_slice %87 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %142 = vector.extract_strided_slice %87 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %143 = arith.maximumf %141, %142 : vector<1xf32>
          %144 = vector.extract_strided_slice %87 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %145 = arith.maximumf %143, %144 : vector<1xf32>
          %146 = vector.extract_strided_slice %87 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %147 = arith.maximumf %145, %146 : vector<1xf32>
          %148 = vector.extract %147[0] : f32 from vector<1xf32>
          %shuffleResult_26, %valid_27 = gpu.shuffle  xor %148, %c1_i32, %c64_i32 : f32
          %149 = vector.broadcast %shuffleResult_26 : f32 to vector<1xf32>
          %150 = arith.maximumf %147, %149 : vector<1xf32>
          %151 = vector.extract %150[0] : f32 from vector<1xf32>
          %shuffleResult_28, %valid_29 = gpu.shuffle  xor %151, %c2_i32, %c64_i32 : f32
          %152 = vector.broadcast %shuffleResult_28 : f32 to vector<1xf32>
          %153 = arith.maximumf %150, %152 : vector<1xf32>
          %154 = vector.extract %153[0] : f32 from vector<1xf32>
          %shuffleResult_30, %valid_31 = gpu.shuffle  xor %154, %c4_i32, %c64_i32 : f32
          %155 = vector.broadcast %shuffleResult_30 : f32 to vector<1xf32>
          %156 = arith.maximumf %153, %155 : vector<1xf32>
          %157 = vector.extract %156[0] : f32 from vector<1xf32>
          %shuffleResult_32, %valid_33 = gpu.shuffle  xor %157, %c8_i32, %c64_i32 : f32
          %158 = vector.broadcast %shuffleResult_32 : f32 to vector<1xf32>
          %159 = arith.maximumf %156, %158 : vector<1xf32>
          %160 = vector.extract %159[0] : f32 from vector<1xf32>
          %shuffleResult_34, %valid_35 = gpu.shuffle  xor %160, %c16_i32, %c64_i32 : f32
          %161 = vector.broadcast %shuffleResult_34 : f32 to vector<1xf32>
          %162 = arith.maximumf %159, %161 : vector<1xf32>
          %163 = vector.extract %162[0] : f32 from vector<1xf32>
          %shuffleResult_36, %valid_37 = gpu.shuffle  xor %163, %c32_i32, %c64_i32 : f32
          %164 = vector.broadcast %shuffleResult_36 : f32 to vector<1xf32>
          %165 = arith.maximumf %162, %164 : vector<1xf32>
          %166 = arith.maximumf %arg9, %165 : vector<1xf32>
          %167 = vector.extract_strided_slice %86 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %168 = vector.extract_strided_slice %86 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %169 = arith.maximumf %167, %168 : vector<1xf32>
          %170 = vector.extract_strided_slice %86 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %171 = arith.maximumf %169, %170 : vector<1xf32>
          %172 = vector.extract_strided_slice %86 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %173 = arith.maximumf %171, %172 : vector<1xf32>
          %174 = vector.extract %173[0] : f32 from vector<1xf32>
          %shuffleResult_38, %valid_39 = gpu.shuffle  xor %174, %c1_i32, %c64_i32 : f32
          %175 = vector.broadcast %shuffleResult_38 : f32 to vector<1xf32>
          %176 = arith.maximumf %173, %175 : vector<1xf32>
          %177 = vector.extract %176[0] : f32 from vector<1xf32>
          %shuffleResult_40, %valid_41 = gpu.shuffle  xor %177, %c2_i32, %c64_i32 : f32
          %178 = vector.broadcast %shuffleResult_40 : f32 to vector<1xf32>
          %179 = arith.maximumf %176, %178 : vector<1xf32>
          %180 = vector.extract %179[0] : f32 from vector<1xf32>
          %shuffleResult_42, %valid_43 = gpu.shuffle  xor %180, %c4_i32, %c64_i32 : f32
          %181 = vector.broadcast %shuffleResult_42 : f32 to vector<1xf32>
          %182 = arith.maximumf %179, %181 : vector<1xf32>
          %183 = vector.extract %182[0] : f32 from vector<1xf32>
          %shuffleResult_44, %valid_45 = gpu.shuffle  xor %183, %c8_i32, %c64_i32 : f32
          %184 = vector.broadcast %shuffleResult_44 : f32 to vector<1xf32>
          %185 = arith.maximumf %182, %184 : vector<1xf32>
          %186 = vector.extract %185[0] : f32 from vector<1xf32>
          %shuffleResult_46, %valid_47 = gpu.shuffle  xor %186, %c16_i32, %c64_i32 : f32
          %187 = vector.broadcast %shuffleResult_46 : f32 to vector<1xf32>
          %188 = arith.maximumf %185, %187 : vector<1xf32>
          %189 = vector.extract %188[0] : f32 from vector<1xf32>
          %shuffleResult_48, %valid_49 = gpu.shuffle  xor %189, %c32_i32, %c64_i32 : f32
          %190 = vector.broadcast %shuffleResult_48 : f32 to vector<1xf32>
          %191 = arith.maximumf %188, %190 : vector<1xf32>
          %192 = arith.maximumf %166, %191 : vector<1xf32>
          %193 = arith.subf %arg5, %140 : vector<1xf32>
          %194 = arith.subf %arg9, %192 : vector<1xf32>
          %195 = math.exp2 %193 : vector<1xf32>
          %196 = math.exp2 %194 : vector<1xf32>
          %197 = arith.mulf %arg6, %195 : vector<1xf32>
          %198 = arith.mulf %arg10, %196 : vector<1xf32>
          %199 = vector.extract %140[0] : f32 from vector<1xf32>
          %200 = vector.splat %199 : vector<4xf32>
          %201 = arith.subf %85, %200 : vector<4xf32>
          %202 = arith.subf %88, %200 : vector<4xf32>
          %203 = vector.extract %192[0] : f32 from vector<1xf32>
          %204 = vector.splat %203 : vector<4xf32>
          %205 = arith.subf %87, %204 : vector<4xf32>
          %206 = arith.subf %86, %204 : vector<4xf32>
          %207 = math.exp2 %201 : vector<4xf32>
          %208 = math.exp2 %202 : vector<4xf32>
          %209 = math.exp2 %205 : vector<4xf32>
          %210 = math.exp2 %206 : vector<4xf32>
          %211 = vector.extract_strided_slice %207 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %212 = vector.extract_strided_slice %207 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %213 = arith.addf %211, %212 : vector<1xf32>
          %214 = vector.extract_strided_slice %207 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %215 = arith.addf %213, %214 : vector<1xf32>
          %216 = vector.extract_strided_slice %207 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %217 = arith.addf %215, %216 : vector<1xf32>
          %218 = vector.extract %217[0] : f32 from vector<1xf32>
          %shuffleResult_50, %valid_51 = gpu.shuffle  xor %218, %c1_i32, %c64_i32 : f32
          %219 = vector.broadcast %shuffleResult_50 : f32 to vector<1xf32>
          %220 = arith.addf %217, %219 : vector<1xf32>
          %221 = vector.extract %220[0] : f32 from vector<1xf32>
          %shuffleResult_52, %valid_53 = gpu.shuffle  xor %221, %c2_i32, %c64_i32 : f32
          %222 = vector.broadcast %shuffleResult_52 : f32 to vector<1xf32>
          %223 = arith.addf %220, %222 : vector<1xf32>
          %224 = vector.extract %223[0] : f32 from vector<1xf32>
          %shuffleResult_54, %valid_55 = gpu.shuffle  xor %224, %c4_i32, %c64_i32 : f32
          %225 = vector.broadcast %shuffleResult_54 : f32 to vector<1xf32>
          %226 = arith.addf %223, %225 : vector<1xf32>
          %227 = vector.extract %226[0] : f32 from vector<1xf32>
          %shuffleResult_56, %valid_57 = gpu.shuffle  xor %227, %c8_i32, %c64_i32 : f32
          %228 = vector.broadcast %shuffleResult_56 : f32 to vector<1xf32>
          %229 = arith.addf %226, %228 : vector<1xf32>
          %230 = vector.extract %229[0] : f32 from vector<1xf32>
          %shuffleResult_58, %valid_59 = gpu.shuffle  xor %230, %c16_i32, %c64_i32 : f32
          %231 = vector.broadcast %shuffleResult_58 : f32 to vector<1xf32>
          %232 = arith.addf %229, %231 : vector<1xf32>
          %233 = vector.extract %232[0] : f32 from vector<1xf32>
          %shuffleResult_60, %valid_61 = gpu.shuffle  xor %233, %c32_i32, %c64_i32 : f32
          %234 = vector.broadcast %shuffleResult_60 : f32 to vector<1xf32>
          %235 = arith.addf %232, %234 : vector<1xf32>
          %236 = arith.addf %197, %235 : vector<1xf32>
          %237 = vector.extract_strided_slice %208 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %238 = vector.extract_strided_slice %208 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %239 = arith.addf %237, %238 : vector<1xf32>
          %240 = vector.extract_strided_slice %208 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %241 = arith.addf %239, %240 : vector<1xf32>
          %242 = vector.extract_strided_slice %208 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %243 = arith.addf %241, %242 : vector<1xf32>
          %244 = vector.extract %243[0] : f32 from vector<1xf32>
          %shuffleResult_62, %valid_63 = gpu.shuffle  xor %244, %c1_i32, %c64_i32 : f32
          %245 = vector.broadcast %shuffleResult_62 : f32 to vector<1xf32>
          %246 = arith.addf %243, %245 : vector<1xf32>
          %247 = vector.extract %246[0] : f32 from vector<1xf32>
          %shuffleResult_64, %valid_65 = gpu.shuffle  xor %247, %c2_i32, %c64_i32 : f32
          %248 = vector.broadcast %shuffleResult_64 : f32 to vector<1xf32>
          %249 = arith.addf %246, %248 : vector<1xf32>
          %250 = vector.extract %249[0] : f32 from vector<1xf32>
          %shuffleResult_66, %valid_67 = gpu.shuffle  xor %250, %c4_i32, %c64_i32 : f32
          %251 = vector.broadcast %shuffleResult_66 : f32 to vector<1xf32>
          %252 = arith.addf %249, %251 : vector<1xf32>
          %253 = vector.extract %252[0] : f32 from vector<1xf32>
          %shuffleResult_68, %valid_69 = gpu.shuffle  xor %253, %c8_i32, %c64_i32 : f32
          %254 = vector.broadcast %shuffleResult_68 : f32 to vector<1xf32>
          %255 = arith.addf %252, %254 : vector<1xf32>
          %256 = vector.extract %255[0] : f32 from vector<1xf32>
          %shuffleResult_70, %valid_71 = gpu.shuffle  xor %256, %c16_i32, %c64_i32 : f32
          %257 = vector.broadcast %shuffleResult_70 : f32 to vector<1xf32>
          %258 = arith.addf %255, %257 : vector<1xf32>
          %259 = vector.extract %258[0] : f32 from vector<1xf32>
          %shuffleResult_72, %valid_73 = gpu.shuffle  xor %259, %c32_i32, %c64_i32 : f32
          %260 = vector.broadcast %shuffleResult_72 : f32 to vector<1xf32>
          %261 = arith.addf %258, %260 : vector<1xf32>
          %262 = arith.addf %236, %261 : vector<1xf32>
          %263 = vector.extract_strided_slice %209 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %264 = vector.extract_strided_slice %209 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %265 = arith.addf %263, %264 : vector<1xf32>
          %266 = vector.extract_strided_slice %209 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %267 = arith.addf %265, %266 : vector<1xf32>
          %268 = vector.extract_strided_slice %209 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %269 = arith.addf %267, %268 : vector<1xf32>
          %270 = vector.extract %269[0] : f32 from vector<1xf32>
          %shuffleResult_74, %valid_75 = gpu.shuffle  xor %270, %c1_i32, %c64_i32 : f32
          %271 = vector.broadcast %shuffleResult_74 : f32 to vector<1xf32>
          %272 = arith.addf %269, %271 : vector<1xf32>
          %273 = vector.extract %272[0] : f32 from vector<1xf32>
          %shuffleResult_76, %valid_77 = gpu.shuffle  xor %273, %c2_i32, %c64_i32 : f32
          %274 = vector.broadcast %shuffleResult_76 : f32 to vector<1xf32>
          %275 = arith.addf %272, %274 : vector<1xf32>
          %276 = vector.extract %275[0] : f32 from vector<1xf32>
          %shuffleResult_78, %valid_79 = gpu.shuffle  xor %276, %c4_i32, %c64_i32 : f32
          %277 = vector.broadcast %shuffleResult_78 : f32 to vector<1xf32>
          %278 = arith.addf %275, %277 : vector<1xf32>
          %279 = vector.extract %278[0] : f32 from vector<1xf32>
          %shuffleResult_80, %valid_81 = gpu.shuffle  xor %279, %c8_i32, %c64_i32 : f32
          %280 = vector.broadcast %shuffleResult_80 : f32 to vector<1xf32>
          %281 = arith.addf %278, %280 : vector<1xf32>
          %282 = vector.extract %281[0] : f32 from vector<1xf32>
          %shuffleResult_82, %valid_83 = gpu.shuffle  xor %282, %c16_i32, %c64_i32 : f32
          %283 = vector.broadcast %shuffleResult_82 : f32 to vector<1xf32>
          %284 = arith.addf %281, %283 : vector<1xf32>
          %285 = vector.extract %284[0] : f32 from vector<1xf32>
          %shuffleResult_84, %valid_85 = gpu.shuffle  xor %285, %c32_i32, %c64_i32 : f32
          %286 = vector.broadcast %shuffleResult_84 : f32 to vector<1xf32>
          %287 = arith.addf %284, %286 : vector<1xf32>
          %288 = arith.addf %198, %287 : vector<1xf32>
          %289 = vector.extract_strided_slice %210 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %290 = vector.extract_strided_slice %210 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %291 = arith.addf %289, %290 : vector<1xf32>
          %292 = vector.extract_strided_slice %210 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %293 = arith.addf %291, %292 : vector<1xf32>
          %294 = vector.extract_strided_slice %210 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
          %295 = arith.addf %293, %294 : vector<1xf32>
          %296 = vector.extract %295[0] : f32 from vector<1xf32>
          %shuffleResult_86, %valid_87 = gpu.shuffle  xor %296, %c1_i32, %c64_i32 : f32
          %297 = vector.broadcast %shuffleResult_86 : f32 to vector<1xf32>
          %298 = arith.addf %295, %297 : vector<1xf32>
          %299 = vector.extract %298[0] : f32 from vector<1xf32>
          %shuffleResult_88, %valid_89 = gpu.shuffle  xor %299, %c2_i32, %c64_i32 : f32
          %300 = vector.broadcast %shuffleResult_88 : f32 to vector<1xf32>
          %301 = arith.addf %298, %300 : vector<1xf32>
          %302 = vector.extract %301[0] : f32 from vector<1xf32>
          %shuffleResult_90, %valid_91 = gpu.shuffle  xor %302, %c4_i32, %c64_i32 : f32
          %303 = vector.broadcast %shuffleResult_90 : f32 to vector<1xf32>
          %304 = arith.addf %301, %303 : vector<1xf32>
          %305 = vector.extract %304[0] : f32 from vector<1xf32>
          %shuffleResult_92, %valid_93 = gpu.shuffle  xor %305, %c8_i32, %c64_i32 : f32
          %306 = vector.broadcast %shuffleResult_92 : f32 to vector<1xf32>
          %307 = arith.addf %304, %306 : vector<1xf32>
          %308 = vector.extract %307[0] : f32 from vector<1xf32>
          %shuffleResult_94, %valid_95 = gpu.shuffle  xor %308, %c16_i32, %c64_i32 : f32
          %309 = vector.broadcast %shuffleResult_94 : f32 to vector<1xf32>
          %310 = arith.addf %307, %309 : vector<1xf32>
          %311 = vector.extract %310[0] : f32 from vector<1xf32>
          %shuffleResult_96, %valid_97 = gpu.shuffle  xor %311, %c32_i32, %c64_i32 : f32
          %312 = vector.broadcast %shuffleResult_96 : f32 to vector<1xf32>
          %313 = arith.addf %310, %312 : vector<1xf32>
          %314 = arith.addf %288, %313 : vector<1xf32>
          %315 = arith.truncf %207 : vector<4xf32> to vector<4xf16>
          %316 = arith.truncf %208 : vector<4xf32> to vector<4xf16>
          %317 = arith.truncf %209 : vector<4xf32> to vector<4xf16>
          %318 = arith.truncf %210 : vector<4xf32> to vector<4xf16>
          %319 = arith.addi %78, %23 : index
          %320 = vector.load %2[%21, %319] : memref<128x256xf16, strided<[256, 1], offset: ?>>, vector<8xf16>
          amdgpu.lds_barrier
          vector.store %320, %alloc[%20, %23] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          amdgpu.lds_barrier
          %321 = vector.load %alloc[%24, %11] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %322 = vector.load %alloc[%24, %25] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %323 = vector.load %alloc[%26, %11] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %324 = vector.load %alloc[%26, %25] : memref<64x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %325 = vector.extract %195[0] : f32 from vector<1xf32>
          %326 = vector.splat %325 : vector<4xf32>
          %327 = arith.mulf %arg7, %326 : vector<4xf32>
          %328 = vector.extract %196[0] : f32 from vector<1xf32>
          %329 = vector.splat %328 : vector<4xf32>
          %330 = arith.mulf %arg12, %329 : vector<4xf32>
          %331 = arith.mulf %arg11, %329 : vector<4xf32>
          %332 = arith.mulf %arg8, %326 : vector<4xf32>
          %333 = amdgpu.mfma %315 * %321 + %327 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %334 = amdgpu.mfma %316 * %322 + %333 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %335 = amdgpu.mfma %317 * %323 + %330 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %336 = amdgpu.mfma %318 * %324 + %335 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %337 = amdgpu.mfma %317 * %321 + %331 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %338 = amdgpu.mfma %318 * %322 + %337 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %339 = amdgpu.mfma %315 * %323 + %332 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %340 = amdgpu.mfma %316 * %324 + %339 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          scf.yield %140, %262, %334, %340, %192, %314, %338, %336 : vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>
        }
        %28 = vector.extract %27#1[0] : f32 from vector<1xf32>
        %29 = vector.splat %28 : vector<4xf32>
        %30 = arith.divf %27#2, %29 : vector<4xf32>
        %31 = vector.extract %27#5[0] : f32 from vector<1xf32>
        %32 = vector.splat %31 : vector<4xf32>
        %33 = arith.divf %27#7, %32 : vector<4xf32>
        %34 = arith.divf %27#6, %32 : vector<4xf32>
        %35 = arith.divf %27#3, %29 : vector<4xf32>
        %36 = vector.extract_strided_slice %30 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %37 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<128x128xf32, strided<[128, 1], offset: ?>>
        %38 = arith.remsi %thread_id_x, %c64 : index
        %39 = arith.divsi %38, %c16 : index
        %40 = arith.muli %39, %c4 : index
        %41 = arith.divsi %thread_id_x, %c64 : index
        %42 = arith.muli %41, %c32 : index
        %43 = arith.muli %workgroup_id_0, %c64 : index
        %44 = arith.addi %43, %42 : index
        %45 = arith.addi %44, %40 : index
        %46 = arith.muli %thread_id_y, %c32 : index
        %47 = arith.muli %workgroup_id_1, %c64 : index
        %48 = arith.remsi %thread_id_x, %c16 : index
        %49 = arith.addi %48, %47 : index
        %50 = arith.addi %49, %46 : index
        vector.store %36, %37[%45, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %51 = vector.extract_strided_slice %30 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %52 = arith.addi %45, %c1 : index
        vector.store %51, %37[%52, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %53 = vector.extract_strided_slice %30 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %54 = arith.addi %45, %c2 : index
        vector.store %53, %37[%54, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %55 = vector.extract_strided_slice %30 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %56 = arith.addi %45, %c3 : index
        vector.store %55, %37[%56, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %57 = vector.extract_strided_slice %33 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %58 = arith.addi %45, %c16 : index
        %59 = arith.addi %50, %c16 : index
        vector.store %57, %37[%58, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %60 = vector.extract_strided_slice %33 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %61 = arith.addi %45, %c17 : index
        vector.store %60, %37[%61, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %62 = vector.extract_strided_slice %33 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %63 = arith.addi %45, %c18 : index
        vector.store %62, %37[%63, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %64 = vector.extract_strided_slice %33 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %65 = arith.addi %45, %c19 : index
        vector.store %64, %37[%65, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %66 = vector.extract_strided_slice %34 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %66, %37[%58, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %67 = vector.extract_strided_slice %34 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %67, %37[%61, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %68 = vector.extract_strided_slice %34 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %68, %37[%63, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %69 = vector.extract_strided_slice %34 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %69, %37[%65, %50] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %70 = vector.extract_strided_slice %35 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %70, %37[%45, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %71 = vector.extract_strided_slice %35 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %71, %37[%52, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %72 = vector.extract_strided_slice %35 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %72, %37[%54, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        %73 = vector.extract_strided_slice %35 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %73, %37[%56, %59] : memref<128x128xf32, strided<[128, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<128x16xf16>, %arg1: tensor<256x16xf16>, %arg2: tensor<128x256xf16>) -> tensor<128x128xf32> {
    %0 = flow.dispatch @gemm::@gemm(%arg0, %arg1, %arg2) : (tensor<128x16xf16>, tensor<256x16xf16>, tensor<128x256xf16>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}

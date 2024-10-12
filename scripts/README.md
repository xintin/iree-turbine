### Testing of Attention on Chained Matmul

Please activate venv and run:
```sh
python reduce_kernel.py
Expected output:

"Careful! You passed a custom op where an fx.Node was required.
(~50x)
SUCCESS"
```

To try attention, comment out base_ref_gemm and the chained torch matmuls, and then
uncomment torch attention and gemm()

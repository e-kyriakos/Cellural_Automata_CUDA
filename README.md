# Cellural Automata CUDA

Here is the implementation of the third Parallel & Distributed systems homework of computing a Ising model on NVIDIA GPGPU using CUDA.

To compile each of the CUDA scripts of version X do:

```bash
#!/bin/bash

  nvcc -O3 cuda_vX.cu -o cuda_vX
  ./cuda_vX n k
```
If want an animation (that shouldn't be tested when time stressing) type:

```bash
#!/bin/bash
./cuda_vX n k a
```
For copetative time test please comment print commands.

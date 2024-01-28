# Cellural Automata CUDA

Here is the implementation of the third Parallel & Distributed systems homework of computing a Ising model on NVIDIA GPGPU using CUDA.

To compile each of the CUDA scripts of version X do:

```bash
#!/bin/bash

  nvcc -O3 cuda_vX.cu -o cuda_vX
  ./cuda_vX n k

  For example:
  ./cuda_vX 20 20
```
If want an animation (that shouldn't be tested when time stressing) type:

```bash
#!/bin/bash
./cuda_vX n k a

as for instance:
./cuda_vX 20 20 a
```
For copetative time scores please comment print commands.

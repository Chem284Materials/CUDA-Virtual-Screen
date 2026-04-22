# 💻 CHEM 284 - Parallelizing pose screening with PyCUDA

## 🧪 Goal

The goal of this lab is to:

1. Learn about **GPU programming**.
2. Learn how to use **PyCUDA**. 
3. Practice using **exercises including molecular docking with a grid**.
4. Profile serial and parallel versions of the code.

---
## 🗂️ Provided

- A python notebook will all the necessary cells.
- A `utils.py` file with the functions related to the serial implementation.
- A `data` directory with the relevant files including the grid and initial ligand state.

---
## 💻 Setup
1) Go to https://colab.research.google.com/ and upload the notebook in the repository.
2) Select the T4 GPU runtime.
3) Upload the `utils.py` file into the Files section.
4) Create a new directory in the Files section called `data` and upload the `grid.pts` and `ligand.xyz` files into it.
5) You can start running the cells!

Note: If you have an NVIDIA GPU, you can run the notebook on your local PC.

## GPU Architecture Review
```
Grid (All blocks for kernel)
┌─────────────────────────────┐
│ Block 0                     │
│ ┌───────────────────────┐   │
│ │ Threads 0 – 127       │   │
│ │ Warp 0: 0–31          │ → SM cores execute in lock-step
│ │ Warp 1: 32–63         │
│ │ Warp 2: 64–95         │
│ │ Warp 3: 96–127        │
│ └───────────────────────┘
│ Block 1                     │
│ ┌───────────────────────┐   │
│ │ Threads 0 – 127       │   │
│ │ Warp 0: 0–31          │ → Scheduled on same or different SM
│ │ Warp 1: 32–63         │
│ │ Warp 2: 64–95         │
│ │ Warp 3: 96–127        │
│ └───────────────────────┘
│   ...                       │
└─────────────────────────────┘

Streaming Multiprocessor (SM)
┌─────────────────────────────┐
│ CUDA cores: 64 (example)    │
│ Registers & shared memory   │
│ Warp 0: threads 0–31        │ Executed together on cores
│ Warp 1: threads 32–63       │ Executed together on cores
│ Warp 2: threads 64–95       │ Scheduled when cores are free
│ Warp 3: threads 96–127      │
└─────────────────────────────┘
```

## ✅ Tasks
### Parallelize Molecular Docking using PyCUDA:
Now as we've seen before with MPI and OpenMP we will be parallelizing a "virtual screen". The process is the same as before: create a bunch of random poses from the base ligand using a pose index and use [trilinear interpolation](https://en.wikipedia.org/wiki/Trilinear_interpolation) to determine the value of each atom in the grid based on the sampled points to calculate the total energy for the pose.

1) To parallelize the code onto the GPU we need to create a kernel function. You have been provided with some boiler plate code our kernel function `score_poses` as well as device functions `grid_value` and `trilinear_interp`, remember the device functions can only be called from the GPU while the kernel function can be called from the CPU and each GPU thread executes the kernel function.
2) The first step to parallelize is to transfer the data from the CPU to the GPU. This can be done in 2 ways, using `gpuarray.to_gpu` which will manage the memory for you, OR allocating memory onto the GPU for our data and then copying it from the CPU memory. This will need to be done inside of `score_poses_gpu` as this is the main CPU-GPU driver. You can use `cuda.mem_alloc` and `cuda.memcpy_htod`, respectively if you want to manage it yourself.

```python
# PyCUDA memory management
grid_gpu = gpuarray.to_gpu(grid)

# Manual memory management
grid_gpu = cuda.mem_alloc(grid.nbytes)
cuda.memcpy_htod(grid_gpu, grid)
grid_gpu.free()
```

3) Now that we have transferred our data from the CPU to the GPU we can determine the grid size! Remember GPU hierarchy, grid -> block -> threads, each grid holds all the blocks for a kernel launch, a block holds a group of threads (a block of 128 means, each block will have 128 threads scheduled by (128 threads / 32 threads per warp) = 4 warps) and each thread is the smallest execution unit. For this example each thread will be responsible for a single pose, so we need a way for all `100,000 poses` to have a thread to work on and we know we have 128 threads per block. It is important that we do not under provision the grid size because if `poses > threads`, then some poses will never get scored. It is totally fine to over shoot this because in our kernel we return early if we have a pose index > num_poses.

```
Grid
┌───────────────┐
│  Block 0      │
│ ┌───────────┐ │
│ │ Thread 0  │ │ → pose 0
│ │ Thread 1  │ │ → pose 1
│ │   ...     │ │
│ │ Thread127 │ │ → pose 127
│ └───────────┘ │
│  Block 1      │
│ ┌───────────┐ │
│ │ Thread 0  │ │ → pose 128
│ │ Thread 1  │ │ → pose 129
│ │   ...     │ │
│ │ Thread127 │ │ → pose 255
│ └───────────┘ │
│     ...       │
│  Block N-1    │
│ ┌───────────┐ │
│ │ Thread 0  │ │ → pose ...
│ │   ...     │ │
│ │ Thread127 │ │ → pose nposes-1
│ └───────────┘ │
└───────────────┘
```

4) Once you have determined the size of the grid we can look closer at `score_poses`. This kernel will be executed by each thread on the GPU, so each thread will be responsible for 1 pose. So you will need to use the blockIdx, blockDim and threadIdx to determine which pose id, the thread is responsible for. This is important because if a thread's `pose` is greater than the number of poses we want, that thread should early exit and return.
5) After determining what pose the thread is working on, you have some code provided to you that probably looks familiar. This is the code for calculating the rotation matrix (R00 - R22) and translation vector ([tx, ty, tz]) for generating our poses. In this case we will transform each atom on the fly before we calculate its energy. This is much faster because previously we would be generating all the poses once and then distributing them. There is nothing else you need to do here.
6) Now we can set our initial energy and then start looping over all the `natoms` for this ligand.
7) For each atom in the ligand you will need to do the following
    - Convert the x, y, z coordinates for the atom from world coordinates to the ligand coordinate system by substracting the centroid of the ligand.
    - Rotate the atom using the rotation matrix created earlier (R00 - R22).
    - Move the atom back into the world coordinates and translate the pose by adding back in the centroid and the translation vector.
    - The ligand atoms coordinates are stored as a FLAT array in `const double* ligand` so you must index them appropriately!

```cpp
// Applying the logic to the x coordinate for the atom
double px = ligand[x_index] - centroid[0]
double py = ligand[y_index] - centroid[0]
double pz = ligand[z_index] - centroid[0]

double rx = R00*px + R01*py + R02*pz + centroid[0] + tx;
```

Note: `ligand` is a flattened array, so indexing is not as straight forward as it may seem.

8) Once you have completed this, you have translated the atom to where it would be for the specific pose and the thread can now calculate its energy and add it to the running total using the `trilinear_interp` device function.
9) After all the atoms energies have been calculated you can store the energy into the `energies` array which functions as our return. This avoids race conditions as each thread should have its own position in the `energies` array to place its energy.
10) Finally, back in `score_poses_gpu` we need to get all the energies from `energies_gpu` that lives on the GPU back onto the CPU so we can find the lowest energy and the pose for the lowest energy. This has to happen after synchronize step so we know the kernel has finished on all threads.

Do both methods and compare the time taken at 100,000 poses:

| Method           | Time taken |
|------------------|------------|
| CPU serial       |            |
| GPU parallel     |            |

Make sure your parallel solution has the same answer as the serial!

### Extra time
Try to use different values for the block size, does anything change?

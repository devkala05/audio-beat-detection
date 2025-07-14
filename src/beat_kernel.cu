#include <cuda_runtime.h>
#include "beat_kernel.cuh"

__global__ void compute_energy(float* audioData, float* energyOut, int numSamples, int frameSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numSamples / frameSize) {
        float sum = 0.0f;
        for (int j = 0; j < frameSize; ++j) {
            int idx = i * frameSize + j;
            if (idx < numSamples) {
                float sample = audioData[idx];
                sum += sample * sample;
            }
        }
        energyOut[i] = sum;
    }
}

// Only launch kernel — don't allocate or free inside this
void compute_energy_kernel(float* d_audio, float* d_energy, int numSamples, int frameSize) {
    int frames = numSamples / frameSize;
    int blockSize = 256;
    int numBlocks = (frames + blockSize - 1) / blockSize;

    compute_energy<<<numBlocks, blockSize>>>(d_audio, d_energy, numSamples, frameSize);
}

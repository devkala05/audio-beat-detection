#ifndef BEAT_KERNEL_CUH
#define BEAT_KERNEL_CUH

void compute_energy_kernel(float* audioData, float* energyOut, int numSamples, int frameSize);

#endif

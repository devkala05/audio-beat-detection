#include <iostream>
#include <fstream>
#include <vector>
#include <sndfile.h>
#include <cuda_runtime.h>
#include "beat_kernel.cuh"

#define FRAME_SIZE 1024
#define THRESHOLD 1.0f

void loadWavFile(const std::string &filepath, std::vector<float> &samples, int &samplerate) {
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(filepath.c_str(), SFM_READ, &sfinfo);

    if (!infile) {
        std::cerr << "❌ Failed to open file: " << filepath << std::endl;
        exit(1);
    }

    std::cout << "✅ Loaded: " << filepath << std::endl;
    std::cout << "   Sample Rate: " << sfinfo.samplerate << "\n";
    std::cout << "   Channels:    " << sfinfo.channels << "\n";
    std::cout << "   Frames:      " << sfinfo.frames << "\n";

    samplerate = sfinfo.samplerate;
    samples.resize(sfinfo.frames * sfinfo.channels);

    sf_readf_float(infile, samples.data(), sfinfo.frames);
    sf_close(infile);

    std::cout << "   Total samples read: " << samples.size() << std::endl;

    // Optional: show first 10 samples
    std::cout << "First 10 samples: ";
    for (int i = 0; i < std::min(10, (int)samples.size()); ++i) {
        std::cout << samples[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./beat_detect <audio_file.wav>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::vector<float> samples;
    int samplerate;
    loadWavFile(filepath, samples, samplerate);

    int total_frames = samples.size() / FRAME_SIZE;
    std::vector<float> energy(total_frames);

    float *d_samples, *d_energy;
    cudaMalloc(&d_samples, samples.size() * sizeof(float));
    cudaMemcpy(d_samples, samples.data(), samples.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_energy, total_frames * sizeof(float));
    compute_energy_kernel(d_samples, d_energy, samples.size(), FRAME_SIZE);

    cudaMemcpy(energy.data(), d_energy, total_frames * sizeof(float), cudaMemcpyDeviceToHost);

    // DEBUG: print energy values
    std::cout << "First 10 energy values:\n";
    for (int i = 0; i < std::min(10, total_frames); ++i) {
        std::cout << "Frame " << i << ": " << energy[i] << "\n";
    }

    // Compute mean
    float mean_energy = 0;
    for (auto &e : energy) mean_energy += e;
    mean_energy /= total_frames;

    std::cout << "Mean energy: " << mean_energy << std::endl;

    // Generate output path
    std::string outpath = "data/output/" + filepath.substr(filepath.find_last_of("/\\") + 1);
    outpath = outpath.substr(0, outpath.find_last_of(".")) + "_beats.txt";
    std::ofstream outfile(outpath);

    int beats = 0;
    for (int i = 0; i < total_frames; ++i) {
        if (energy[i] > mean_energy * THRESHOLD) {
            float time = i * (float)FRAME_SIZE / samplerate;
            outfile << time << std::endl;
            beats++;
        }
    }

    outfile.close();
    cudaFree(d_samples);
    cudaFree(d_energy);

    if (beats == 0) {
        std::cout << "⚠️  No beats detected in file: " << filepath << std::endl;
    } else {
        std::cout << "✅ " << beats << " beats detected for: " << filepath << "\nOutput: " << outpath << std::endl;
    }

    return 0;
}

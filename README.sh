## 🎧 CUDA Audio Beat Detection

This project implements a **GPU-accelerated audio beat detection system** using CUDA. It analyzes `.wav` audio files by computing short-time energy across frames and identifying beat timings where energy significantly exceeds the average.

---

### 🚀 Features

* 📂 Processes 100+ `.wav` files in parallel (via script)
* ⚡ Uses CUDA to accelerate energy calculations
* 🔊 Outputs detected beat timestamps per file
* 📈 Logs run data and results for validation

---

### 📦 Requirements

* CUDA Toolkit (>= 10.0)
* `libsndfile` installed (for reading `.wav` files)

To install `libsndfile`:

```bash
sudo apt-get install libsndfile1-dev
```

---

### 🛠️ How to Build & Run

1. Place your `.wav` audio files inside `data/input/`
2. Build and run the detector:

```bash
chmod +x run.sh
./run.sh
```

3. Check outputs:

   * Beat time files → `data/output/`
   * Run summary → `logs/run_logs.txt`

---

### ⚙️ How It Works

1. **Load Audio**: `.wav` file is read using `libsndfile`.
2. **Split into Frames**: Each audio is split into short frames (1024 samples).
3. **Compute Energy**: A CUDA kernel calculates the sum of squares for each frame.
4. **Detect Beats**: Beats are frames where energy > `mean_energy × THRESHOLD`.
5. **Export**: Beat timestamps (in seconds) are saved per file.

---

### 📊 Example Output (`file.wav_beats.txt`)

```
0.42
1.11
2.02
...
```

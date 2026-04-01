# Technical Research: Anatomy of the Dreamtonics Neural Network Inference (DNNI) Engine

## 1. Executive Summary
DNNI (Dreamtonics Neural Network Inference) is a proprietary, high-performance neural execution environment optimized for real-time vocal synthesis on x86/ARM CPUs. Unlike standard frameworks (ONNX, LibTorch), DNNI utilizes **Structured Sparsity** and **JIT-compilation** to minimize latency. This research outlines the binary structure of `.dnni` containers and the logic of the underlying inference engine.

---

## 2. Global File Anatomy
The `.dnni` file is a hierarchical, chunk-based container. It is designed for memory-mapped I/O and strict 16-byte alignment to facilitate SIMD (AVX/AVX-512) processing.

### 2.1 File Header (8 bytes)
All DNNI files begin with a fixed signature:
`ff 00 ca 7f 01 00 00 00`
*   `ff 00 ca 7f`: Magic Identifier (Dreamtonics Signature).
*   `01 00 00 00`: Container Format Version.

---

## 3. Chunk System Taxonomy
The container is divided into "Chunks" signaled by specific 4-byte markers. Every chunk consists of a marker, a name (ASCII), and a payload.

### 3.1 Markers
*   **Metadata Chunk (`ff 41 ca 7f`):** Contains strings, vocabulary sets, and layer definitions.
*   **Data Chunk (`ff 40 ca 7f`):** Contains raw numerical arrays (weights, biases, masks).

### 3.2 Key Chunk Types

| Chunk Name | Category | Description |
| :--- | :--- | :--- |
| `_psv1` | Linguistic | **Phoneme Set Vocabulary.** Maps strings (e.g., "a", "k", "pau") to neural indices. Categorizes phones into `vowel`, `stop`, `fricative`, etc. |
| `modm` | Metadata | **Model Metadata.** Stores normalization constants (`mean`, `std`). Crucial for input scaling. |
| `modl` | Layout | **Model Layout.** Defines the architecture (e.g., number of hidden units, activation functions like ReLU/Tanh). |
| `prim1` | Primitives | **Bias/Scale Vectors.** Small arrays (usually `float32`) representing layer biases or normalization offsets. |
| `prim4` | Primitives | **Weight Matrices.** Large arrays of weights. These are the "muscles" of the model. |
| `cmpg` / `cmpu` | Logic | **Compute Graph/Unit.** Instructions for the JIT compiler on how to link layers. |
| `_vwrv2` | Engine | **Vocal Waveform Renderer.** Specific to `payload.dnni`, indicates the timbre-generation core. |

---

## 4. Inference Engine Mechanics
The DNNI engine does not perform standard dense matrix multiplication. Its speed comes from two proprietary technologies:

### 4.1 SpMVM (Sparse Matrix-Vector Multiplication)
The neural networks are trained with "Sparsity Induction." In blocks like `prim4`, roughly 75%–90% of the weights are zero. 
*   **Mechanism:** Instead of multiplying by zero, the engine uses a "Sparsity Mask" to skip operations at the CPU instruction level.
*   **Impact:** This allows a model with 260+ layers (like `audio2score`) to run in real-time on a standard consumer laptop.

### 4.2 JIT Compilation & SIMD
Upon loading a `.dnni` file, the engine generates machine code tailored to the user's CPU. It heavily utilizes:
*   **AVX2 / AVX-512:** Processing 8 to 16 `float32` values in a single clock cycle.
*   **Memory Mapping:** The file is mapped directly into RAM, allowing the CPU to treat the `.dnni` data as a native memory array.

---

## 5. Model Heuristics & Sizing (mostly inccorect)
Based on heuristic "float-scanning" and chunk boundary analysis, we can categorize the models as follows:


| Model File | Estimated Depth | Neural Strategy |
| :--- | :--- | :--- |
| **f0.dnni** | 56 Layers | **Multilingual RNN/LSTM.** Predicts pitch based on phoneme context. |
| **payload.dnni** | Variable | **Sparse CNN/Vocoder.** Generates the unique "voice" or timbre of the singer. |
| **audio2score** | 261 Layers | **Deep ResNet/Transformer.** Transcribes raw audio into MIDI and phonemes. |

*Note: Sizing is based on raw data density. Active parameters may be lower due to sparsity.*

---

This is just a theory! WIP.

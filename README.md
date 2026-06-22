# MC-CDMA System Simulation using OFDM in MATLAB

## 📌 Project Overview
This project implements a complete Multi-Carrier Code Division Multiple Access (MC-CDMA) communication system in MATLAB. The system integrates OFDM modulation, Walsh code spreading, Forward Error Correction (FEC), AGC training sequences, Barker synchronization, MMSE channel estimation, and performance evaluation through BER, SER, and FER measurements. The objective of this work is to study the performance of spread-spectrum multiuser communication systems under fading and noisy wireless channels.

---

## ✨ Key Features
* **Multi-User MC-CDMA Communication**
* **OFDM Modulation and Demodulation**
* **Walsh-Hadamard Spreading Codes**
* **Convolutional Encoding (FEC)**
* **Viterbi Decoding**
* **AGC Training Sequence**
* **Barker Synchronization**
* **MMSE Channel Estimation**
* **Rayleigh Fading Channel**
* **AWGN Channel**
* **BER, SER, and FER Analysis**
* **PSD and Spectrum Analysis**

---

## 📊 System Parameters

| Parameter | Value |
| :--- | :--- |
| **Number of Users** | 4 |
| **FFT Size** | 256 |
| **Cyclic Prefix** | 16 |
| **Guard Subcarriers** | 8 Left + 8 Right |
| **Used Subcarriers** | 240 |
| **Spreading Factor** | 4 |
| **OFDM Symbols per Frame** | 4 |
| **Modulation** | QPSK |
| **Channel Coding** | Convolutional Code |
| **Constraint Length** | 3 |
| **Generator Polynomials** | `[7 5]` |

---

## 🏗️ System Architecture

### Frame Structure
> **AGC Field** $\rightarrow$ **Barker Sync** $\rightarrow$ **Training Sequence** $\rightarrow$ **OFDM Payload** $\rightarrow$ **Frame Guard**

### Transmitter Flow
1. Information Bits
2. FEC Encoder (Convolutional)
3. QPSK Mapper
4. Walsh Spreader
5. Multiuser Combiner
6. OFDM Modulator (IFFT)
7. Cyclic Prefix Addition
8. Frame Generator

### Receiver Flow
1. Received Signal
2. AGC Processing
3. Barker Correlation Synchronization
4. Channel Estimator (MMSE)
5. Equalizer
6. OFDM Demodulator (FFT)
7. Despreader
8. QPSK Demapper
9. Viterbi Decoder
10. Recovered Data

---

## 📡 Channel Model & Analysis

### The Communication Channel
* **Rayleigh Flat Fading**
* **Additive White Gaussian Noise (AWGN)**
* **MMSE-Based Channel Estimation**

### Performance Metrics
* **BER (Bit Error Rate):** Measures bit recovery accuracy.
* **SER (Symbol Error Rate):** Measures symbol detection performance.
* **FER (Frame Error Rate):** Measures complete frame recovery success.

### Spectrum Analysis
The project compares the **Unspread OFDM Signal** against the **Spread MC-CDMA Signal**. Analysis includes:
* Frequency Spectrum
* Power Spectral Density (PSD)
* Bandwidth Expansion Due to Spreading

---

## 📁 MATLAB Files
* `final_mc_cdma_tx.m` — Main Transmitter Implementation
* `generateFrames.m` — Frame Generation Module
* `final_mc_cdma_rx.m` — Receiver Processing and Performance Evaluation

---

## 🛠️ Skills Demonstrated
* Wireless Communication & Digital Signal Processing
* OFDM & MC-CDMA Systems
* Spread Spectrum Communication
* Error Control Coding & Synchronization Techniques
* Channel Estimation
* MATLAB Programming
* Research-Oriented System Design

---

## 👨‍💻 Author
**Rajesh Chauhan** M.Tech – Wireless Communication and Networks  
Indian Institute of Technology Kharagpur  

**Research Areas:**
* Wireless Communication
* Digital Signal Processing
* Embedded Systems
* Electronic System Design
* AI Applications in Communication Systems

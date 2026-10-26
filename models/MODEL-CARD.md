---
license: openmdw-1.1
track_downloads: true
pipeline_tag: audio-to-audio
library_name: mamba-ssm
tags:
- speech-enhancement
- universal speech enhancement
- multiple input sampling rates
- language-agnostic
---
# **<span style="color:#76b900;">🤫 NVIDIA NemotronLabs RE-USE: Multilingual Universal Speech Enhancement</span>**
  
# Model Overview

## Description:  
NVIDIA NemotronLabs RE-USE is a 9.6-million-parameter **multilingual universal speech enhancement (USE)** model designed to restore speech quality across a wide range of real-world degradations while preserving speech fidelity. It has the following desirable properties:  
- Robustness to **diverse degradations**, including additive noise, reverberation, clipping, bandwidth limitation, codec artifacts, packet loss and low-quality mics.
- Support for **multiple input sampling rates**, including 8, 16, 22.05, 24, 32, 44.1, and 48 kHz.
- Strong **language-agnostic** capability, enabling effective performance across different languages.
- If better laughter preservation is desired, please use: [NVIDIA-NemotronLabs-RE-USE-laughter](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE-laughter).

(Note: For overlapping speech, this model is trained to preserve the main speaker while suppressing other speakers.)

NVIDIA NemotronLabs RE-USE was developed by NVIDIA as a part of RE-USE.  

This model is ready for commercial use. 

### Performance

Comparison with other open-source USE models on the (subsets) of the URGENT 2025 non-blind test set, referenced against original anechoic
clean speech. 

**SBERT**: SpeechBERTScore, **LPS**: Levenshtein Phoneme Similarity,

**Spk**: Speaker similarity (ECAPA-TDNN), **CAcc**: Character accuracy of ASR (OWSM v3.1)


| Full set | # of parameters | DNSMOS | NISQA | UTMOS | PESQ | STOI | SBERT | LPS |  Spk | CAcc   |
| ------ | -----: | -----: | ----: | ----: | ---: | ---: | ----: | ---: | ---: | ----: |
| Noisy  |  - | 1.84 |  1.69 |  1.56 | 1.34 | 0.50 |  0.74 | 0.61 | 0.73 | 81.29 |
| Sidon  | 250M  | **3.34** |  3.87 |  **3.21** | 1.73 | 0.62 |  0.85 | 0.80 | 0.74 | 84.68 |
| UniPASE  | 500M  | 3.21 |  4.03 |  2.96 | 2.17 | 0.76 |  **0.89** | 0.85 | 0.81 | 87.28 |
| **NemotronLabs RE-USE** | **9.6M**  | 3.29 |  **4.33** |  2.94 | **2.57** | **0.78** |  0.88 | **0.85** | **0.86** | **89.17** |


| 16kHz subset  | DNSMOS | NISQA | UTMOS | PESQ | STOI | SBERT |  LPS |  Spk |  CAcc |
| ------- | -----: | ----: | ----: | ---: | ---: | ----: | ---: | ---: | ----: |
| Noisy   |   1.84 |  1.59 |  1.41 | 1.29 | 0.53 |  0.70 | 0.58 | 0.63 | 76.54 |
| UniSE   |   **3.23** |  3.71 |  **2.77** | 1.46 | 0.62 |  0.77 | 0.60 | 0.57 | 54.82 |
| **NemotronLabs RE-USE** |   3.19 |  **4.20** |  2.64 | **2.47** | **0.82** |  **0.87** | **0.84** | **0.82** | **87.09** |


| 44.1kHz subset           | DNSMOS | NISQA | UTMOS | PESQ | STOI | SBERT |  LPS |  Spk |  CAcc |
| ---------------- | -----: | ----: | ----: | ---: | ---: | ----: | ---: | ---: | ----: |
| Noisy            |   1.91 |  1.79 |  1.52 | 1.30 | 0.52 |  0.76 | 0.61 | 0.80 | 83.65 |
| Resemble Enhance |   3.13 |  3.68 |  2.11 | 1.33 | 0.41 |  0.69 | 0.46 | 0.57 | 47.22 |
| **NemotronLabs RE-USE**           |   **3.36** |  **4.44** |  **2.80** | **2.47** | **0.80** |  **0.89** | **0.86** | **0.90** | **91.54** |


| 48kHz subset                                            | DNSMOS | NISQA | UTMOS | PESQ | STOI | SBERT |  LPS |  Spk |  CACC |
| ------------------------------------------------- | -----: | ----: | ----: | ---: | ---: | ----: | ---: | ---: | ----: |
| Noisy                                             |   2.04 |  1.83 |  1.99 | 1.28 | 0.55 |  0.77 | 0.77 | 0.78 | 90.64 |
| ClearVoice                                      |   2.97 |  3.38 |  3.01 | 2.09 | 0.70 |  0.84 | 0.87 | 0.77 | 89.92 |
| **NemotronLabs RE-USE** |   **3.33** |  **4.49** |  **3.63** | **2.59** | **0.77** |  **0.89** | **0.91** | **0.86** | **91.05** |

## Usage
Directly try our [**Gradio Interactive Demo**](https://huggingface.co/spaces/nvidia/NVIDIA-NemotronLabs-RE-USE) by uploading your noisy audio/video !!

## Environment Setup
1. (For **Mamba** setup)Pre-built Docker environments can be downloaded [here](https://github.com/RoyChao19477/SEMamba?tab=readme-ov-file#-docker-support) to simplify **Mamba** setup.

2. If you need bandwidth extension:

```bash
pip install resampy 
```
3. Download and navigate to the HuggingFace repository:
```
huggingface-cli download nvidia/NVIDIA-NemotronLabs-RE-USE  --local-dir ./NemotronLabs_REUSE --local-dir-use-symlinks False
cd ./NemotronLabs_REUSE
```

## Inference
Follow the simple steps below to generate enhanced speech using our model:
1. Place your noisy speech files in the folder `noisy_audio/`
2. Run the following command:
```bash
sh inference.sh
```
3. The enhanced speech files will be saved in `enhanced_audio/`.

That's all !

**Note:**

a. You can enable bandwidth extension by setting the target bandwidth using the `BWE argument` in the script.

---

If your noisy speech files are **long (e.g., >30s, compared to our training data, which consists of clips < 10s)** or cause GPU out-of-memory (OOM) errors, please use the following procedure instead:
1. Place your long noisy speech files in the folder `long_noisy_audio/`
2. Run the following command:
```bash
sh inference_chunk.sh
```
3. The enhanced speech files will be saved in `Long_enhanced_audio/`.

**Note:**

a. You can also configure the `chunk_size_in_seconds` and `hop_length_portion` directly in the script.

### License/Terms of Use:
[OpenMDW License Agreement, version 1.1](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1) (OpenMDW-1.1)

This project is currently not accepting contributions.

### Deployment Geography:  
Global

### Use Case:  
Researchers and general users can use this model to enhance the quality of speech recordings across a wide range of applications. For example, content creators can use it to improve the speech quality of their recordings, while researchers can use it to enhance speech datasets for training speech generation models, such as text-to-speech (TTS) and speech large language models (Speech LLMs).  
  

### Release Date:
[Hugging Face](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) 10/22/2026 

  

## Reference(s):
[Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement (NeurIPS 2026)](https://arxiv.org/abs/2603.02641)  
[One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications (IEEE SLT 2026)](http://arxiv.org/abs/2606.25621)  

## Model Architecture:   
**Architecture Type:** Convolutional encoder, Convolutional decoder, and Mamba for time–frequency modeling   
**Network Architecture:** Bi-directional Mamba with 30 layers  
**Number of model parameters:** 9.6M (9.6*10^6)  
   
 
## Input:  
**Input Type(s):** Audio  
**Input Format(s):** .wav files  
**Input Parameters:** One-Dimensional (1D)  
**Other Properties Related to Input:** 8000 Hz - 48000 Hz Mono-channel Audio   
  

## Output:  
**Output Type(s):** Audio  
**Output Format:** .wav files  
**Output Parameters:** One-Dimensional (1D)  
**Other Properties Related to Output:** 8000 Hz - 48000 Hz Mono-channel Audio  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** N/A    
**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere  
**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  


## Model Version(s): 
NVIDIA-NemotronLabs-RE-USE_v1   
 
## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:** Audio  

**Audio Training Data Size:** ~2500 hours  
**Data Collection Method by dataset:** Synthetic  
**Labeling Method by dataset:** Synthetic  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The size of the speech data is roughly 2500 hours. Single channel, single speaker, 8kHz - 48kHz sampled data.

* [LibriVox data from DNS5 challenge (EN)](https://github.com/microsoft/DNS-Challenge/tree/master) (~350 hours of speech data)
* [LibriTTS (EN)](https://openslr.org/60/) (~200 hours of speech data)
* [VCTK (EN)](https://datashare.ed.ac.uk/handle/10283/3443) (~80 hours of speech data)
* [WSJ (EN)](https://catalog.ldc.upenn.edu/LDC93S6A) (~85 hours of speech data)
* [Multilingual Librispeech (De, En, Es, Fr)](https://www.openslr.org/94/) (~450 hours of speech data)
* [CommonVoice 19.0 (De, En, Es, Fr, zh-CN)](https://huggingface.co/datasets/fsicoli/common_voice_19_0) (~1300 hours of speech data)
* Riva Speakers: Internal Dataset
* [Audioset+FreeSound noise in DNS5 challenge](https://github.com/microsoft/DNS-Challenge/tree/master) (~180 hours of noise data)
* [Simulated RIRs from DNS5 challenge](https://github.com/microsoft/DNS-Challenge/tree/master) (~60k samples of room impulse response)
* [MicIRP](https://micirp.blogspot.com/p/about-micirp.html) (~70 samples of microphone impulse response)


### Testing/Evaluation Dataset:  

**Data Collection Method by dataset:** Synthetic  
**Labeling Method by dataset:** Synthetic  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** URGENT 2025 non-blind test set, evaluated against the original clean speech as the reference, without convolution with early reflections.


## Inference:
**Acceleration Engine:** Triton

**Hardware Requirements (GPU Architecture, Model):**  NVIDIA Ampere - A100

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/main/bias.md), [Explainability](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/main/explainability.md), [Safety & Security](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/main/safety_security.md), and [Privacy](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/main/privacy.md) Subcards.  
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.

```bibtex
@article{fu2026rethinking,
  title={Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement},
  author={Fu, Szu-Wei and Chao, Rong and Yang, Xuesong and Huang, Sung-Feng and Zezario, Ryandhimas E and Nasretdinov, Rauf and Juki{\'c}, Ante and Tsao, Yu and Wang, Yu-Chiang Frank},
  journal={arXiv preprint arXiv:2603.02641},
  year={2026}
}
```

```bibtex
@article{fu2026one,
  title={One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications},
  author={Fu, Szu-Wei and Chao, Rong and Yang, Xuesong and Huang, Sung-Feng and Juki{\'c}, Ante and Tsao, Yu and Wang, Yu-Chiang Frank},
  journal={arXiv preprint arXiv:2606.25621},
  year={2026}
}
```
# Crop Growth Stage Detection from Satellite Imagery

Sentinel-1 and Sentinel-2 are sourced from Google Earth Engine. The scripts for fetching them can be found in `Pheno_Crop.ipynb` at project root.

### References:

+ [Temporal Self Attention and Multi-Sensor Fusion](https://github.com/ellaampy/CropTypeMapping)
+ [Cloud Imputation - SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
+ [U-Net with Temporal Attention Encoder](https://github.com/Many98/Crop2Seg)
+ [Implementations of Cross-Attention Layers](https://github.com/likyoo/awesome-multimodal-remote-sensing-classification)
+ [L-TAE: Lightweight Temporal Attention Encoder](https://arxiv.org/abs/2007.00586)

# Models:

## 1. Dual Stream Network

- The data is passed through an async pipeline for pre-processing. It outputs Optical and Radar data. See `Model_Experimentation.ipynb` Model 1 for details.
- The Optical network consists of 1D CNN with time embedding and a CLS token with Multi-Head Attention. 
- The Radar network consists of 1D CNN with time embedding processed with a GRU.
- The outputs from both the networks are then concatenated (they run in parallel) in a fusion network for the final prediction.

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/96b4a82a-cf53-4994-9064-2d371b3818cf" />

**Latent Space:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/64d13128-29eb-44d5-a677-4ca0bfae8dc6" />

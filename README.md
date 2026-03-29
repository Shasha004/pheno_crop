# Crop Growth Stage Detection from Satellite Imagery

Sentinel-1 and Sentinel-2 are sourced from Google Earth Engine. The scripts for fetching them can be found in `Pheno_Crop_Data.ipynb` at project root.

The primary challenge in remote sensing lies in the nature of the data itself: agricultural environments are highly dynamic, and the satellite telemetry used to monitor them is inherently asynchronous and noisy. To capture the full spectrum of a winter wheat crop's growth cycle (Stages 0 through 4), this architecture relies on two independent data streams extracted via Google Earth Engine.

---

## 1. Dual Stream Hybrid Network

**Architecture Overview:**

- The data is passed through an async pipeline for pre-processing. It outputs Optical and Radar data. See `model_1.ipynb` Model 1 for details.
- The Optical network consists of a 1D CNN with time embedding and a CLS token with Multi-Head Attention.
- The Radar network consists of a 1D CNN with time embedding processed with a GRU.
- The outputs from both the networks are then concatenated (they run in parallel) in a fusion network for the final prediction.

### The Dual-Sensor Approach

The first stream is Sentinel-2 optical imagery, fetched via a custom automated pipeline that calculates specific vegetation indices such as NDVI, NDWI, NDRE, EVI, and SAVI at a 10-meter spatial resolution. While rich in chemical information regarding chlorophyll and leaf water content, this optical data is highly susceptible to atmospheric occlusion and cloud cover.

To counterbalance this, the pipeline integrates a second stream: Sentinel-1 Synthetic Aperture Radar (SAR). By capturing the Vertical-Vertical (VV) and Vertical-Horizontal (VH) backscatter, along with their derived ratio, the radar stream provides continuous, cloud-penetrating measurements of the physical canopy structure and biomass density.

### Asynchronous Temporal Alignment

To bridge the gap between this raw, asynchronous satellite telemetry and the rigid input requirements of standard neural networks, the data undergoes a temporal alignment process within the dataset generation phase. Traditional machine learning models often force multi-sensor data onto a shared, interpolated daily grid, which inevitably introduces hallucinated biological data during long cloudy periods.

Instead, this system utilizes a fully asynchronous timeline anchored directly to the ground truth observation, designated as Day 0. The pipeline looks backward over a 90-day window, extracting a maximum of 30 valid satellite events for each modality. Rather than interpolating, the system calculates a strict relative temporal distance, represented as a discrete `days_ago` integer, for every single reading.

Consequently, the dataset yields two entirely separate sets of tensors for every agricultural plot:

- A feature tensor containing the sensor values.
- A parallel time tensor containing the exact chronological placement of those values.

### The Optical Branch (Chemistry & Filtering)

Operating on this aligned temporal data, the first half of the architecture focuses on the chemical and biological signatures via the Optical Attention Branch.

The network ingests a tensor of shape $[\text{Batch}, 30, 6]$, representing the sequence of optical indices alongside the critical `cloud_pct` metric. In classical time-series analysis, researchers manually calculate temporal derivatives, such as the slope of NDVI, to determine how rapidly a crop is greening up. This approach is brittle and highly susceptible to sensor noise.

Instead, this architecture natively integrates a 1-Dimensional Convolutional Neural Network (1D CNN) with a sliding window of size 3 as its front-end feature extractor. As this convolutional window sweeps across the 30-day temporal sequence, it dynamically calculates local derivatives, smoothing minor atmospheric noise and extracting the biological momentum of the indices into a richer 64-dimensional latent space.

To ensure the network understands the massive, irregular gaps between these clear readings, the integer time tensor is passed through an $\text{Embedding}(120, 64)$ layer, generating a unique mathematical signature for the exact day the reading occurred. This temporal signature is added directly to the CNN output.

A learnable 64-dimensional classification token is then prepended to the sequence, expanding the temporal length to 31. This sequence is processed by a deep Transformer Encoder utilizing Multi-Head Attention. By analyzing the entire 90-day timeline simultaneously, the attention mechanism acts as a dynamic biological filter. When it encounters an event with a high cloud percentage and a consequently crashed NDVI, it mathematically drops the attention weight for that specific day to near zero.

The network actively routes around atmospheric corruption, mapping the long-term biological relationships of the clear days and compressing that knowledge entirely into the classification token, which serves as the final 64-dimensional optical summary vector.

> **Context Note:** The "Transformer Encoder" acts as a smart filter. Instead of stepping through the timeline day-by-day, it looks at the entire 3-month history at once. It learns to automatically mute data points that are corrupted by clouds, ensuring only clean biological trends are passed forward.

### The Radar Branch (Structure & Continuity)

In stark contrast to the optical stream, the second half of the network processes the Sentinel-1 SAR telemetry through a Recurrent Radar Branch.

The input here takes the shape of $[\text{Batch}, 30, 3]$, capturing the VV, VH, and VH/VV ratio. Because SAR data is inherently plagued by microwave interference known as speckle, a parallel 1D CNN acts as a learned speckle filter. It identifies short-term structural shapes, such as the steep drop in VH backscatter that reliably occurs just before harvest as the stalks dry out and lose physical volume.

After being fused with an identical 64-dimensional positional time embedding, the sequence is fed sequentially into a Gated Recurrent Unit (GRU) with 2 hidden layers. Because radar microwaves effortlessly punch through cloud cover, the Sentinel-1 timeline is highly continuous and rarely features the massive temporal gaps seen in the optical stream.

This continuous nature makes a recurrent memory network the mathematically optimal choice, allowing the GRU to step chronologically through the days, smoothly updating its internal hidden state to track the physical accumulation and eventual dry-down of the crop canopy.

The final hidden state is extracted as the 64-dimensional radar summary vector.

### Late-Fusion and Regularization

With the biological chemistry and physical structure independently encoded, the network employs a late-fusion strategy to make its final prediction. The optical and radar summary vectors are concatenated into a unified 128-dimensional multi-modal tensor.

This delayed integration is a critical architectural choice; by forcing the network to understand the optical curve and the radar curve entirely independently before combining them, the system prevents a highly noisy optical reading from preemptively corrupting the continuous radar memory state.

The fused manifold is finally passed through a Multi-Layer Perceptron classifier. This classifier scales the features down to 64 dimensions before outputting the final 5 class logits.

Crucially, this block utilizes Batch Normalization and a Dropout probability of 30 percent. This heavy regularization actively prevents the massive capacity of the dual-stream network from simply memorizing the specific layouts of the training plots, forcing it to generalize the underlying biophysical rules of crop phenology.

### Architectural Defenses and Physics Constraints

The structural integrity of this entire pipeline relies heavily on its ability to mathematically mirror physical realities, a trait that directly answers the most common technical critiques in satellite machine learning.

For instance, differing acquisition dates between sensors are handled flawlessly precisely because the sensors are never artificially forced onto the same temporal grid; the asynchronous embeddings ensure that an optical reading from 14 days ago and a radar reading from 12 days ago are treated as distinct physical events.

Furthermore, SAR backscatter is highly sensitive to the satellite's specific viewing geometry, particularly the incident angle and whether the orbit pass is ascending or descending. This architecture actively mitigates these geometry-induced variations by utilizing the VH/VV ratio as a core input feature, which mathematically cancels out much of the background soil moisture and baseline structural variation. Coupled with the local smoothing of the front-end CNN, this prevents the recurrent network from overreacting to sudden backscatter jumps caused purely by alternating orbit directions.

Finally, while lighter architectures like the Lightweight Temporal Attention Encoder rely on a single master query to extract temporal features, this system utilizes a full self-attention stack. By allowing every temporal observation to mathematically attend to every other observation within the lookback window, the network captures much deeper, non-linear chronological relationships, such as explicitly linking the exact day of peak vegetative green-up to the exact day the harvest senescence begins.

---

## Results

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/96b4a82a-cf53-4994-9064-2d371b3818cf" />

**Latent Space:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/64d13128-29eb-44d5-a677-4ca0bfae8dc6" />


> **Important Note on Current Metrics:** Some currently reported scores are likely inflated because of evaluation leakage risks, including event-level random splitting (instead of strict plot-level/group splits), temporal overlap within the same plot context between train/validation samples, and running inference over the full dataset used during model development. Treat those numbers as exploratory, not final generalization performance. Working on this is here: `model_2.ipynb`

---

## References

- [Temporal Self Attention and Multi-Sensor Fusion](https://github.com/ellaampy/CropTypeMapping)
- [Cloud Imputation - SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
- [U-Net with Temporal Attention Encoder](https://github.com/Many98/Crop2Seg)
- [Implementations of Cross-Attention Layers](https://github.com/likyoo/awesome-multimodal-remote-sensing-classification)
- [L-TAE: Lightweight Temporal Attention Encoder](https://arxiv.org/abs/2007.00586)

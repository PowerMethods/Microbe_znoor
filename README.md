# Introduction

Microbial transmission plays a central role in shaping human-associated microbiomes and has important implications for infectious disease dynamics, epidemiology, and public health. Advances in high-throughput sequencing have enabled detailed characterisation of microbial genomes across individuals, households, and populations. However, inferring *directionality of transmission*—that is, determining who infected whom—remains a challenging problem due to the massive dimensionality of genomic data, within-host diversity, and noise introduced by sequencing and sampling processes.

Traditional approaches to microbial transmission inference often rely on phylogenetic reconstruction, variant counting, or summary statistics derived from genomic distances. While effective in small-scale or well-curated datasets, these methods struggle to scale to whole-genome metagenomic data comprising millions of genomic positions and complex intra-host variation. Moreover, such approaches typically require strong modeling assumptions and extensive manual feature engineering, which may limit their ability to capture subtle, high-dimensional transmission signals.

Recent progress in deep learning offers an alternative paradigm. Neural networks can learn discriminative representations directly from raw or minimally processed data, making them particularly suitable for large-scale genomic applications. In this thesis, deep learning models are explored for inferring microbial transmission directionality from paired metagenomic samples. The core hypothesis is that direction-specific genomic signatures—encoded in patterns of diversity, entropy, and nucleotide composition—can be learned automatically when genomic data are represented in a structured, segment-based format.

This work focuses on three complementary deep-learning paradigms: **Convolutional Neural Networks (CNNs)**, **Transformer-based models**, and **Variational Autoencoders (VAEs)**. Each model family captures different aspects of the data, ranging from local spatial patterns to long-range dependencies and latent structure. By comparing and integrating insights from these approaches, the thesis aims to provide a robust and scalable framework for transmission directionality inference.

---

# Deep Learning Framework for Transmission Directionality

## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks form the primary modeling backbone of this thesis. CNNs are particularly well suited for genomic data represented as structured matrices, where local spatial patterns are informative. In this work, paired metagenomic samples are transformed into fixed-size tensors representing genomic windows, with multiple channels encoding per-position metrics such as entropy, diversity indices, and nucleotide frequencies.

To address the challenge of whole-genome scale data, a **segment-based CNN approach** is adopted. Instead of processing entire genomes at once, the genome is divided into fixed-length segments composed of multiple consecutive windows. Each segment is treated as an independent training example, enabling efficient learning while preserving local genomic context. Segment-level predictions are later aggregated at the pair level to infer the final transmission direction.

CNNs are chosen due to their stability during training, strong inductive bias toward local pattern detection, and relatively low computational overhead compared to attention-based models. These properties make CNNs particularly suitable for high-performance computing (HPC) environments and large-scale experimental pipelines.

**Sections to be expanded:**

* CNN input representation and feature channels
* CNN256 architecture and design rationale
* Segment-level training and pair-level aggregation
* Loss functions, class imbalance handling, and optimization

---

## Transformer-Based Models

Transformer architectures have emerged as a powerful alternative to convolutional models, particularly in domains requiring modeling of long-range dependencies. Unlike CNNs, Transformers rely on self-attention mechanisms to dynamically weight relationships between distant positions in an input sequence. This property is appealing for genomic data, where informative patterns may span large genomic distances.

In the context of microbial transmission inference, Transformers are explored as a complementary modeling approach. Genomic windows or segments are treated as sequences, allowing the model to learn global dependencies across the genome. However, the high dimensionality of metagenomic data introduces practical challenges, including memory constraints, sensitivity to feature scaling, and training instability.

Within this thesis, Transformer models are investigated experimentally to assess their ability to capture transmission-relevant signals. Their performance and limitations are critically compared against CNN-based models, with particular attention to stability, interpretability, and scalability.

**Sections to be expanded:**

* Transformer input encoding for genomic data
* Attention mechanisms and positional encoding
* Training stability and normalization strategies
* Comparative analysis with CNN-based models

---

## Variational Autoencoders (VAEs)

Variational Autoencoders provide a probabilistic framework for learning compact latent representations of high-dimensional data. Unlike discriminative models such as CNNs and Transformers, VAEs are generative models that aim to capture the underlying structure of the data distribution. In this thesis, VAEs are explored as an unsupervised or semi-supervised tool for understanding genomic variation patterns between paired samples.

By encoding genomic segments into a low-dimensional latent space, VAEs can reveal clustering structures, variation gradients, and uncertainty patterns that may correspond to transmission dynamics. These latent representations can be used for exploratory analysis, anomaly detection, or as input features for downstream classifiers.

VAEs are particularly valuable for assessing whether transmission directionality signals emerge naturally from the data without explicit supervision. Their probabilistic nature also enables uncertainty quantification, which is an important consideration in biological inference.

**Sections to be expanded:**

* VAE architecture and latent space design
* Reconstruction loss and regularization (KL divergence)
* Interpretation of latent representations
* Integration of VAE outputs with supervised models

---

# Thesis Structure (Planned)

1. Introduction and background
2. Data generation and preprocessing pipeline
3. CNN-based transmission directionality modeling
4. Transformer-based modeling and comparative analysis
5. Variational Autoencoder exploration of genomic structure
6. Evaluation methodology and metrics
7. Results and discussion
8. Limitations and future work
9. Conclusion

---

This modular structure allows each modeling approach to be developed and evaluated independently while maintaining a coherent narrative centered on the problem of microbial transmission directionality.

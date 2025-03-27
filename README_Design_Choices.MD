# Model and Architectural Choices

The study focuses on developing a Computer-Aided Detection (CADe) system to assist endoscopists in identifying early neoplasia in Barrett’s Esophagus. Due to subtle lesion appearance and variability in image quality, the architecture choice became crucial.

## Backbone: CaFormer-S18

We conducted an extensive comparison of 10 state-of-the-art backbone architectures, including CNN-based, Transformer-based, hybrid CNN-Transformer, and State-Space Models (SSM). The CaFormer-S18 was selected as it offered the best classification performance on the key validation sets (Images + High Quality (HQ) frames and Medium Quality (MQ) frames), indicating its ability to robustly handle varied quality scenarios.

Reasons for Choosing CaFormer-S18:
* Hybrid CNN-Transformer Architecture:
    This architecture leverages the strengths of CNNs (local spatial features, robustness to image distortions) and Transformers (global context awareness, better generalization) to effectively handle the complexity and variability of endoscopic imagery.
* Performance Gains:
    CaFormer-S18 consistently demonstrated higher AUROC and AUPRC metrics compared to other models, especially for the critical validation sets (clean and medium-quality images).

## Decoder head: FPN (Feature Pyramid Network)

Four segmentation decoders were compared (FPN, UperNet, DeepLabV3+, and U-Net++). Although the DeepLabV3+ slightly outperformed the FPN in certain segmentation metrics, the FPN was chosen due to its balanced performance and lightweight design, offering substantial performance with fewer parameters.
Choosing FPN:
* Computational Efficiency: Lower complexity and fewer parameters compared to other competitive segmentation decoders.
* Balanced Performance: Achieved competitive segmentation accuracy, closely matching DeepLabV3+ in AUROC and AUPRC segmentation metrics.

## Pretraining Strategies

#### Domain-Specific Pretraining: GastroNet SSL

The selected backbone (CaFormer-S18) was pretrained using GastroNet-5M, a large unlabeled dataset specifically from gastrointestinal endoscopic images, using a self-supervised learning (SSL) approach (DINO framework). This method significantly outperformed standard ImageNet pretraining, indicating improved generalization and robustness to variability in endoscopic image quality.
* GastroNet SSL pretraining notably enhanced performance across all validation sets.
* Specifically improved results on lower-quality validation images (MQ, LQ), demonstrating robustness to domain-specific challenges (motion blur, poor lighting).

## Data Augmentation Strategy
Two types of data augmentations were evaluated: non-domain-specific augmentations (N-DSA) and domain-specific augmentations (DSA).
Findings:
* Employing DSA improved performance on MQ and LQ frames, highlighting robustness improvements.
* However, using DSA at both SSL pretraining and downstream training stages compromised performance on clean images, prompting the authors to choose the default N-DSA strategy for both stages to maintain high accuracy on optimal images.

## Additional Design Choices
Inclusion of lower-quality video frames (HQ, MQ, LQ) in training demonstrated incremental improvements in performance, especially regarding robustness against medium-quality frames. The use of additional frames improved segmentation performance without additional manual segmentation annotations, providing a cost-effective strategy for performance enhancement.
Findings:
* Incorporating additional MQ and HQ frames offered significant improvements in robustness.
* Maximum improvements observed when a balanced amount of medium-quality video frames were added to the training set.

## Segmentation Ground Truth Utilization

Due to variability in expert segmentation annotations, several ground-truth consensus strategies (soft-spot, plausible-spot, sweet-spot, hard-spot) were evaluated.
Chosen Strategy: "Plausible-spot GT"
* Balances sensitivity (detection of subtle lesions) with specificity, providing robust segmentation annotations.

## Inference Strategy Choices
The model output decision strategies were explored, including combining classification and segmentation outputs through averaging, logical operations (OR, AND), or using individual outputs.
Recommended Inference Strategy:
* Averaging classification and segmentation outputs offered improved accuracy and consistency in clinical deployment, ensuring a balanced sensitivity-specificity trade-off.

## TensorRT Optimization
The authors employed TensorRT to optimize inference speed, enhancing efficiency without significant loss in performance. This allowed potential real-time clinical integration on resource-constrained hardware.

## Benchmarking Results
When benchmarked against six state-of-the-art architectures for CADe Barrett’s neoplasia detection, the optimized CaFormer-S18 with FPN showed very strong performance:
* Classification and localization
* Statistical Significance: Achieved statistical significance (p-values between 0.0019 and 0.031), demonstrating clear superiority.

# Conclusion and Clinical Relevance
USE THIS MODEL
# Constrained Concordance Index
Constraiend Concordance Index (CCI) is a performance evaluation metric for speech, image, and multimedia quality models. It is based on measuring the ability of MOS predictors to rank stimuli pairs where MOS difference is statistically significant. Experiments in this repo compare CCI against Pearson's correlation coefficient (PCC), Spearman's rank correlation coefficient (SRCC), Kendall's Tau (KTAU). 
We have used 2 speech quality models (PESQ, VISQOL), 1 image quality model (SSIM) and 4 MOS databases (P23 EXP1, P23 EXP3, TCD-VOIP, JPEG XR) to validate the CCI metric.  

Coefficients calculated on the full database are called population parameters and indicated with $\rho$. Sample estimates are indicated with $\hat{\rho}$ e.g. PCC calculated on a random subset of ViSQOL predictions on the TCD-VoIP database.

<!-- ##  Indicators
* $|\hat{\rho}(n) - \rho|$ is the difference between the sample mean $\hat{\rho}$ and the population mean $\rho$. The sample mean depends on the sample size $n$. For each $n$ we estimate the parameter $S$ times, which is the number of samples. The final parameter $\hat{\rho}(n)$ is the mean of the $S$ sample means for a particular $N$.
* $c_{v}(\%)(n)$ is the coefficient of variation or relative standard deviation. -->


## Experiment 1
<!-- This experiment evaluates the robustness of CCI against small sample sizes. The experiment consists of incrementally varying the sample size and measuring the difference between the CCI sample and the CCI population. The closer is the difference the better. Samples are taken randomly from the population to avoid bias sampling.

| Dataset  | N - Sample Size             | $S$ Num of samples | Comparison                 | Measures |
| -------- | -----------                 | -----------    | -------                    | -------- |
| TCD-VoIP | From 0 to 382 log spaced    | 200            | Pearson, Spearman, Kendall | Mean Difference, Std Difference, Difference between 5th and 95th percentile and the population mean -->
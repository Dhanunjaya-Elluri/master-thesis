![tests](https://github.com/Dhanunjaya-Elluri/master-thesis/actions/workflows/tests.yaml/badge.svg)

# Master's Thesis

## Transformers for quantized time series forecasting (Work in progress)

The utilization of Transformer models [4], originally designed for natural language processing tasks,
has garnered attention due to their remarkable ability to predict subsequent elements in a sequence.
These models, often referred to as "stochastic parrots," have exhibited a profound mastery of learning
sequential patterns, leading to text generation that closely mimics human-written content. This
remarkable capacity suggests that Transformers might offer a powerful approach to time series forecasting,
leveraging their inherent understanding of sequential dependencies.

In the context of time series forecasting, accurate quantization holds significant potential. By representing continuous data as discrete symbols from a finite alphabet, the complexity of the input domain is reduced, simplifying the forecasting task. This reduction in complexity is particularly appealing for enhancing human interpretability, a crucial factor for the practical application of forecasting methods. Addressing this aspect could bridge the gap between advanced machine learning techniques and human comprehension, facilitating the adoption of cutting-edge forecasting methods in real-world scenarios.

A crucial aspect pertains to the design of time-series-specific positional encodings within the context of quantization. As the focus shifts to a finite alphabet, novel positional encoding strategies could hold the key to capturing the essential temporal dynamics of the data. This exploration aligns with the objective of uncovering more efficient representations that align with the discrete nature of the quantized input domain. Moreover, the proposal aims to conduct a comparative analysis of two prominent quantization methods: Vanilla Symbolic Aggregate Approximation (SAX) [3] and Kernel SAX [1]. While Vanilla SAX assumes a normal distribution of data, Kernel SAX capitalizes on estimating data density for quantization. The flexibility of the Kernel SAX approach raises intriguing questions about its potential to outperform Vanilla SAX in terms of forecasting accuracy. Understanding the practical implications of these quantization methods could provide valuable insights into their applicability in diverse forecasting scenarios.

## Goal of the Thesis

The primary goal of this thesis is to investigate and enhance the effectiveness of time series forecasting through a comprehensive exploration of quantization and transformative modeling techniques. The research aims to achieve the following objectives:

### Quantization with Kernel SAX Method

The thesis seeks to quantize univariate time series data using the Kernel SAX method, transforming continuous data into discrete symbols from a finite alphabet. The focus on Kernel SAX, with its ability to estimate data density for quantization, aims to capture nuanced patterns inherent in the time series data while simplifying the input domain for subsequent analysis.

### Leveraging Transformers for Forecasting

Building upon the groundwork laid out in "Are Transformers Effective for Time Series Forecasting," [5] the thesis aims to apply a curated selection of Transformer models to the quantized time series data. These Transformers, originally designed for natural language processing, will be adapted to the time series forecasting domain to explore their potential for capturing intricate temporal dependencies and predicting future values.

### Distance Metric-driven Evaluation

The research seeks to establish a rigorous evaluation framework by employing appropriate distance metrics to assess the forecasting performance of the quantized alphabets. By meticulously measuring the dissimilarity between predicted and actual values, the thesis aims to provide a comprehensive understanding of the forecasting accuracy achieved through the proposed approach.

### Benchmarking and Generalization

To ensure the robustness and generalization of the proposed methodologies, the thesis will benchmark its techniques across a diverse set of 50 univariate time series datasets [2]. This extensive benchmarking will allow for a thorough assessment of the proposed methods’ efficacy across various real-world scenarios and data characteristics.

## Getting Started

To reproduce the experiments and analyses conducted in this thesis, follow these steps:

Clone this repository:
`git clone https://github.com/Dhanunjaya-Elluri/master-thesis.git`

Install the required dependencies:
`pip install -r requirements.txt`

Navigate to the `notebooks/` directory and follow the notebooks in numerical order to reproduce the experiments step by step.

## References

[1] Konstantinos Bountrogiannis, George Tzagkarakis, and Panagiotis Tsakalides. “Data-driven Kernelbased
Probabilistic SAX for Time Series Dimensionality Reduction”. In: 2020 28th European Signal
Processing Conference (EUSIPCO). 2021, pp. 2343–2347. DOI: 10.23919/Eusipco47968.2020.9287311.

[2] Rakshitha Godahewa et al. “Monash Time Series Forecasting Archive”. In: Neural Information
Processing Systems Track on Datasets and Benchmarks. 2021.

[3] Jessica Lin et al. “Experiencing SAX: a novel symbolic representation of time series”. In: Data
Mining and Knowledge Discovery 15.2 (2007), pp. 107–144.

[4] Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information processing
systems. Vol. 30. 2017.

[5] A. Zeng et al. “Are Transformers Effective for Time Series Forecasting?” In: ArXiv preprint
(2022). arXiv: 2205.13504 [cs.LG].

## Contact

If you have any questions, suggestions, or issues regarding this repository or the implemented model, please feel free to contact the author:

Dhanunjaya Elluri <br>
Email: <dhanunjayet@gmail.com> | <dhanunjaya.elluri@tu-dortmund.de>

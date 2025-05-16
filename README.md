# Carbon-Twin Dataset

## Overview
The Carbon-Twin dataset is a comprehensive collection of remote sensing data focused on carbon cycle processes. This dataset is designed to support research in climate science, environmental monitoring, and ecological studies. By providing detailed observations and measurements, the Carbon-Twin dataset aims to enhance our understanding of carbon dynamics and their impact on global climate change.

## Features
- **Global-Scale Coverage**: The first global-scale machine learning-ready dataset for monitoring and forecasting forest carbon dynamics, provided at a 0.5° resolution.
- **Extended Temporal Range**: Spans over 40 years, offering a comprehensive historical perspective for long-term analysis.
- **Extensive Variable Set**: Covers over 100+ variables integrated from heterogeneous sources, ensuring a rich and diverse dataset.
- **Scenario-Based Training and Testing**: Includes different scenarios to resemble diverse application conditions, enhancing the dataset's versatility for various research needs.
- **Benchmark Experiments**: Provides benchmark experiments with a suite of machine learning forecasting models and new metrics specifically designed for the carbon forecasting problem.


## Data Information
### Dataset Access and Versioning
The dataset will be publicly available on Hugging Face and Zenodo after the publication.


### Source Details and Licensing

All datasets used in Carbon-Twin are openly accessible. Below are the details for each data source:

- **Meteorological Data**:  
  - Source: [NASA’s Modern-Era Retrospective analysis for Research and Applications, version 2](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) by [Gelaro et al., 2017](https://doi.org/10.1175/JCLI-D-16-0758.1).
  - Usage: Please reference the relevatn [paper citation requirements](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/citing_MERRA-2/).

- **CO₂ Data**:  
  - Source: [NOAA CarbonTracker](https://gml.noaa.gov/ccgg/carbontracker/) by [Peters et al., 2007](https://doi.org/10.1073/pnas.0708986104).
  - Usage: Please reference the relevant [paper citation requirements](https://gml.noaa.gov/ccgg/carbontracker/CT2007/citation.php).

- **Land Use Change Data**:  
  - Sources:
    - [Land Use Harmonization v2](https://luh.umd.edu/) by [Hurtt et al., 2019](https://doi.org/10.22033/ESGF/input4MIPs.10454) and [2020](https://doi.org/10.5194/gmd-13-5425-2020).
    - [Global Fire Emissions Database v4 (GFED4)](https://daac.ornl.gov/VEGETATION/guides/fire_emissions_v4_R1.html) by [Randerson et al., 2018](https://doi.org/10.3334/ORNLDAAC/1293).
  - Usage: Refer to the respective publications.

- **Soil Properties**:  
  - Source: [ROSETTA](https://doi.pangaea.de/10.1594/PANGAEA.870605) by [Montzka et al., 2017](https://doi.org/10.5194/essd-9-529-2017).
  - Usage: Available under the Creative Commons Attribution 3.0 License.

- **Climate Classification Data**:  
  - Source: [Köppen-Geiger Climate Classification](https://figshare.com/articles/dataset/Present_and_future_K_ppen-Geiger_climate_classification_maps_at_1-km_resolution/6396959/2) by [Beck et al., 2018](https://doi.org/10.1038/sdata.2018.214).
  - Usage: Available under the Creative Commons Attribution 4.0 International License.


## Benchmarks
The repository includes codes for the following methods:
- **LSTM**: A standard LSTM taking time-series ED inputs and outputting targets ([Graves, 2012](https://www.cs.toronto.edu/~graves/phd.pdf)).
- **LSTNet**: A time-series model extracting short-term dependency using convolution along the time-feature dimension and long-term temporal patterns using RNN ([Lai et al., 2018](https://arxiv.org/abs/1703.07015)).
- **DeepED**: An LSTM-based deep learning emulator for ED with specialized designs on error accumulation reduction and knowledge-guided learning ([Wang et al., 2023](https://doi.org/10.1145/3589132.3625577)).
- **Transformer (TF)**: A vanilla transformer model with a self-attention mechanism to capture dependencies across all time steps during the forecasting ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).
- **Informer (IF)**: A transformer variant with a ProbSparse attention for reduced complexity ([Zhou et al., 2021](https://arxiv.org/abs/2012.07436)).
- **DLinear (DL)**: A linear decomposition model to separate data into trend and seasonality, showing comparable performance among transformer-based models while maintaining efficiency ([Zeng et al., 2023](https://arxiv.org/abs/2205.13504)).
- **Crossformer (CF)**: A transformer variant with a two-stage attention mechanism for modeling both cross-time and cross-feature dependencies ([Zhang et al., 2023](https://openreview.net/forum?id=vSVLM2j9eie)).


## Contacts
Feel free to contact Zhihao Wang (zhwang1@umd.edu) and Yiqun Xie (xie@umd.edu) for any questions.

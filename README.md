# Deep-Learning-Based Gridded Downscaling of Surface Meteorological Variables

This repository contains supplemental information of the following two publications:

* Sha, Y., D. J. Gagne II, G. West, and R. Stull, 2020a: Deep-learning-based gridded downscaling of surface meteorological variables in complex terrain. 
Part I: Daily maximum and minimum 2-m temperature. J. Appl. Meteor. Climatol., 59, 2057–2073, https://doi.org/10.1175/JAMC-D-20-0057.1.

* Sha, Y., D. J. Gagne II, G. West, and R. Stull, 2020b: Deep-learning-based gridded downscaling of surface meteorological variables in complex terrain. 
Part II: Daily precipitation. J. Appl. Meteor. Climatol., 59, 2075–2092, https://doi.org/10.1175/JAMC-D-20-0058.1.

# Overview

A deterministic encoder-decoder convolutional neural network, UNet, is applied for the gridded downscaling of daily maximum and minimum 2-m temperature (Sha et al. 2020a) and precipitation (Sha et al.  2020b). 

For the downscaling of 2-m temperature, UNet takes low resolution (LR) temperature, LR elevation and high resolution (HR) elevation as three inputs and is trained based on the HR temperature as targets. The original UNet is also modified (named as "UNet-AE") by assigning an extra HR elevation output branch/loss function. UNet-AE is trained on both the supervised HR temperature and the unsupervised HR elevation (one of the inputs). UNet-AE has a slightly better transfer learning performance than the original UNet. When 2-m temperature downscaling is needed in a new spatial region, UNet-AE can be fine-tuned through its HR elevation output branch (elevation is available worldwide, but HR temperature is available in very limited areas).

For the downscaling of daily precipitation,  UNet takes (LR) precipitation, HR precipitation climatology, and HR elevation as three inputs and is trained based on the HR precipitation as targets. Based on the skewed distribution of precipitation (massive drizzle events and infrequently occurred extreme events), a variant of UNet (named "Nest-UNet") is considered and found to improve the downscaling performance. The idea of Nest-UNet is based on the work of UNet++ (Zhou et al. 2018).

# Data

In coming ...

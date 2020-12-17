# Deep-Learning-Based Gridded Downscaling of Surface Meteorological Variables

This repository contains supplemental information of the following two publications:

* Sha, Y., D. J. Gagne II, G. West, and R. Stull, 2020a: Deep-learning-based gridded downscaling of surface meteorological variables in complex terrain. 
Part I: Daily maximum and minimum 2-m temperature. J. Appl. Meteor. Climatol., 59, 2057–2073, https://doi.org/10.1175/JAMC-D-20-0057.1.

* Sha, Y., D. J. Gagne II, G. West, and R. Stull, 2020b: Deep-learning-based gridded downscaling of surface meteorological variables in complex terrain. 
Part II: Daily precipitation. J. Appl. Meteor. Climatol., 59, 2075–2092, https://doi.org/10.1175/JAMC-D-20-0058.1.

# Overview

A deterministic encoder-decoder convolutional neural network, UNet, is applied to the gridded downscaling of daily maximum/minimum 2-m temperature (TMAX/TMIN; Sha et al. 2020a) and precipitation (Sha et al.  2020b). 

For the downscaling of TMAX/TMIN, UNet takes low-resolution (LR) TMAX/TMIN, LR elevation and high-resolution (HR) elevation as three inputs and is trained based on the HR TMAX/TMIN. The original UNet is also modified (named as "UNet-AE") by assigning an extra HR elevation output branch. UNet-AE is trained on both the supervised loss of HR TMAX/TMIN and the unsupervised loss of HR elevation (elevation is one of the inputs). UNet-AE showed slightly better transfer learning performance than the original UNet. When 2-m temperature downscaling is needed in a new spatial region, UNet-AE can be fine-tuned by its HR elevation output branch (elevation is available worldwide, but HR 2-m temperature is available in very limited areas).

For the downscaling of daily precipitation,  UNet takes (LR) precipitation, HR precipitation climatology, and HR elevation as three inputs and is trained based on the HR precipitation as targets. Based on the skewed distribution of precipitation, that said, massive drizzle events and infrequently occurred extreme events, a variant of UNet (named "Nest-UNet") is considered and found to improve the downscaling performance. The idea of Nest-UNet is based on the work of UNet++ (Zhou et al. 2018).

**Identified issues**

1. The fine-tuning steps of Table 2, Sha et al. (2020a) can improve the performance of UNet-AE. However, the authors found that adversarial training can show even larger benefits. The transition from semi-supervised fine-tuning to adversarial training is not complicated --- replacing the HR elevation output branch to a CNN classifier and update UNet with classification loss.

2. when UNet is applied to precipitation downscaling, a shift-of-distribution can be found. Nest-UNet is doing slightly better but not free of this issue. Replacing the thresholding approach in Sha et al. (2020b) with a grid-point-wise quantile mapping step can solve the problem.

3. The authors are still working on this downscaling project. Contacting us if you have any concerns.

# Data

HR TMAX/TMIN and daily precipitation fields are obtained from the 4-km PRISM (Parameter Regressions on Independent Slopes Model).

The near-real-time PRISM, and PRISM monthly normals in the Continental US are availabe at the PRISM Climate Group (this repository provides an example of downloading script):

* PRISM Climate Group, 2004: Daily total precipitation and monthlynormals. Oregon State University, https://prism.oregonstate.edu.

The 800-m PRISM precipitation monthly normals (we re-gridded them  to 4-km) are available at the Pacific Climate Impacts Consortium website:

* Pacific Climate Impacts Consortium, 2014: High resolution PRISM climatology. And monthly time series portal. Pacific Climate Impacts Consortium, University of Victoria, and PRISM Climate Group, Oregon State University, https://www.pacificclimate.org/data/prism-climatology-and-monthly-timeseries-portal.

Elevation data is obtained from ETOPO1 1 Arc-Minute Global Relief Model (accessible through NGDC/NOAA website):

* Amante, C., and B. Eakins, 2009: ETOPO1 arc-minute global reliefmodel: Procedures, data sources and analysis. NOAA Tech.Memo. NESDIS NGDC-24, 25 pp., https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/docs/ETOPO

**Preprocessing notes**

* Elevation and 2-m temperature re-gridding is based on cubic interpolation, e.g., `scipy.interpolate.interp2d`. For precipitation, bilinear scheme is applied, and negative values are corrected to zero.

* Ocean grid points, not-a-number values are corrected to zero.

* 2-m temperature is standardized, precipitation is normalized through power transformation [log(X+1)] and a minimum-maximum normalization. Elevation is standarized when paired with 2-m temperature, and is normalized to [0, 1] when paired with precipitation.

# Contact

Yingkai (Kyle) Sha <yingkai@eoas.ubc.ca>


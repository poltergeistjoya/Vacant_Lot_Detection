# Clustering NYC Lots

1.  **Data Loading:** Load MapPluto data (e.g., GeoJSON or Shapefile) and conceptually define NAIP data access.
    - MapPluto 22v3 (around fall) shoreline clipped (no water)
2.  **Feature Calculation:** Placeholder for GEE-based feature extraction, and local calculation of geometric features.
3.  **Data Preparation:** Feature scaling.
4.  **Clustering:** K-Means implementation.
5.  **Visualization:** Basic plotting of clusters.


From Manus AI
## Research Findings: Remote Sensing Features and Clustering for Vacant Lot Identification

### Remote Sensing Features (NAIP Data)

NAIP imagery typically provides Red, Green, Blue (RGB) and Near-Infrared (NIR) bands. These bands can be used to derive various spectral features useful for distinguishing vacant land from other urban land covers.

**Key Spectral Features:**

*   **Raw Band Values:** Red, Green, Blue, Near-Infrared (NIR) band reflectance values directly provide information about the surface cover.
*   **Vegetation Indices:**
    *   **Normalized Difference Vegetation Index (NDVI):** `(NIR - Red) / (NIR + Red)`. High values indicate dense vegetation, while low or negative values suggest bare soil, water, or impervious surfaces. Vacant lots often exhibit lower NDVI than vegetated areas but higher than fully impervious surfaces.
    *   **Soil-Adjusted Vegetation Index (SAVI):** `((NIR - Red) / (NIR + Red + L)) * (1 + L)`, where L is a soil brightness correction factor (typically 0.5). Useful in areas with varying soil brightness.
    *   **Normalized Difference Built-up Index (NDBI):** `(SWIR - NIR) / (SWIR + NIR)`. While NAIP typically lacks SWIR, this concept highlights the importance of distinguishing built-up areas. Proxies or alternative indices might be considered if SWIR-like information can be derived or approximated.
*   **Bare Soil Indices:**
    *   **Normalized Difference Bare Soil Index (NDBSI):** `(SWIR - Green) / (SWIR + Green)`. Similar to NDBI, this often requires SWIR. However, the principle is to highlight bare soil. With NAIP, a combination of low NDVI and specific RGB/NIR reflectance patterns might serve as a proxy.
    *   **Brightness Index (BI):** A simple index often derived from multiple bands to represent overall brightness, which can be high for bare soil or impervious surfaces.
*   **Texture Features:** High-resolution imagery like NAIP allows for the extraction of texture features (e.g., using Gray-Level Co-occurrence Matrix - GLCM). Vacant lots might exhibit distinct texture patterns compared to regularly maintained lawns, paved areas, or buildings.
    *   **Homogeneity, Contrast, Dissimilarity, Entropy, Angular Second Moment:** These GLCM features can capture the spatial arrangement and variability of pixel intensities, which can differentiate between smooth surfaces (pavement), uniform vegetation, and heterogeneous vacant lots.
*   **Geometric Features (from MapPluto):**
    *   **Area:** Size of the parcel.
    *   **Perimeter:** Boundary length of the parcel.
    *   **Shape Index/Compactness:** Ratio of perimeter to area, indicating how irregular a parcel's shape is. Irregular shapes might correlate with less developed or harder-to-develop parcels.
    *   **Centroid Coordinates:** Location information (latitude, longitude).
    *   **Proximity to infrastructure/amenities:** (Derived) Distance to roads, public transport, parks, etc., which can influence development potential.

### Clustering Approaches

Clustering algorithms are suitable for unsupervised classification, grouping similar pixels or parcels based on their feature values. For remote sensing and urban analysis, several methods are commonly used:

*   **K-Means:** A popular, efficient, and widely used algorithm that partitions data into `k` clusters, where each data point belongs to the cluster with the nearest mean. It requires specifying the number of clusters (`k`) beforehand.
*   **ISODATA (Iterative Self-Organizing Data Analysis Technique Algorithm):** An unsupervised clustering algorithm that is a more sophisticated version of K-Means. It allows the number of clusters to change during the iteration process by merging and splitting clusters based on certain criteria. This can be advantageous when the optimal `k` is unknown.
*   **Hierarchical Clustering:** Builds a hierarchy of clusters. It can be agglomerative (bottom-up, starting with individual points and merging them) or divisive (top-down, starting with one large cluster and splitting it). It does not require specifying `k` in advance but can be computationally intensive for large datasets.
*   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It's good for discovering clusters of arbitrary shape and handling noise, but can struggle with varying densities.
*   **Gaussian Mixture Models (GMM):** A probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions with unknown parameters. It provides a more flexible clustering approach than K-Means by modeling the covariance of clusters.

**Considerations for NYC MapPluto Data (800k parcels):**

*   **Scalability:** K-Means and ISODATA are generally more scalable for large datasets than hierarchical clustering. DBSCAN can also be efficient if implemented correctly.
*   **Subset Approach:** Given the large number of parcels, working with a subset for initial exploration and model development is a pragmatic approach, as suggested by the user.
*   **Integration with GIS:** The clustering results will need to be integrated back with the MapPluto polygons. This implies that the features extracted from NAIP imagery should be aggregated or summarized per parcel (e.g., mean, median, standard deviation of spectral/texture features within each parcel polygon).

### Recommended Approach

For this task, a combination of spectral and texture features derived from NAIP, combined with geometric features from MapPluto, will provide a robust feature set. K-Means or ISODATA are good starting points for clustering due to their scalability and interpretability. DBSCAN or GMM could be explored for more nuanced cluster shapes if initial results are not satisfactory.

=====================================================


# EDA and Clustering Methodology for Vacant Lot Identification

This document outlines a comprehensive methodology for Exploratory Data Analysis (EDA) and clustering to identify vacant lots in urban areas, specifically focusing on Northeast America using NAIP imagery and NYC MapPluto data. The approach integrates remote sensing features with parcel-level geographic information to enhance the accuracy of vacant lot detection.

## 1. Data Sources

*   **National Agricultural Imagery Program (NAIP) Data:** High-resolution (1-meter) aerial imagery providing Red, Green, Blue (RGB), and Near-Infrared (NIR) bands. This data will be the primary source for spectral and textural features.
*   **NYC MapPluto Dataset:** A comprehensive dataset of tax lots in New York City, containing geometric information (polygons) and various attribute data (e.g., land use, building characteristics, ownership). This dataset will define the parcels of interest and provide geometric features.

## 2. Exploratory Data Analysis (EDA) Strategy

EDA will be conducted on both the raw and derived features to understand their distributions, relationships, and potential for distinguishing vacant lots. Given the large dataset, initial EDA may focus on a representative subset.

### 2.1. NAIP Imagery EDA

*   **Visual Inspection:** Manually inspect NAIP imagery for selected parcels to visually identify characteristics of vacant lots (e.g., presence of bare soil, sparse vegetation, debris, lack of structures) versus developed lots or parks.
*   **Spectral Profile Analysis:** For a sample of known vacant and non-vacant parcels, extract and plot spectral reflectance values across the RGB and NIR bands. This helps in understanding the spectral signatures of different land cover types.
*   **Feature Distribution Analysis:** Analyze the distribution of derived spectral indices (NDVI, potentially bare soil proxies) and texture features (GLCM) using histograms and box plots. Compare distributions between visually identified vacant and non-vacant areas.

### 2.2. MapPluto Data EDA

*   **Attribute Summary Statistics:** Calculate descriptive statistics (mean, median, standard deviation, min, max) for relevant numerical attributes such as parcel area, building footprint area, and year built. This helps identify outliers or common characteristics.
*   **Spatial Distribution Analysis:** Visualize the spatial distribution of parcels with certain attributes (e.g., undeveloped parcels, parcels with low building value) to identify potential clusters or patterns of vacant land.
*   **Correlation Analysis:** Examine correlations between different MapPluto attributes and between MapPluto attributes and derived NAIP features to identify redundant or highly predictive features.

## 3. Feature Engineering Strategy

Features will be engineered from both NAIP imagery and MapPluto data to create a comprehensive set for clustering. The goal is to capture spectral, textural, and geometric characteristics that differentiate vacant lots.

### 3.1. Features from NAIP Imagery

For each parcel polygon from MapPluto, the following features will be extracted or aggregated from the underlying NAIP imagery:

*   **Spectral Band Statistics:** Mean, median, standard deviation, minimum, and maximum values for each of the Red, Green, Blue, and Near-Infrared (NIR) bands within each parcel polygon. These statistics capture the central tendency and variability of reflectance.
*   **Vegetation Indices:**
    *   **Normalized Difference Vegetation Index (NDVI):** `(NIR - Red) / (NIR + Red)`. Calculate mean, median, and standard deviation of NDVI within each parcel. Low NDVI values are indicative of bare soil or sparse vegetation, common in vacant lots.
    *   **Soil-Adjusted Vegetation Index (SAVI):** `((NIR - Red) / (NIR + Red + L)) * (1 + L)`, with L=0.5. Similar statistics as NDVI. SAVI can be more robust in areas with varying soil brightness.
*   **Bare Soil Proxies:** Since NAIP lacks SWIR, proxies for bare soil indices will be explored. This might involve combinations of RGB and NIR bands that highlight exposed soil, or a classification based on low NDVI and high brightness in visible bands.
*   **Texture Features (Gray-Level Co-occurrence Matrix - GLCM):** Extract GLCM features (e.g., Homogeneity, Contrast, Dissimilarity, Entropy, Angular Second Moment) from the NIR band or a principal component of the NAIP bands. These features capture the spatial arrangement of pixel intensities, which can distinguish between uniform surfaces (e.g., lawns, pavement) and heterogeneous surfaces (e.g., weeds, patchy bare soil, debris) often found in vacant lots.

### 3.2. Features from NYC MapPluto Data

*   **Geometric Features:**
    *   **Area:** The total area of the parcel (e.g., in square meters or acres).
    *   **Perimeter:** The length of the parcel boundary.
    *   **Shape Index/Compactness:** A measure derived from area and perimeter (e.g., `Perimeter / (2 * sqrt(Area * pi))`) to quantify how irregular the parcel shape is. Irregular shapes might be less desirable for development.
*   **Land Use/Zoning Information:** Categorical features indicating current land use or zoning (e.g., residential, commercial, manufacturing, open space). These can be one-hot encoded or used to filter parcels.
*   **Building Characteristics:** Attributes like `BBL` (Borough, Block, Lot), `BuiltFAR` (Floor Area Ratio), `NumBldgs` (Number of Buildings), `AssessTot` (Total Assessed Value). Parcels with `NumBldgs = 0` or very low `BuiltFAR` are strong candidates for vacant lots.
*   **Proximity Features (Derived):** Distance to key urban features such as major roads, public transportation hubs, parks, and existing developed areas. These can be calculated using GIS tools and influence development potential.

## 4. Clustering Methodology

The clustering process will aim to group parcels with similar characteristics, with the expectation that vacant lots will form one or more distinct clusters.

### 4.1. Data Preparation

*   **Feature Aggregation:** For each MapPluto parcel, aggregate the extracted NAIP features (mean, std dev, etc.) to a single value per parcel. This creates a feature vector for each parcel.
*   **Feature Scaling:** Standardize or normalize all numerical features to ensure that features with larger ranges do not dominate the clustering process.
*   **Dimensionality Reduction (Optional):** If the number of features is very high, techniques like Principal Component Analysis (PCA) might be considered to reduce dimensionality and improve computational efficiency, while retaining most of the variance.

### 4.2. Clustering Algorithm Selection

Given the large number of parcels (800k), **K-Means** is a suitable choice due to its efficiency and scalability. **ISODATA** could be an alternative if the number of clusters is unknown and needs to be determined adaptively. For initial exploration, K-Means will be prioritized.

### 4.3. Clustering Steps

1.  **Determine Optimal Number of Clusters (k):** Use methods like the Elbow Method or Silhouette Score on a subset of the data to estimate an appropriate `k` value for K-Means. Start with a reasonable range (e.g., 3-10 clusters) based on expected land cover types.
2.  **Apply K-Means Clustering:** Run the K-Means algorithm on the prepared feature vectors of the parcels. Each parcel will be assigned a cluster label.
3.  **Cluster Interpretation:** Analyze the characteristics of each cluster by examining the mean feature values within each cluster. Identify clusters that exhibit characteristics consistent with vacant lots (e.g., low NDVI, high bare soil proxy, low building count, specific texture patterns).
4.  **Spatial Visualization:** Map the clustered parcels back onto the NYC geography to visually inspect the spatial distribution of clusters and validate their interpretation against visual inspection of NAIP imagery.

## 5. Handling Large Datasets (800k parcels) and GEE

*   **Subset for Development:** For preliminary EDA and code development, a geographically constrained subset of NYC (e.g., a single borough or a few representative neighborhoods) will be used. This allows for faster iteration and resource management.
*   **Google Earth Engine (GEE) for Feature Extraction:** GEE is highly efficient for large-scale remote sensing data processing. While clustering itself might be performed outside GEE (due to its limitations on complex unsupervised learning or direct integration with MapPluto polygons), GEE is ideal for:
    *   Accessing and pre-processing NAIP imagery.
    *   Calculating spectral indices (NDVI, SAVI).
    *   Extracting zonal statistics (mean, std dev) of these features within MapPluto parcel polygons. This can be done by uploading MapPluto polygons as `ee.FeatureCollection`.
*   **Local Processing for Clustering:** Once features are extracted from GEE and downloaded (or processed in batches), the clustering algorithms (e.g., K-Means from `scikit-learn` in Python) can be applied locally on the aggregated feature vectors. This approach leverages GEE's strengths for raster processing and local environments for flexible machine learning.

## 6. Preliminary Code Structure

The preliminary code will focus on demonstrating feature extraction (conceptually for GEE, practically for local data if a small sample is available) and a basic K-Means clustering workflow.

1.  **Data Loading:** Load MapPluto data (e.g., GeoJSON or Shapefile) and conceptually define NAIP data access.
2.  **Feature Calculation:** Placeholder for GEE-based feature extraction, and local calculation of geometric features.
3.  **Data Preparation:** Feature scaling.
4.  **Clustering:** K-Means implementation.
5.  **Visualization:** Basic plotting of clusters.

This methodology provides a structured approach to identifying vacant lots, combining the strengths of high-resolution remote sensing data with comprehensive urban parcel information.

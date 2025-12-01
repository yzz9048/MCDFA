# CDFA

# For now, only the core code is displayed and the complete code is coming soon.

We have provided the backbone of CDFA (backbone.py) and the diffusion process (gaussian_diffusion.py) here.

# Metric prediction results

We have shown the metric prediction resuts of all methods on all datasets, and they are named by "{model_name}-{dataset_name}" primarily. An overall display is provided in the PDF file.

# Datasets

For AIops18, please refer to AIOps18.csv in Folder /Dataset or https://github.com/BEbillionaireUSD/Maat

For GAIA, please refer to to GAIA.csv in Folder /Dataset or https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main/Companion_Data

SoN and Med are coming soon, and we present a toy dataset (the illustration expriment in the motivation section of the paper), similar to SoN and Med, except that the data volume is smaller: https://figshare.com/s/92cef85e44f2223d2112

Specifically, the json files in the data warehouse are the raw data collected by tools, i.e.., Prometheus and k6, and the csv files are extracted from json files. Metrics are collected at intervals of 10 seconds. Therefore, RPS data can perform average aggregation at intervals of 10 seconds as guiding condition data. We have provided the specific number of requests per second. Readers can handle and generate other conditional data by themselves.

This toy dataset is collected from microservice benchmark SocialNetwork, which comprises 13 microservices and 13 database microservices, each microservice has four instances deployed on two worker servers, each equipped with Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz, 4* 8G DDR4 Registered (Buffered) 2133 MHz, 2T HHD. A master server (the same equipment as work server) generates workloads using k6, collects monitoring data via Prometheus and injects failures via ChaosBlade. We have also provided a full view of the dataset and since this dataset is illustrated for motivation, we do not label the failures in the .csv files. Moreover, we have provided a partial view to exhibit the lag between RPS and metrics more clearly, and readers can examine it from the metrics-withRPS.csv file.

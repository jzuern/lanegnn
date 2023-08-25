# Dataset processing for model training

In this document, we describe how to process the UrbanLaneGraph dataset for model training.

## 0. Download the dataset

For instructions on how to download the dataset, please refer to [urbanlanegraph_dataset/DOWNLOAD.md](urbanlanegraph_dataset/DOWNLOAD.md).






## 1. Generate raw samples (for regressor model traning)

```bash
python generate_raw_samples.py /path/to/urbanlanegraph/dataset/ /path/to/raw/output <city_name> <split>
```

The parameter `<city_name>` can be either of `miami`, `paloalto`, `pittsburgh`, `austin`, `washington`, `detroit`.
The parameter `<split>` can be either of `train`, `val`.

We do not provide the samples for the dataset `test` split.

Make sure that the directory `lanegnn/dataset_preparation` is in your `PYTHONPATH` environment variable. (To do so, run `export PYTHONPATH=$PYTHONPATH:/path/to/lanegnn/dataset_preparation`)


## 2. Generate pth samples (for LaneGNN model training)



```bash
python generate_pth_samples.py --config ../methods/lanegnn/config/config.yaml --raw_dataset /path/to/raw/output/ --processed_dataset /path/to/processed/output/ --ego_regressor_ckpt /path/to/checkpoint.pth --context_regressor_ckpt /path/to/checkpoint.pth
```


Note that you have to run the script for all the directories you might have created in step 1.


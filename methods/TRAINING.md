# LaneGNN Training Guide


In order to train the models described in the paper, we need to first download and pre-process the dataset. 
To do so, please follow the instructions in [urbanlanegraph_dataset/DOWNLOAD.md](urbanlanegraph_dataset/DOWNLOAD.md) and
[urbanlanegraph_dataset/PROCESS_DATA.md](urbanlanegraph_dataset/PROCESS_DATA.md).



## 1 Training of Regressor models

We leverage two centerline regression modules in our paper:

- lane centerline regression (regressing all visible lane centerlines)
- ego lane centerline regression (regressing the centerline of the ego-agent lane)


The ego lane centerline regression model gets as input both the RGB images and the lane centerline regression model output.



### 1.1 Training of lane centerline regression model

```bash
python train_centerline_regression.py --dataset-root /path/to/raw/cropped/dataset --sdf_version centerlines-sdf
```


### 1.2 Training of *ego* lane centerline regression model


```bash
python train_centerline_regression.py --dataset-root /path/to/raw/cropped/dataset --sdf_version centerlines-sdf-ego-context --checkpoint_path_context_regression /path/to/context_regression_model.pth 
```


## Training of LaneGNN model (with pre-trained lane regressors)

```bash
python train_lanegnn.py --config lanegnn/config/config.yaml --train_path /path/to/processsed/cropped/dataset/train --eval_path /path/to/processsed/cropped/dataset/eval
```

The config file contains the following parameters:

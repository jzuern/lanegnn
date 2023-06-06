
# UrbanLaneGraph Dataset Download Guide

This document describes how to download and visualize the UrbanLaneGraph dataset.



<p align="center">
  <img src="img/banner.gif" />
</p>


## 0 Pre-requisites

In the following, the following environment variables are used. Please adjust the values to your local setup.


```bash
export DATASET_ROOT=/path/to/urbanlanegraph/dataset
```



## 1 Download the UrbanLaneGraph dataset

### 1.1 Request dataset access

Please request dataset access at http://urbanlanegraph.cs.uni-freiburg.de/download.
Upon acceptance of your request, we will send you an email containing the download link.

### 1.2 Download and unpack

Please replace the link below with the link sent to you via Email.

```bash
export DATASET_LINK=http://link/to/dataset/urbanlanegraph-dataset.zip
```

To download and extract, please execute:

```bash
cd $DATASET_ROOT
wget $DATASET_LINK
unzip urbanlanegraph-dataset.zip
rm urbanlanegraph-dataset.zip
```

You should see the following output for `tree $DATASET_ROOT/*`:


```bash
urbanlanegraph-dataset
├── austin
│   ├── austin_direction.png          # color-coded lane direction
│   ├── austin_drivable.png           # drivable surfaces
│   ├── austin_centerlines.png        # heatmap of centerlines
│   ├── austin_intersections.png      # binary map of intersection regions
│   ├── austin.png                    # RGB aerial image
│   └── tiles
│       ├── eval
│           ├── *.gpickle  # graph annotation files of eval tiles.
│           ├── ...
│       └── train
│           ├── *.gpickle  # graph annotation files of train tiles.
│           ├── ...
├── detroit
│   ├── detroit_direction.png
│   ├── detroit_drivable.png
│   ├── detroit_intersections.png
│   ├── detroit.png
│   └── tiles
│       ├── eval
│           ├── *.gpickle  # graph annotation files of eval tiles.
│           ├── ...
│       └── train
│           ├── *.gpickle  # graph annotation files of train tiles.
│           ├── ...
├── miami
│   ├── miami_direction.png
│   ├── miami_drivable.png
│   ├── miami_centerlines.png  
│   ├── miami_intersections.png
│   ├── miami.png
│   └── tiles
│       ├── eval
│           ├── *.gpickle  # graph annotation files of eval tiles.
│           ├── ...
│       └── train
│           ├── *.gpickle  # graph annotation files of train tiles.
│           ├── ...
├── paloalto
│   ├── paloalto_direction.png
│   ├── paloalto_drivable.png
│   ├── paloalto_centerlines.png
│   ├── paloalto_intersections.png
│   ├── paloalto.png
│   └── tiles
│       ├── eval
│           ├── *.gpickle  # graph annotation files of eval tiles.
│           ├── ...
│       └── train
│           ├── *.gpickle  # graph annotation files of train tiles.
│           ├── ...
├── pittsburgh
│   ├── pittsburgh_direction.png
│   ├── pittsburgh_drivable.png
│   ├── pittsburgh_centerlines.png
│   ├── pittsburgh_intersections.png
│   ├── pittsburgh.png
│   └── tiles
│       ├── eval
│           ├── *.gpickle  # graph annotation files of eval tiles.
│           ├── ...
│       └── train
│           ├── *.gpickle  # graph annotation files of train tiles.
│           ├── ...
└── washington
    ├── washington_direction.png
    ├── washington_drivable.png
    ├── washington_centerlines.png
    ├── washington_intersections.png
    └── washington.png
    └── tiles
        ├── eval
            ├── *.gpickle  # graph annotation files of eval tiles.
            ├── ...
        └── train
            ├── *.gpickle  # graph annotation files of train tiles.
            ├── ...

```

As can be seen, we omit the tile annotations for the test splits of all cities. These are used for the model evaluation
in the Eval.ai benchmark leaderboard and are not available to the public.



### 1.3 Explore the images

Feel free to explore the dataset by opening the images in your favorite image viewer.

Due to the large image sizes, we recommend using GIMP. Please also note that substantial amount of
memory is needed to open the images (>= 32GB RAM).

In case you cannot open the images due to memory limitations, please consider cutting them into tiles, corresponding to the graph annotation tiles.





### 2 Visualizing the dataset


For visualization, we provide a script that overlays the graph annotation files on top of the aerial image.

You may create a new conda environment and install the required dependencies:

```bash
conda env create python=3.10 --name urbanlanegraph --file environment.yml
conda activate urbanlanegraph
```


```bash
python visualize_dataset.py --dataset_root $DATASET_ROOT --city austin
```
This script produces an overlay of the annotation graph files on top of the aerial image for each city (in this case austin).

If we add the `--plot-single-tiles` flag, the script will also plot the individual tiles of the graph annotation files.




### 3 Metadata API and Helper functions and additional information

As part of this dataset we provide a small set of helper functions for typical use cases in the form of a API. 
Feel free to import the ```UrbanLaneGraphMetadata``` class under ```urbanlanegraph_dataset/api.py```.

You may utilize the API to get tile-specific offsets wrt. the global aerial image, obtain dataset splits, tile images or tile GT graphs.
Additionally, this includes functions for aligning the argoverse2 graph annotation coordinate frame with the pixel-coordinates of the aerial images of the UrbanLaneGraph dataset. 




### Congratulations!

If you made it this far, you may already use the dataset for your own purposes.




# License
We offer the UrbanLaneGraph dataset free of charge under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (“CC BY-NC-SA 4.0”).

By downloading the data, you accept the license agreement which is available at http://urbanlanegraph.cs.uni-freiburg.de/assets/license.txt.


The UrbanLaneGraph dataset is based on the Argoverse2 dataset (https://www.argoverse.org/av2.html). We thank the Argoverse2 team for making the Argoverse2 datset publicly available and for allowing the re-distribution of their dataset in remixed form.


The UrbanLaneGraph Dataset scripts are released under the MIT License.



# Contact
Please feel free to contact us with any questions, suggestions or comments:

- Jannik Zürn: zuern (at) cs.uni-freiburg.de
- Martin Büchner: buechner (at) cs.uni-freiburg.de

Copyright **Autonomous Intelligent Systems, Robot Learning Lab** | University of Freiburg, Germany.









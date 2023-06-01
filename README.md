# SAGDFN
 This is the PyTorch official implementation of SAGDFN: A Scalable Adaptive Graph Diffusion Forecasting Network for Multivariate Time Series Forecasting
# Requirements
```bash
pip install -r requirements.txt
```
## Data Preparation

The traffic data files for Los Angeles (METR-LA)should be put into the `data/` folder. It is provided by [DCRNN](https://github.com/chnsh/DCRNN_PyTorch).

The Carpark1918 dataset can be downlaod from the [GoogleDrive](https://drive.google.com/drive/folders/1oVOGL-gFR2osHjStaw5_Ma8YTchgw0r3?usp=sharing).


Run the following commands to generate train/test/val dataset at  `data/{METR-LA,Carpark}/{train,val,test}.npz`.
```bash
# Unzip the datasets
unzip data/metr-la.h5.zip -d data/

# Create data directories
mkdir -p data/{METR-LA,Carpark}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# Carpark1918
python -m scripts.generate_training_data --output_dir=data/Carpark --traffic_df_filename=data/carpark_05_06.h5
```

## Train Model

When you train the model, you can run:

```bash
# Use METR-LA dataset
python train.py --config_filename=data/model/para_la.yaml

# Use Carpark1918 dataset
python train.py --config_filename=data/model/para_carpark.yaml
```

Hyperparameters can be modified in the `para_la.yaml` and `para_carpark.yaml` files.

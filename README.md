# m5-forecasting
Using Pytorch's LSTM for the m5 forecasting competition (time series forecasting)

# Training
To install the prerequisites into a conda environment, run
``` 
conda env create -f environment.yml
```
Add a folder called "m5" and "out" under the data directory, and extract the contents of the m5 datasets to the m5 folder ([available here](https://www.kaggle.com/c/m5-forecasting-accuracy)). To train, run prepare_m5_dataset then train.py. 

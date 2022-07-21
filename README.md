# AQI_Predict

A long-term AQI Prediction method using LSTM network

代码文件为`AQI_Predict.ipynb`以及`AQI_Predict.py`，二者实现的功能完全相同，使用任意一个即可。其中.ipynb文件需要jupyter notebook打开。

数据文件为`data_clean.xlsx`，为2015-2020年的成都市温江区原始气象数据经过处理得到，其中包含一些异常数据，已在代码中进行筛选剔除了。

`my_model.h5`文件为训练好的预测模型，可以直接进行数据预测。

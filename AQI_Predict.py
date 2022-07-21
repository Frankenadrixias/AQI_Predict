import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error

# 全局变量
filter_window_size = 10
col_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


# 清理异常数据
def drop_outliers(dataframe: pd.DataFrame):
    # 对不同的数据列设置阈值范围，筛选合理的数据
    dataframe_drop = dataframe[(dataframe['AQI'] > -0.5) &
                               (dataframe['PM2.5'] > -0.5) &
                               (dataframe['PM10'] > -0.5) &
                               (dataframe['wind'] < 1000) &
                               (dataframe['precipitation'] < 1000) &
                               (dataframe['temperature'] < 1000) &
                               (dataframe['plbh'] > 0)]

    return dataframe_drop


# 定义数据滤波函数：将数据减去附近一段时间（window_size）的均值
# 去均值化处理：保留下来的是数据的高频部分
def data_filter(dataframe: pd.DataFrame, col_index: int, window_size: int):
    # 分别保留原始数据与处理过的数据
    original_array = np.array(dataframe.iloc[:, col_index].values)
    processed_array = np.array(dataframe.iloc[:, col_index].values)

    for i in range(len(original_array)):

        # 存储数据值总和与数据数目
        local_sum, local_count = 0, 0

        for j in range(max(0, i - window_size), min(len(original_array), i + window_size)):
            local_sum += original_array[j]
            local_count += 1

        # 计算平均值，并将原始数据减去均值
        mean_value = local_sum / local_count
        processed_array[i] = original_array[i] - mean_value

    return processed_array


# 去均值化处理：保留下来的是数据的高频部分
def data_inverse_filter(dataframe: pd.DataFrame, inverse_yhat: np.array, inverse_y: np.array,
                        col_name: str, split_value: float):

    # 将最终得到的数据进行反滤波处理，还原原始数据
    start = int(dataframe.shape[0] * split_value)
    original_array = np.array(dataframe[col_name].values)

    for i in range(start, len(dataset) - 1):

        # 存储数据值总和与数据数目
        local_sum, local_count = 0, 0

        for j in range((i - filter_window_size), min(len(dataset), i + filter_window_size)):
            local_sum += original_array[j]
            local_count += 1

        # 计算平均值，并将原始数据加上均值
        mean_value = local_sum / local_count
        inverse_yhat[i - start, -1] += mean_value
        inverse_y[i - start, -1] += mean_value

    return inverse_yhat, inverse_y


if __name__ == '__main__':

    # 读取xlsx文件数据到dataframe
    df = pd.read_excel('data_clean.xlsx')
    print('原始数据总量：', len(df))

    df_drop = drop_outliers(df)
    print('数据清理后数据量：\n', len(df_drop))

    df_filter = pd.DataFrame()
    for col_i in col_list:
        df_filter.insert(df_filter.shape[1], list(df.columns)[col_i],
                         data_filter(df_drop, col_i, filter_window_size))

    print('数据预处理后的数据集：', df_filter)

    # 筛选所需数据
    features = ['plbh', 'air_pres', 'temperature', 'precipitation', 'wind',
                '700hPa_temp', '700hPa_humi', '700hPa_wind', 'AQI']
    dataset = df_filter[features]

    # 划分训练集、测试集
    split = 0.7
    train = dataset.loc[:int(len(dataset) * split)]
    test = dataset.loc[int(len(dataset) * split):]

    # 展示数据
    plt.figure(figsize=(16, 8))
    for i in range(len(features)):
        plt.plot(train[features[i]], label=features[i] + '_train')
        plt.plot(test[features[i]], label=features[i] + '_test')
    plt.legend(loc='upper left')
    plt.show()

    # 将数据归一化、标准化并转化为特征向量
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train.values)
    test_sc = scaler.transform(test.values)

    # 选择第k天的气象数据和第k天的空气质量数据预测第k+1天的AQI
    X_train = train_sc[:-1, :]
    y_train = train_sc[1:, -1]
    X_test = test_sc[:-1, :]
    y_test = test_sc[1:, -1]

    # 对数组进行维度转换
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # 设计lstm网络模型
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]),
                        activation='elu',
                        kernel_initializer='lecun_uniform',
                        return_sequences=True,
                        dropout=0.2,
                        recurrent_dropout=0.2))
    lstm_model.add(LSTM(64, activation='elu',
                        return_sequences=False,
                        dropout=0.2))
    lstm_model.add(Dense(32))
    lstm_model.add(Dense(16))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.summary()

    # 设置早停法防止过拟合
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

    # 回调函数记录训练过程（损失和准确率等）
    history_lstm_model = lstm_model.fit(X_train, y_train,
                                        epochs=30, batch_size=30,
                                        verbose=1, shuffle=True,
                                        callbacks=[early_stop],
                                        validation_split=0.2)

    # 打印loss曲线
    loss = history_lstm_model.history['loss']
    val_loss = history_lstm_model.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize = (8, 6))
    plt.plot(epochs, loss, 'bo-', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label = 'Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 保存或读取 HDF5 文件'my_model.h5'
    # lstm_model = load_model('my_model.h5')
    lstm_model.save('my_model.h5')

    # 使用第 k 日 AQI 数据和 k+1 日气象数据迭代预测 k+1 日 AQI 数据
    X_test_new = X_test.copy()
    X_test_new[1, -1] = lstm_model.predict(X_test_new[[0]]).flatten()
    for i in range(len(X_test_new) - 1):
        X_test_new[:i + 1, -1] = lstm_model.predict(X_test_new[:i + 1])

    y_test_predict_lstm = lstm_model.predict(X_test_new)
    y_train_predict_lstm = lstm_model.predict(X_train)

    # 输出预测结果的各种评分
    print("\nThe R2 score on the Train set is:\t {:0.4f}"
          .format(r2_score(y_train, y_train_predict_lstm)))
    print("The R2 score on the Test set is:\t {:0.4f}"
          .format(r2_score(y_test, y_test_predict_lstm)))

    print("\nThe Mean Absolute Error(MAE) on the Train set is:\t {:0.4f}"
          .format(mean_absolute_error(y_train, y_train_predict_lstm)))
    print("The Mean Absolute Error(MAE) on the Test set is:\t {:0.4f}"
          .format(mean_absolute_error(y_test, y_test_predict_lstm)))

    # 对数据进行逆标准化和逆归一化
    X_test_re = X_test_new.reshape((X_test.shape[0], X_test.shape[1]))
    inv_yhat = np.concatenate((X_test_re[:, :-1], y_test_predict_lstm), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat

    y_test_re = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((X_test_re[:, :-1], y_test_re), axis=1)
    inv_y = scaler.inverse_transform(inv_y)

    inv_yhat, inv_y = data_inverse_filter(df_drop, inv_yhat, inv_y, 'AQI', split)

    # 展示最终的预测结果与观测值的比较
    plt.figure(figsize=(16, 6))
    plt.plot(inv_y[:60, -1], 'ro-', label='AQI Observation Value')
    plt.plot(inv_yhat[:60, -1], 'g^-', label='AQI LSTM Predict Value')
    plt.title("LSTM's Prediction")
    plt.xlabel('Days')
    plt.ylabel('AQI')
    plt.legend()
    plt.show()

    # 计算平均相对误差（MRE）与均方相对误差（MSRE）
    time = 30
    mre, msre = 0, 0

    for i in range(time):
        mre += abs((inv_yhat[i,-1] - inv_y[i,-1])/inv_y[i,-1])
        msre += ((inv_yhat[i,-1] - inv_y[i,-1])/inv_y[i,-1]) ** 2

    msre = (msre / (time + 1)) ** 0.5
    mre = mre / (time + 1)

    print("The Mean Squared Relative Error of the prediction is:\t {:0.4f}".format(mre))
    print("\nThe Mean Relative Error of the prediction is:\t {:0.4f}".format(msre))
    print("\nThe overall accuracy of the prediction is:\t {:0.3f}%".format((1-mre) * 100))

    # 打印 MRE 与 MSRE 的变化趋势
    time = 30
    mre_arr, msre_arr = [], []
    for t in range(time):
        mre, msre = 0, 0
        for i in range(t):
            mre += abs((inv_yhat[i,-1] - inv_y[i,-1])/inv_y[i,-1])
            msre += ((inv_yhat[i,-1] - inv_y[i,-1])/inv_y[i,-1]) ** 2
        mre_arr.append(1 - mre / (t + 1))
        msre_arr.append(1 -(msre / (t + 1)) ** 0.5)

    plt.figure(figsize = (10, 6))
    plt.plot(msre_arr[:], 'bo-', label = 'msre Value')
    plt.plot(mre_arr[:], 'r^-', label = 'mre Value')
    plt.title("LSTM's Prediction")
    plt.xlabel('Days')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

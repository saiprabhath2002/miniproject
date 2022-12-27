from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request


def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)

n1features=[]
n2features=[]
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)
"""

#importing the dataset


"""


# print('predicted output :=', regr.predict(X[:10]))
# print('Actual output :=', y[:10])

# print(features)

# data[features].head(5)

"""#User input"""

# ip = [0, 288.28, 2, 9, 2, 2012, 10, 2, 8]
# ip = x_scaler.transform([ip])
# out = regr.predict(ip)
# print('Before inverse Scaling :', out)

# y_pred = y_scaler.inverse_transform([out])
# print('Traffic Volume : ', y_pred)

# if(y_pred <= 1000):
#     print("No Traffic ")
# elif y_pred > 1000 and y_pred <= 3000:
#     print("Busy or Normal Traffic")
# elif y_pred > 3000 and y_pred <= 5500:
#     print("heavy Traffic")
# else:
#     print("Worst case")

"""#Evaluating Metrics """


# def relative_absolute_error(true, pred):
#     true_mean = np.mean(true)
#     squared_error_num = np.sum(np.abs(true - pred))
#     squared_error_den = np.sum(np.abs(true - true_mean))
#     rae_loss = squared_error_num / squared_error_den
#     return rae_loss


# def mean_bias_error(true, pred):
#     bias_error = true - pred
#     mbe_loss = np.mean(np.sum(diff) / true.size)
#     return mbe_loss


# def mean_absolute_percentage_error(true, pred):
#     abs_error = (np.abs(true - pred)) / true
#     sum_abs_error = np.sum(abs_error)
#     mape_loss = (sum_abs_error / true.size) * 100
#     return mape_loss


# def mean_squared_error(true, pred):
#     squared_error = np.square(true - pred)
#     sum_squared_error = np.sum(squared_error)
#     mse_loss = sum_squared_error / true.size
#     return mse_loss


# def root_mean_squared_error(true, pred):
#     squared_error = np.square(true - pred)
#     sum_squared_error = np.sum(squared_error)
#     rmse_loss = np.sqrt(sum_squared_error / true.size)
#     return rmse_loss


#####################################################frontend#############################
app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/train')
def train():
    data = pd.read_csv('static/Train.csv')
    data = data.sort_values(
        by=['date_time'], ascending=True).reset_index(drop=True)
    last_n_hours = [1, 2, 3, 4, 5, 6]
    for n in last_n_hours:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
    data = data.dropna().reset_index(drop=True)
    data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
    data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
    data['is_holiday'] = data['is_holiday'].astype(int)

    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
    data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
    data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
    data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
    data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
    data.to_csv("traffic_volume_data.csv", index=None)

    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')
    data = pd.read_csv("traffic_volume_data.csv")
    data = data.sample(10000).reset_index(drop=True)
    label_columns = ['weather_type', 'weather_description']
    numeric_columns = ['is_holiday', 'temperature',
                       'weekday', 'hour', 'month_day', 'year', 'month']
    n1 = data['weather_type']
    n2 = data['weather_description']
    unique(n1)
    unique(n2)
    n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist',
                  'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
    n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain',
                  'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']
    """#Data Preparation"""
    n11 = []
    n22 = []
    for i in range(10000):
        if(n1[i]) not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1[i]))+1)
        if n2[i] not in n2features:
            n22.append(0)
        else:
            n22.append((n2features.index(n2[i]))+1)
    # print(n11)
    # print(n22)
    data['weather_type'] = n11
    data['weather_description'] = n22
    features = numeric_columns+label_columns
    target = ['traffic_volume']
    X = data[features]
    y = data[target]
    print(X)
    print(data[features].hist(bins=20,))

    data['traffic_volume'].hist(bins=20)

    """#Feature Scaling"""

    
    X = x_scaler.fit_transform(X)
    
    y = y_scaler.fit_transform(y).flatten()
    warnings.filterwarnings('ignore')


    """#Visualization"""

    # metrics = ['month', 'month_day', 'weekday', 'hour']
    # fig = plt.figure(figsize=(8, 4*len(metrics)))
    # for i, metric in enumerate(metrics):
    #     ax = fig.add_subplot(len(metrics), 1, i+1)
    #     ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
    #     ax.set_xlabel(metric)
    #     ax.set_ylabel("Mean Traffic")
    #     ax.set_title(f"Traffic Trend by {metric}")
    # plt.tight_layout()
    # # plt.savefig('plot1.png')
    # plt.show()

    """#Train the Model"""
    regr.fit(X, y)
    print('predicted output :=', regr.predict(X[:10]))
    print('Actual output :=', y[:10])
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ip = []
    if(request.form['isholiday'] == 'yes'):
        ip.append(1)
    else:
        ip.append(0)
    ip.append(int(request.form['temperature']))
    ip.append(int(request.form['day']))
    ip.append(int(request.form['time'][:2]))
    D = request.form['date']
    ip.append(int(D[8:]))
    ip.append(int(D[:4]))
    ip.append(int(D[5:7]))
    s1 = request.form.get('x0')
    s2 = request.form.get('x1')
    if(s1) not in n1features:
        ip.append(0)
    else:
        ip.append((n1features.index(s1))+1)
    if s2 not in n2features:
        ip.append(0)
    else:
        ip.append((n2features.index(s2))+1)
    ip = x_scaler.transform([ip])
    out = regr.predict(ip)
    print('Before inverse Scaling :', out)
    y_pred = y_scaler.inverse_transform([out])
    print('Traffic Volume : ', y_pred)
    s = ''
    if(y_pred <= 1000):
        print("No Traffic ")
        s = "No Traffic "
    elif y_pred > 1000 and y_pred <= 3000:
        print("Busy or Normal Traffic")
        s = "Busy or Normal Traffic"
    elif y_pred > 3000 and y_pred <= 5500:
        print("heavy Traffic")
        s = "heavy Traffic"
    else:
        print("Worst case")
        s = "Worst case"
    return render_template('output.html', data1=ip, op=y_pred, statement=s)


if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import pandas as pd
import sqlalchemy as db
import time
import telnetlib
import tensorflow as tf
from tensorflow import keras
# import tensorboard

# pymysql is needed for connection

### Make changes here!
### Additional changes may be necessary at the end of the document.

# List of values with identical meanings to make analysing easier.
identical_list = [['set_on', 'on'], ['set_off', 'off'], ['set_toggle','off']]

# List of controllable devices with binary-style input. On/Off - Values like dim 50 are possible, but more values make it increasingly harder to predict.
# Add Values like ['DEVICE', 'READING'] and remove all values that are not contained in your System
sensor_bin_list = [['ZWave_SENSOR_NOTIFICATION_7', 'state'], ['ZWave_SENSOR_NOTIFICATION_11', 'state'], ['ZWave_SENSOR_NOTIFICATION_9', 'state']]
# List of binary-style sensors. e.g. motion sensor. Readings that contain floating values like luminance are not supported!
device_bin_list = [['ZWave_SWITCH_BINARY_5.01', 'state'], ['ZWave_SWITCH_BINARY_5.02', 'state'], ['Fabian.Relais' ,'state'], ['Kueche.LED','state'], ['Kueche.Relais','state']]
# List of static sensors with floating values, do only enter numeric sensors!
static_list = [['ZWave_SENSOR_NOTIFICATION_7', 'luminance'], ['ZWave_SENSOR_NOTIFICATION_9','luminance'],
               ['ZWave_SENSOR_NOTIFICATION_11','luminance'], ['Pflanze1', 'lux'], ['Pflanze1', 'temperature'],
               ['Pflanze2', 'lux'], ['Pflanze2', 'temperature'], ['Pflanze3', 'lux'], ['Pflanze3', 'temperature'],
               ['Pflanze4', 'lux'], ['Pflanze4', 'temperature'],['CUL_WS_1', 'humidity'], ['CUL_WS_1', 'temperature']]

# IP of your mysql-Server/fhem-instance
IP = '192.168.37.33'
# User and password of your mysql-Server preferably with read-only access
USER = 'fhemuser'
PSWD = 'fhempassword'
# telnetport, password currently not supported.
telnetport = 7072


class getData:

    def __init__(self, ip, user, pswd):
        self.myIP = ip
        self.myUser = user
        self.myPSWD = pswd

    def remoteConnect(self, list, table='history'):
        start_time = time.time()
        engine = db.create_engine(f'mysql+pymysql://{USER}:{PSWD}@{IP}:3306/fhem')
        queryString = f'SELECT TIMESTAMP, DEVICE, VALUE, READING FROM {table} WHERE DEVICE IN {str(tuple([row[0] for row in list]))} ORDER BY TIMESTAMP DESC'
        data = pd.read_sql_query(queryString, engine, index_col='TIMESTAMP')
        engine.dispose()  # Close Connection?!

        elements = [tuple(element) for element in list]
        data = data[data[['DEVICE', 'READING']].apply(tuple, axis=1).isin(elements)]  # Löschen der nicht genutzten Readings

        print(f'Executed remote connect in {(time.time() - start_time):.3f} s')

        self.df = data
        self.myList = list

    def sort_df(self):
        data = self.df
        data['join'] = data['DEVICE'] + '.' + data['READING']

        for idx, element in enumerate(self.myList):
            data['join'] = data['join'].replace(f'{element[0]}.{element[1]}', idx)

        data = data.sort_values(['TIMESTAMP', 'join'])
        data.drop(columns='join', inplace=True)

        self.df = data

    def order_df(self, identical_list):   #Irreführender Name?
        data = self.df
        data = data.pivot_table(index='TIMESTAMP', columns=['DEVICE', 'READING'],
                                values='VALUE', dropna=True, aggfunc='first')
        data.columns = ['.-.'.join(col).strip() for col in data.columns]  # Kombiniert zu single column df
        data.dropna(how='all', axis=1, inplace=True)

        for identical in identical_list:
            data = data.replace(identical[0], identical[1])

        # data.ffill(inplace=True)
        # data.bfill(inplace=True)

        self.df = data


    def printdf(self):
        print([self.df])
        self.df.to_csv(f'df/{self.myList[0][0]}-{self.myList[0][1]}.csv', sep=';', decimal=',')

class prepareData:
    def __init__(self, df):
        self.df = df

    def printdf(self):
        print(self.df)
        self.df.to_csv(f'df/{time.time()}.csv', sep=';', decimal=',')

    def to_one_hot(self):
        data = self.df
        data.ffill(inplace=True)
        for col in list(data):
            for val in data[col].dropna().unique():  # Alle Werte, welche das device annehmen kann.
                data[f'{col}.-.{val}'] = (data[col] == val) * 1  # Zuordnen der Werte wenn der Wert == val, *1 um True --> 1
            data.drop(columns=col, inplace=True)  # Alte Zeile wird gelöscht, da nicht mehr benötigt, nachdem alle überführt.
        self.df = data
        self.tok_lst = data.columns.values.tolist()

    def combine_static(self, static_df):  #Run after to_one_hot!
        df = pd.concat([self.df, static_df], sort=False)
        df.sort_index(inplace=True)
        # df.iloc[:, -len(static_df.columns):] = df.ffill()         # Every added Column gets ffilled, every old column stays ignored.
        df_num = df.apply(pd.to_numeric)
        df.iloc[:, -len(static_df.columns):] = df_num.where(df_num.notnull(), other=(df_num.fillna(method='ffill') + df_num.fillna(method='bfill'))/2)  # Generating mean values
        df.iloc[:, -len(static_df.columns):] = df.ffill() # Add Values after the last value to be up-to-date
        df.dropna(inplace=True) # Remove all rows, that contain no new information (removes rows where just static data was added)
        self.df = df

    def combine_current(self, current_df):
        df = pd.concat([self.df, current_df], sort=False)
        df.sort_index(inplace=True)
        df = df.ffill().bfill().dropna()
        self.df = df


    def normalize(self):
        self.df = self.df.ffill().bfill().dropna()
        self.df = self.df.apply(pd.to_numeric)
        self.df = (self.df - self.df.mean(axis=0)) / self.df.std(axis=0)   #Z-Transformation/Normalisierung

class rnn_analyse:
    def __init__(self, df, size):
        self.df = np.array(df)
        self.size = size

    def to_intervals(self, length, y_length):
        self.myLength = length
        self.out_shape = y_length
        data = self.df

        x_data = []
        y_data = []
        self.y_raw_data = []

        y_val = np.zeros(len(data[-1]))

        for idx in range(length, len(data) + 1):
            x_val = []
            for jdx in range(length, 1, -1):  # Rückwärts, bsp i = 10, length = 3 --> 7,8,9 -> y = 10
                x_val.append(data[idx - jdx])
            x_data.append(x_val)
            self.y_raw_data.append(data[idx-1])
            y = data[idx - 1] - y_val  # Nur die Änderungen sind von Interesse
            y[y < 0] = 0  # Alle negativen Werte werden zu null, da nicht interessant.
            y_data.append(y)

            y_val = data[idx - 1]

        self.x_data = np.array(x_data)
        # self.y_data = np.array(y_data)[0, 0:y_length]
        self.y_data = np.array(y_data)

    def shuffle(self):
        assert len(self.x_data) == len(self.y_data)
        perm = np.random.permutation(len(self.x_data))
        self.x_shuffled = self.x_data[perm]
        self.y_shuffled = self.y_data[perm, 0:self.out_shape] # Cut off all non-controllable Devices.

    def create_test_and_train(self, test_pct):
        pct = int(test_pct * len(self.x_shuffled))
        self.train_x = self.x_shuffled[:-pct]
        self.train_y = self.y_shuffled[:-pct]
        self.test_x = self.x_shuffled[-pct:]
        self.test_y = self.y_shuffled[-pct:]

    def build_model(self):

        model = keras.Sequential()
        model.add(keras.layers.LSTM(self.size, return_sequences=True, input_shape=self.train_x.shape[-2:]))
        model.add(keras.layers.LSTM(self.size, return_sequences=True))
        model.add(keras.layers.LSTM(self.size))
        model.add(keras.layers.Dense(self.out_shape, activation='softmax'))
        self.myModel = model

    def compile_model(self):
        optimizer = tf.train.AdamOptimizer()
        self.myModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def run_model(self):
        # callbacks
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
        #
        self.history = self.myModel.fit(self.train_x,
                                        self.train_y,
                                        validation_data=(self.test_x, self.test_y),
                                        batch_size=1024,
                                        epochs=1024,
                                        callbacks=[early_stop])

    def save_model(self, name):
        self.myModel.save_weights(f'./model/{name}.h5')

    def load_model(self, name):
        self.myModel.load_weights(f'./model/{name}.h5')

    def print_current(self): #Experimentell

        old_line = np.zeros(self.out_shape)

        for line in self.curr_data[:, 0:self.out_shape]:   ## Sollten hier nicht vielleicht auch sensor_bin_list angezeigt werden?
            arr = line - old_line
            arr[arr < 0] = 0
            val = np.argmax(arr)
            if(val > 0):   ## Problematisch!
                action = self.tok_lst[val].split('.-.')
                print(f'{action[0]} {action[1]} {action[2]}')
            old_line = line

    def gen_analyse_current(self, tok_lst):
        self.tok_lst = tok_lst
        self.curr_data = np.append(self.x_data[-1], [self.y_raw_data[-1]], axis=0)[1:]

        pred = rnn.myModel.predict(np.expand_dims(self.curr_data, axis=0))
        self.pred_val = np.argmax(pred)

        self.curr_action = self.tok_lst[self.pred_val].split('.-.')

        print(f'Gerät {self.curr_action[0]} mit {self.curr_action[1]} = {self.curr_action[2]} wird mit {(pred[-1][self.pred_val]) * 100:.3f} % am wahrscheinlichsten geschaltet!')
        print('')
        for line in range(len(pred[-1])):
            if (pred[-1][line - 1]) > 0.01:
                action = self.tok_lst[line - 1].split('.-.')
                print(f'Wahrscheinlichkeit für {action[0]} mit {action[1]} = {action[2]} ist {(pred[-1][line - 1]) * 100:.3f} %')

class fhemConnect:
    def __init__(self, ip, telnetport):
        self.host = ip
        self.port = telnetport
        self.telnet = telnetlib.Telnet()

    def execFhem(self, command):
        connect = self.telnet
        connect.open(self.host, self.port)
        connect.write((command + "\n").encode('ascii'))
        connect.close()


# Read controllable Devices
dev_bin_data = getData(IP, USER, PSWD)
dev_bin_data.remoteConnect(device_bin_list)
dev_bin_data.sort_df()
dev_bin_data.order_df(identical_list)

# Read Sensors with active changes (e.g. motion sensors)
sens_bin_data = getData(IP, USER, PSWD)
sens_bin_data.remoteConnect(sensor_bin_list)
sens_bin_data.sort_df()
sens_bin_data.order_df(identical_list)

# Read Sensors with passive data (e.g. temperature)
stat_lst_data = getData(IP, USER, PSWD)
stat_lst_data.remoteConnect(static_list)
stat_lst_data.sort_df()
stat_lst_data.order_df(identical_list)

# Start preparing Data
dev_bin_prep = prepareData(dev_bin_data.df)
dev_bin_prep.to_one_hot()
sens_bin_prep = prepareData(sens_bin_data.df)
sens_bin_prep.to_one_hot()
stat_lst_prep = prepareData(stat_lst_data.df)
# stat_lst_prep.normalize()

# Combine controllable Devices with Sensor Data
dev_bin_prep.combine_current(sens_bin_prep.df)
dev_bin_prep.combine_static(stat_lst_prep.df)

# Normalize Data
dev_bin_prep.normalize()

# RNN
rnn = rnn_analyse(dev_bin_prep.df, 64)
rnn.to_intervals(32, len(dev_bin_prep.tok_lst))
rnn.shuffle()
rnn.create_test_and_train(1/3)
rnn.build_model()
rnn.myModel.summary()

rnn.compile_model()

## Uncommenct when running and training a new model
# rnn.run_model()
# rnn.save_model('test_01')

## Uncomment when loading an allready trained model
rnn.load_model('test_01')

# analyse the current data and predict next values
rnn.gen_analyse_current(dev_bin_prep.tok_lst)

# print out the last n values used for the prediction
print("Basierend auf den vorrangegangen Werten:")
rnn.print_current()


## Uncomment, if you want to run the predicted command on your fhem instance.
# fhem = fhemConnect(IP, telnetport)
# if rnn.curr_action[1] == 'state':
#     set_string = f"set {rnn.curr_action[0]} {rnn.curr_action[2]}"
# else:
#     set_string = f"set {rnn.curr_action[0]} {rnn.curr_action[1]} {rnn.curr_action[2]}"
# print(set_string)
# fhem.execFhem(set_string)

## Uncomment to evaluate the accuracy of your model.

# loss, acc = rnn.myModel.evaluate(rnn.test_x, rnn.test_y)
# print(f'Durschnittliche Genauigkeit liegt bei {acc*100:.2f} %')
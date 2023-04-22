import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential

# column names from covtype.info
soil_names = [f'Soil_Type{i+1}' for i in range(40)]
wilderness_area_names = [f'Wilderness_Area{i+1}' for i in range(4)]
col_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
             'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'] \
                + wilderness_area_names + soil_names + ['Cover_Type']

# 1 - read dataset
covtype_df = pd.read_csv('covtype.data', names=col_names, header=None)

# scale all numerical values - first 10 columns
numerical = covtype_df.iloc[:, :10]
categorical = covtype_df.iloc[:, 10:]

scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(numerical), columns=numerical.columns)

covtype_df_scaled = pd.concat([numerical_scaled, categorical], axis=1, join='inner')
y = covtype_df_scaled['Cover_Type']
covtype_df_scaled = covtype_df_scaled.drop(['Cover_Type'] , axis = 1)

X_train ,X_test, y_train, y_test = train_test_split(covtype_df_scaled, y, test_size = 0.30, random_state=42)
n_classes = y_train.nunique() # number of unique classes

# one hot encoding
ys_train = tf.one_hot(y_train, depth=n_classes)
ys_test = tf.one_hot(y_test, depth=n_classes)

# 2 - heuristic model 
class HeuristicModel:
   def predict(self, x_test):
      if len(x_test.shape) == 1:
      # based on 'Elevation' feature
         if x_test[0] < -2: return 1
         elif x_test[0] < -1: return 2
         elif x_test[0] < 0: return 3
         elif x_test[0] < 1: return 4
         elif x_test[0] < 1.5: return 5
         elif x_test[0] < 2.5: return 6
         else: return 7
      else:
         return [self.predict(i) for i in x_test]
      
heuristic_model = HeuristicModel()
heur_pred = heuristic_model.predict(X_test.values)
heuristic_accuracy = metrics.accuracy_score(y_test, heur_pred)

# 3 - two simple ML models
dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
dtc_accuracy = metrics.accuracy_score(y_test, dtc_pred)

rfc = RandomForestClassifier()
rfc = rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_accuracy = metrics.accuracy_score(y_test, rfc_pred)

# 4 - tf neural network 
# 4.1 find hyperparameters
def create_model(layers, activation, lr):
    model= Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(n_classes, activation='softmax'))
    model_optimizer= tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
    return model

def find_hyper_params(model, param_dict, X_train, ys_train, callback):
    grid = GridSearchCV(estimator=model, param_grid=param_dict)
    grid_result = grid.fit(X_train, np.array(ys_train), callbacks=[callback])
    return grid_result 

# to find optimal hyperparameters i used commented code below (long executing time)


# layers = [[128, 64, 64]]
# activations = ['sigmoid', 'relu']
# lrs = [0.001, 0.01]
# param_grid = dict(layers=layers, activation=activations, lr=lrs, batch_size=[128, 256], epochs=[50])
# model = KerasClassifier(model=create_model, verbose=True, activation='sigmoid', layers=[128, 64, 64], lr=0.001)
# grid_result = find_hyper_params(model, param_grid, X_train, ys_train)


# best hyperparameters
best_model = create_model([128, 64, 64], 'sigmoid', 0.001)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
losses = best_model.fit(X_train, ys_train,
                   validation_data=(X_test, ys_test),
                   batch_size=256,
                   epochs=150, 
                   callbacks=[early_stopping]
                   )
nn_pred = np.argmax(best_model.predict(X_test), axis=1)
nn_accuracy = metrics.accuracy_score(y_test, nn_pred)

plt.plot(losses.history['loss'], label='training loss')
plt.plot(losses.history['val_loss'], label='validation loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Categorical Crossentropy Loss')
plt.title('Neural Network Training')
plt.grid()
plt.show()

# 5 - compare models
print('heuristic accuracy = ', heuristic_accuracy)
print('decision tree accuracy = ', dtc_accuracy)
print('random forest accuracy = ', rfc_accuracy)
print('neural network accuracy = ', nn_accuracy)

# 6 - very simple REST API 
from enum import Enum
from fastapi import FastAPI

app = FastAPI()

class ModelName(str, Enum):
    heuristic = 'Heuristic'
    decision_tree = 'Decision Tree'
    random_forest = 'Random Forest'
    neural_network = 'Neural Network'

class TestSample(str, Enum):
    x0 = 'X_test.iloc[0]'
    x1 = 'X_test.iloc[1]'

@app.get('/')
async def choose_model(model_name: ModelName, test_sample: TestSample):
    if model_name is ModelName.heuristic:
      if test_sample is TestSample.x0:
         pred = heuristic_model.predict(X_test.iloc[0].values.reshape(1, -1))[0]
         y_true = y_test.values[0]
      elif test_sample is TestSample.x1:
         pred = dtc.predict(X_test.iloc[1].values.reshape(1, -1))[0]
         y_true = y_test.values[1]
      return {'prediction': int(pred), 'y_true': int(y_true)}
    elif model_name is ModelName.decision_tree:
      if test_sample is TestSample.x0:
         pred = dtc.predict(X_test.iloc[0].values.reshape(1, -1))[0]
         y_true = y_test.values[0]
      elif test_sample is TestSample.x1:
         pred = dtc.predict(X_test.iloc[1].values.reshape(1, -1))
         y_true = y_test.values[1]
      return {'prediction': int(pred), 'y_true': int(y_true)}
    elif model_name is ModelName.random_forest:
      if test_sample is TestSample.x0:
         pred = rfc.predict(X_test.iloc[0].values.reshape(1, -1))[0]
         y_true = y_test.values[0]
      elif test_sample is TestSample.x1:
         pred = rfc.predict(X_test.iloc[1].values.reshape(1, -1))[0]
         y_true = y_test.values[1]
      return {'prediction': int(pred), 'y_true': int(y_true)}
    elif model_name is ModelName.neural_network:
      if test_sample is TestSample.x0:
         pred = np.argmax(best_model.predict(X_test.iloc[0].values.reshape(1, -1)), axis=1)[0]
         y_true = y_test.values[0]
      elif test_sample is TestSample.x1:
         pred = np.argmax(best_model.predict(X_test.iloc[0].values.reshape(1, -1)), axis=1)[0]
         y_true = y_test.values[1]
      return {'prediction': int(pred), 'y_true': int(y_true)}

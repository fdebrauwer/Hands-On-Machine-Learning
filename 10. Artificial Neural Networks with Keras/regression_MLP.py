import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Exploration
# print(features.info())
# print(labels.info())
# print(features.describe())
# print(labels.describe())
# features.hist(bins=50, figsize=(20,15))
# labels.hist(bins=50, figsize=(20,15))
# plt.show()
# corr_matrix = housing.corr()
# print(corr_matrix['medianHouseValue'].sort_values(ascending=False))
# print(housing.isnull().sum())

# Preprocessing
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
x_train, x_test, y_train, y_test = train_test_split(features_scaled, labels)
# tf_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# tf_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(x_train.head())
print(x_train.shape)
print(type(x_train))
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(x_test.head())
print(x_test.shape)
print(type(x_test))
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(y_train.head())
print(y_train.shape)
print(type(y_train))
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print(y_test.head())
print(y_test.shape)
print(type(y_test))
print('--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


# Model building
checkpoint = ModelCheckpoint(
    'regression_MLP_model.h5', 
    save_best_only=True
)
early_stopping = EarlyStopping(
    patience=10,
    restore_best_weights=True
)
model = Sequential([
    Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    Dense(30, activation='relu'),
    Dense(30, activation='relu'),
    Dense(1)
])
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae']
)
history = model.fit(
    x_train,
    y_train,
    epochs=10000,
    batch_size=20,
    validation_split=0.1,
    callbacks=[checkpoint, early_stopping]
)
evaluation = model.evaluate(
    x_test,
    y_test
)
model.save('regression_MLP_model.h5')

history = pd.DataFrame(history.history)
print(model.summary())
print('Test set loss:', evaluation[0])
print('Test set Mean Absolute Error:', evaluation[1])

pd.concat([history.iloc[:,[0]],history.iloc[:,[2]]]).plot()
plt.grid(True)
plt.show()

pd.concat([history.iloc[:,[1]],history.iloc[:,[3]]]).plot()
plt.grid(True)
plt.show()
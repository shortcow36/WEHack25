import numpy as np
import pandas as pd
#import seaborn as sns  # for nicer plots
#sns.set(style="darkgrid")  # default style
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from matplotlib import pyplot as plt

property_data_init = pd.read_csv('Data\WeHack_DataFile - Sheet1 (1).csv', sep=',', encoding='latin-1')
property_data = property_data_init[['Lattitude', 'Longitude', 'Crime Rate','Appreciation Rate', 'Foot Traffic Rate', 'Proximity to Infrastructure', 'Neighborhood Reputation', 'Risk Score' ]].copy()

'''print('Shape of data:', property_data_init.shape)
property_data_init.head()
print(f'property_data data type by column:\n{property_data.dtypes}\n')'''

np.random.seed(0)
indices = property_data.index.to_numpy()
#print(f'indices list:\n{indices}\n')
shuffled_indices = np.random.permutation(indices)
#print(f'shuffled_indices list:\n{shuffled_indices}\n')
property_data = property_data.reindex(shuffled_indices)#.reset_index(drop=True)
#print(f'First 5 rows of reindexed car_data:\n{property_data.head(5)}\n')


Y = property_data[['Risk Score']].copy()
#print(f'Shape of Y:\n{Y.shape}\n')
X = property_data[['Lattitude', 'Longitude', 'Crime Rate', 'Appreciation Rate', 'Foot Traffic Rate', 'Proximity to Infrastructure', 'Neighborhood Reputation']].copy()
#print(f'Shape of X:\n{X.shape}\n')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1234)
'''print(f'Shape of X_train and Y_train:\n\t{X_train.shape}, {Y_train.shape}\n')
print(f'Shape of X_test and Y_test:\n\t{X_test.shape}, {Y_test.shape}\n')
print(f'Shape of X_val and Y_val:\n\t{X_val.shape}, {Y_val.shape}\n')'''


# YOUR CODE HERE
from sklearn.preprocessing import StandardScaler
# Get quantiles
quantiles = X_train.quantile([0.25, 0.5, 0.75, 0.95])
#print(f'Quantile values of X_train:\n{quantiles}\n')
# Get the ratios for each quantile and features
ratios = quantiles.apply(lambda x: x / x.min(), axis=1)
# Get the number of unique ratios for each quantile
ratios = ratios.nunique(axis=1)
# If the number of unique values for a quantile is 1 
# (meaning they are all the same value), the features are uniformly scaled for
# a given quantile. We check across all quantiles to ensure all values are 
# uniformly scaled across all quantiles. 
if (ratios == 1).all():
    print('The values are uniformly scaled across features.')
else:
    print('The values are not uniformly scaled across features.')

# When training a model, the data is standardized using the mean and standard 
# deviation of the training set. To ensure consistency, these same scaling 
# parameters should be used for the validation and test sets. Using different
# statistics would put the validation and test data on a different scale from
# the training data, resulting in inconsistent model performance.

# Creating StandardScaler object
xscaler = StandardScaler()
# Fit scaler on X_train data and transform it
scaled_data_xtr = xscaler.fit_transform(X_train)
# Storing X_train means and standard deviations
xmeans = xscaler.mean_
xstds = xscaler.scale_
# Convert back to dataframe using X_train columns
X_train_std = pd.DataFrame(scaled_data_xtr, columns=X_train.columns)
# Transform X_test data using X_train scaler
scaled_data_xte = xscaler.transform(X_test)
# Convert back to dataframe using X_test columns
X_test_std = pd.DataFrame(scaled_data_xte, columns=X_test.columns)
# Transform X_val data using X_train scaler
scaled_data_xva = xscaler.transform(X_val)
# Convert back to dataframe using X_val columns
X_val_std = pd.DataFrame(scaled_data_xva, columns=X_val.columns)
# Creating StandardScaler object
yscaler = StandardScaler()
# Fit scaler on Y_train data and transform it
scaled_data_ytr = yscaler.fit_transform(Y_train)
# Storing Y_train means and standard deviations
ymeans = yscaler.mean_
ystds = yscaler.scale_
# Convert back to dataframe using Y_train columns
Y_train_std = pd.DataFrame(scaled_data_ytr, columns=Y_train.columns)
# Transform Y_test data using Y_train scaler
scaled_data_yte = yscaler.transform(Y_test)
# Convert back to dataframe using Y_test columns
Y_test_std = pd.DataFrame(scaled_data_yte, columns=Y_test.columns)
# Transform Y_val data using Y_train scaler
scaled_data_yva = yscaler.transform(Y_val)
# Convert back to dataframe using Y_val columns
Y_val_std = pd.DataFrame(scaled_data_yva, columns=Y_val.columns)



'''tot_data = pd.concat([X_train_std, Y_train_std], axis=1)
corr = tot_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, cmap=sns.color_palette(palette='viridis'), center=0, square=True, linewidths=.5, cbar_kws={"shrink": .75})
fig.subplots_adjust(top=.95)
fig.suptitle('Correlation Matrix Heatmap of Linear Relationships Between Features and Outcome Training Data')
plt.show()'''


def bl_model(x, y_std, ystd, ymean):
    return np.mean((y_std * ystd + ymean), axis=0)

#print(bl_model(5, Y_train_std, ystds, ymeans))
#print(bl_model(10, Y_train_std, ystds, ymeans))



def build_model(num_features, learning_rate):
  """Build a TF linear regression model using Keras.

  Args:
    num_features: The number of input features.
    learning_rate: The desired learning rate for SGD.

  Returns:
    model: A tf.keras model (graph).
  """
  # This is not strictly necessary, but each time you build a model, TF adds
  # new nodes (rather than overwriting), so the colab session can end up
  # storing lots of copies of the graph when you only care about the most
  # recent. Also, as there is some randomness built into training with SGD,
  # setting a random seed ensures that results are the same on each identical
  # training run.
  tf.keras.backend.clear_session()
  tf.random.set_seed(0)

  # Build a model using keras.Sequential. While this is intended for neural
  # networks (which may have multiple layers), we want just a single layer for
  # linear regression.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(
      units=1,        # output dim
      input_shape=(num_features,),  # input dim
      use_bias=True,               # use a bias (intercept) param
      kernel_initializer=tf.ones_initializer,  # initialize params to 1
      bias_initializer=tf.ones_initializer,    # initialize bias to 1
  ))

  # We need to choose an optimizer. We'll use GD, which is actually mini-batch GD
  #optimizer = NotImplemented
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

  # Finally, compile the model. This finalizes the graph for training.
  # We specify the loss and the optimizer above
  #NotImplemented
  model.compile(optimizer=optimizer, loss='mean_squared_error')
    
  return model



tf.random.set_seed(0)
# 2. Build and compile model
# YOUR CODE HERE
lr = 0.0001 
model_tf = build_model(num_features=X_train_std.shape[1], learning_rate=lr)
#print(model_tf.summary())
# 3. Fit the model
# YOUR CODE HERE
num_epochs = 5
batch_size = 1
training_results = model_tf.fit(X_train_std, Y_train_std, 
                                epochs=num_epochs, validation_data=(X_val_std, Y_val_std),
                                verbose=2, batch_size=batch_size)
# Get history dict from training results
h = training_results.history
# Get list of epoch numbers
epoch_list = training_results.epoch





tf.random.set_seed(0)
# YOUR CODE HERE
def my_build_model(num_features, learning_rate, optimizer, metrics=None):
  """Build a TF linear regression model using Keras.

  Args:
    num_features: The number of input features.
    learning_rate: The desired learning rate for SGD.

  Returns:
    model: A tf.keras model (graph).
  """
  # This is not strictly necessary, but each time you build a model, TF adds
  # new nodes (rather than overwriting), so the colab session can end up
  # storing lots of copies of the graph when you only care about the most
  # recent. Also, as there is some randomness built into training with SGD,
  # setting a random seed ensures that results are the same on each identical
  # training run.
  tf.keras.backend.clear_session()
  tf.random.set_seed(0)

  # Build a model using keras.Sequential. While this is intended for neural
  # networks (which may have multiple layers), we want just a single layer for
  # linear regression.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(
      units=1,        # output dim
      input_shape=(num_features,),  # input dim
      use_bias=True,               # use a bias (intercept) param
      kernel_initializer=tf.ones_initializer,  # initialize params to 1
      bias_initializer=tf.ones_initializer,    # initialize bias to 1
  ))
  # Finally, compile the model. This finalizes the graph for training.
  # We specify the loss and the optimizer above
  if metrics is None:
      model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='mean_squared_error')
  else:
      model.compile(optimizer=optimizer(learning_rate=learning_rate), loss='mean_squared_error', metrics=metrics)
    
  return model

lr = 0.01 
num_epochs = 1024
batch_size = 1
metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
optimizer = tf.keras.optimizers.Nadam
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model_tf = my_build_model(num_features=X_train_std.shape[1], learning_rate=lr, optimizer=optimizer, metrics=metrics)
training_results = model_tf.fit(X_train_std, Y_train_std, 
                                epochs=num_epochs, validation_data=(X_val_std, Y_val_std),
                                verbose=0, batch_size=batch_size, callbacks=[callback])
# Get history dict from training results
h = training_results.history
# Get list of epoch numbers
epoch_list = training_results.epoch
'''plt.plot(epoch_list, h['loss'])
plt.plot(epoch_list, h['val_loss'])'''
# Set X plot axis labels to be the same as epoch_list 
'''plt.xticks(epoch_list)
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Model Training and Validation Loss')
plt.grid()
plt.show()'''
#print(f'Learned parameters of model:\nWeights:\n{model_tf.layers[0].get_weights()[0]}\nBias:\n{model_tf.layers[0].get_weights()[1]}\n')
# Using -4 for early stopping callback that takes the best weights
val_loss = h['val_loss'][-4] 
train_loss = h['loss'][-4]
#print(f'Training Loss for final epoch:\n{train_loss}\nValidation Loss for final epoch:\n{val_loss}\n')
ab_diff = np.abs(val_loss - train_loss)
av_loss = (val_loss + train_loss) / 2
percent_diff = (ab_diff / av_loss) * 100
#print(f'Percentage difference between the losses observed on the training and validation datasets: {percent_diff}\n')




train_results = model_tf.evaluate(X_train_std, Y_train_std, verbose=0)
#print('Training loss:', train_results[0])
test_results = model_tf.evaluate(X_test_std, Y_test_std, verbose=0)
#print('Test loss:', test_results[0])

# The close similarity between the training loss and test loss indicates that 
# the model is not overfitting, as it performs similarly on both the training 
# data and new, unseen data. The slightly lower test loss further suggests 
# that the model generalizes well to new data.

Y_pred = model_tf.predict(X_test_std)
Y_pred = Y_pred * ystds + ymeans

'''plt.scatter(Y_test, Y_pred)
plt.axline((0, 0), linestyle='dashed', slope=1)
plt.xlabel('Y_test')
plt.ylabel('Y_pred')
plt.title(f'Model Labels and Predictions')
plt.grid()
plt.show()'''





def predict_risk_score(lat, lon):
    # Hardcoded values for the other features
    crime_rate = 5
    appreciation_rate = 2.5
    foot_traffic_rate = 4
    proximity_to_infra = 3
    neighborhood_rep = 3.5

    # Construct input vector
    input_data = np.array([[lat, lon, crime_rate, appreciation_rate, foot_traffic_rate, proximity_to_infra, neighborhood_rep]])
    input_scaled = xscaler.transform(input_data)

    # Predict using the model
    prediction_scaled = model_tf.predict(input_scaled)

    # Reverse scaling
    prediction = yscaler.inverse_transform(prediction_scaled)

    return prediction[0][0]

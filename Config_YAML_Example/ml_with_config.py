import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# with config file
import yaml 
#read yaml file
with open('./config.yaml') as file:  # use your path to config 
    config_data = yaml.load(file)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], config_data["height"], config_data["width"], config_data["channels"])
x_test = x_test.reshape(x_test.shape[0], config_data["height"], config_data["width"], config_data["channels"])

input_shape = (config_data["height"], config_data["width"], config_data["channels"])
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(config_data["conv2d_filters"], kernel_size=(config_data["conv2d_kernel_size"], config_data["conv2d_kernel_size"]), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(config_data["pool_size"], config_data["pool_size"])))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(config_data["dense_units_1"], activation=tf.nn.relu))
model.add(Dropout(config_data["dropout"]))
model.add(Dense(config_data["dense_units_2"], activation=tf.nn.softmax))

model.compile(optimizer=config_data["optimizer"], 
              loss=config_data["loss"], 
              metrics=config_data["metrics"])
model.fit(x=x_train,y=y_train, epochs = config_data["epochs"])


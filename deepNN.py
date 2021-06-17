from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Based on a modified version of the LeNet-5 architecture (LeCun et al.,
# Gradient-Based Learning applied to document recognition, 1998),
# provided by Jay Gupta, https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392.


(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()


net = Sequential([

    # Layer 1
    Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', kernel_regularizer=l2(0.001), input_shape=(28,28,1)),

    # Layer 2
    Conv2D(filters = 32, kernel_size = 5, strides = 1, use_bias=False),

    # Layer 3
    BatchNormalization(),

    # -------------------------------- #
    Activation("relu"),
    MaxPooling2D(pool_size = 2, strides = 2),
    Dropout(0.25),
    # -------------------------------- #

    # Layer 3
    Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=l2(0.001)),

    # Layer 4
    Conv2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False),

    # Layer 5
    BatchNormalization(),

    # -------------------------------- #
    Activation("relu"),
    MaxPooling2D(pool_size = 2, strides = 2),
    Dropout(0.25),
    Flatten(),
    # -------------------------------- #

    # Layer 6
    Dense(units = 256, use_bias=False),

    # Layer 7
    BatchNormalization(),

    # -------------------------------- #
    Activation("relu"),
    # -------------------------------- #

    # Layer 8
    Dense(units = 128, use_bias=False),

    # Layer 9
    BatchNormalization(),

    # -------------------------------- #
    Activation("relu"),
    # -------------------------------- #

    # Layer 10
    Dense(units = 84, use_bias=False),

    # Layer 11
    BatchNormalization(),

    # -------------------------------- #
    Activation("relu"),
    Dropout(0.25),
    # -------------------------------- #

    # Output
    Dense(units = 10, activation = 'softmax')

    ])

net.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center = False,  # set input mean to 0 over the dataset
    samplewise_center = False,  # set each sample mean to 0
    featurewise_std_normalization = False,  # divide inputs by std of the dataset
    samplewise_std_normalization = False,  # divide each input by its std
    zca_whitening = False,  # apply ZCA whitening
    rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image
    width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip = False,  # randomly flip images
    vertical_flip = False)  # randomly flip images

datagen.fit(x_train)

variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor = 0.25, patience = 2)

history = net.fit(x_train, labels_train,
                validation_data=(x_test, labels_test),
                callbacks = [variable_learning_rate],
                epochs=45)

net.save("network_for_mnist.h5")

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

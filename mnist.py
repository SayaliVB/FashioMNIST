import tensorflow as tf
#import matplotlib.pyplot as plot
'''#load data from keras'''

img_data = tf.keras.datasets.fashion_mnist

'''divide the data into training data and testing data'''

(images_training, lables_training), (images_testing, lables_testing) = img_data.load_data()

#print(images_testing[0])
#print(lables_testing[0])
#plot.imshow(images_testing[0])

'''preprocessing'''

#remap the numbers between 0 and 1 for preprocessing
images_training = images_training/255.0
images_testing = images_testing/255.0


'''define model that includes Nueral Network inside it'''

#implement feed forward/ sequential NN

#function1 - flatten: convert the 2d image data to 1d (the images are 28 by 28 pixels)
#function2: create dense network to perform calculation on image. 128 is number of functions in the network, i.e. layers
#           (128 is the number recommened on official website)
#            activation function: will decide the output of every nueron depending on input. relu is recommended function
#function3: to represent the output(kind of image). we have 10 categories in the fashion mnist dataset
#           the putput will give probablity for each number and we will select the probablities
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10)
    ])


'''Compiling the model'''

#describe parameters for performance
#recommended combination on tensorflow website

#optimizer: how to optimize to make network better in case of wrong predictions
#loss: mathematical function that can calculate the loss within model: how wmuch woring is the model
#from_logits: since the output is used directly as it, its not translated to anything
model.compile(optimizer = "adam", loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"])

#epochs: iterations
model.fit(images_training, lables_training, epochs = 20)

'''prediction'''

print(model.predict(images_testing)[0])
print(lables_testing[0])

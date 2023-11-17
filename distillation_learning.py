# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:40:08 2023

@author: Jahirul
"""
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

class TeacherModel(tf.keras.Model):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Define the layers of the teacher model
        self.layer1 = tf.keras.layers.Dense(512, input_shape=(784,), activation='relu')
        self.layer2 = tf.keras.layers.Dense(256, activation='relu')
        self.layer3 = tf.keras.layers.Dense(128, activation='relu')
        self.layer4 = tf.keras.layers.Dense(64, activation='relu')
        self.last   = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # Define the forward pass of the teacher model
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.last(x)

# Define the student model (a smaller, more efficient neural network)
class StudentModel(tf.keras.Model):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Define the layers of the student model
        self.layer1 = tf.keras.layers.Dense(512, input_shape=(784,), activation='relu')
        self.layer2 = tf.keras.layers.Dense(256, activation='relu')
        self.last = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # Define the forward pass of the student model
        x = self.layer1(x)
        x = self.layer2(x)
        return self.last(x)
    
# Load the teacher model
teacher_model = TeacherModel()

# Define the loss function and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.legacy.Adam()

# Train the teacher model
teacher_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
history=teacher_model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test))
    
# Load the student model
student_model = StudentModel()

# Freeze the teacher model layers
for layer in teacher_model.layers:
    layer.trainable = False
    
temp=5

def distillation_los(y_true, y_pred):
    print(y_pred)
    y_true = tf.nn.softmax(y_true / temp)
    y_pred = tf.nn.softmax(y_pred / temp)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

# Train the student model
student_model.compile(optimizer=optimizer, loss=distillation_los, metrics=['accuracy'])
student_model.fit(x_train, history.model.predict(x_train), epochs=5, batch_size=32, validation_data=(x_test, y_test))

test_loss, test_acc = student_model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)














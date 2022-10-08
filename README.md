# Optimizing-neural-networks-
Optimizing neural networks using L2 regularization
```python
from keras import regularizers
tf.keras.layers.Dense(128,kernel_regularizer=regularizers.l2(0.01),activation='relu')
```
Optimizing neural networks using Dropout
```python
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
tf.keras.layers.Dense(10,Dropout(0.2)),
```
Optimizing neural networks using Early Stopping
```python
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
```

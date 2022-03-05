# 자연어처리 Source Code



* 아래 코드는 [『텐서플로 2와 머신러닝으로 시작하는 자연어처리』](https://book.naver.com/bookdb/book_detail.naver?bid=16710393) 을 인용하였음을 밝힙니다.

# tf.keras.layers

``` python
input = tf.keras.layers.input(shape = INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate = 0.2)(inputs)
hidden = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(dropout)
output = tf.keras.layers.Dense(units = 2, activation = tf.nn.sigmoid) (hidden)
```



## Conv1D

```python
INPUT_SIZE = (1, 28, 28)

inputs = tf.keras.Input(shape = INPUT_SIZE)
conv = tf.keras.layers.Conv1D(
    	filter=10,
		kernel_size=3,
		padding='same',
		activation=tf.nn.relue)(inputs)
```

## Dropout 적용한 Conv1D

```python
INPUT_SIZE = (1, 28, 28)

inputs = tf.keras.Input(shape = INPUT_SIZE)
dropout = tf.keras.laters.Dropout(rate=0.2)(inputs)
conv = tf.keras.layers.Conv1D(
		filters=10,
		kernel_size=3,
		padding='same',
		activation=tf.nn.relu)(dropout)
```



# tf.keras.laters.MaxPool1D

```python
# 1. 객체 생성 후 apply 함수를 이용해 입력값 설정
max_pool = tf.keras.layers.MaxPool(...)
max_pool.apply(input)

# 2. 객체 생성 시 입력값 설정
max_pool = tf.keras.laters.MaxPool1D(...)(input)

__init__(
	pool_size=2,
	strides=None,
	padding='valid',
	data_fromat=None,
	**kwargs)

INPUT_SIZE = (1, 28, 28)

inputs = tf.keras.input(shape = INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(input)
conv = tf.keras.layers.Conv1D(
		filters=10,
		kernel_size=3,
		padding='same',
		activation=tf.nn.relu)(input)
max_pool = tf.keras.layers.MaxPool1D(pool_size = 3, padding = 'same')(conv)
flatten = tf.keras.layers.Flatten()(max_pool)
hidden = tf.keras.layers.Dense(units = 50, activation = tf.nn.relu)(flatten)
output = tf.kerass.layers.Dense(units = 10, activation = tf.nn.softmax)(hidden)
```



# Sequential API

```python
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```



# Functional API

```python
inputs = tf.keras.input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
```



# Custom Layer

```python
class CustomLayer(layers.Layer):
    
    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        self.hidden_dimension = hidden_dimension
        self.hidden_dimension2 = hidden_dimension2
        self.output_dimension = output_dimension
        super(CustomLayer, self).__init__()
        
    def build(self, input_shape):
        self.dense_layer1 = layers.Dense(self.hidden_dimension, activation = 'relu')
        self.dense_layer2 = layers.Dense(self.hidden_dimension2, activation = 'relu')
        self.dense_layer3 = layers.Demse(self.output_dimension, activaation = 'softmax')
        
    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layers2(x)
        
        return self.dense_layers3(x)
```

```python
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(CustomLayer(64, 64, 10))
```



# Subclassing (Custom Model)

```python
class MyModel(tf.keras.Model):
    
    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        super(myModel, self).__init__(name='my model')
        self.dense_layer1 = layers.Dense(hidden_dimension, activation = 'relu')
        self.dense_layer2 = layers.Dense(hidden_dimension, activation = 'relu')
        self.dense_layer3 = layers.Dense(output_dimension, activation = 'softmax')
        
    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.demse_layer2(x)
        
        return self.dense_layer3(x)
```




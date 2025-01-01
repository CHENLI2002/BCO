import keras
from keras import layers


class NN:
    def __init__(self, num_hidden_layer, num_hidden_node, input_shape, activation, output_size, output_activation,
                 optimizer, learning_rate, loss_func, epochs, batch_size, initializer):
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_node = num_hidden_node
        self.input_shape = input_shape
        self.activation = activation
        self.model = None
        self.output_size = output_size
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.initializer = initializer

    def init_model(self):
        model = keras.Sequential([
            layers.InputLayer(input_shape=self.input_shape)
        ])

        for _ in range(self.num_hidden_layer):
            model.add(layers.Dense(self.num_hidden_node, activation=self.activation,
                                   kernel_initializer=self.initializer, bias_initializer=self.initializer))

        model.add(layers.Dense(self.output_size, activation=self.output_activation, kernel_initializer=self.initializer,
                               bias_initializer=self.initializer))

        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func)

    def get_model(self):
        return self.model

    def train_model(self, x, y):
        print(x)
        print(y)
        history = self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return history

    def predict(self, data):
        # print(data)
        # print(self.input_shape)
        return self.model.predict(data)

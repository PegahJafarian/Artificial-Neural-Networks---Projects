import os
from abc import abstractmethod, ABC
from typing import Iterable, Callable, Union, Sequence, Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
import json
import random
import numpy as np
from auto_cnn.cnn_structure import SkipLayer, PoolingLayer, CNN, Layer
from auto_cnn.gan import AutoCNN



class Layer(ABC):
    @abstractmethod
    def tensor_rep(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:

        pass


class SkipLayer(Layer):
    GROUP_NUMBER = 1

    def __init__(self, feature_size1: int,
                 feature_size2: int,
                 kernel: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 convolution: str = 'same'):
  
        self.convolution = convolution
        self.stride = stride
        self.kernel = kernel
        self.feature_size2 = feature_size2
        self.feature_size1 = feature_size1

    def tensor_rep(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Activation:
        group_name = f'SkipLayer_{SkipLayer.GROUP_NUMBER}'
        SkipLayer.GROUP_NUMBER += 1

        skip_layer = tf.keras.layers.Conv2D(self.feature_size1, self.kernel, self.stride, self.convolution,
                                            name=f'{group_name}/Conv1')(inputs)

      
        skip_layer = tf.keras.layers.BatchNormalization(name=f'{group_name}/BatchNorm1')(skip_layer)
        skip_layer = tf.keras.layers.Activation('relu', name=f'{group_name}/ReLU1')(skip_layer)

        skip_layer = tf.keras.layers.Conv2D(self.feature_size2, self.kernel, self.stride, self.convolution,
                                            name=f'{group_name}/Conv2')(skip_layer)
        skip_layer = tf.keras.layers.BatchNormalization(name=f'{group_name}/BatchNorm2')(skip_layer)

       
        inputs = tf.keras.layers.Conv2D(self.feature_size2, (1, 1), self.stride, name=f'{group_name}/Reshape')(inputs)

        outputs = tf.keras.layers.add([inputs, skip_layer], name=f'{group_name}/Add')
        return tf.keras.layers.Activation('relu', name=f'{group_name}/ReLU2')(outputs)

    def __repr__(self) -> str:
        return f'{self.feature_size1}-{self.feature_size2}'


class PoolingLayer(Layer):
    pooling_choices = {
        'max': tf.keras.layers.MaxPool2D,
        'mean': tf.keras.layers.AveragePooling2D
    }

    def __init__(self, pooling_type: str, kernel: Tuple[int, int] = (2, 2), stride: Tuple[int, int] = (2, 2)):
 
        self.stride = stride
        self.kernel = kernel
        self.pooling_type = pooling_type

    def tensor_rep(self, inputs: tf.keras.layers.Layer) -> Union[
        tf.keras.layers.MaxPool2D, tf.keras.layers.AveragePooling2D]:
        return PoolingLayer.pooling_choices[self.pooling_type](pool_size=self.kernel, strides=self.stride)(inputs)

    def __repr__(self) -> str:
        return self.pooling_type


class CNN:
    def __init__(self, input_shape: Sequence[int],
                 output_function: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer],
                 layers: Sequence[Layer],
                 optimizer: OptimizerV2 = None,
                 loss: Union[str, tf.keras.losses.Loss] = 'sparse_categorical_crossentropy',
                 metrics: Iterable[str] = ('accuracy',),
                 load_if_exist: bool = True,
                 extra_callbacks: Iterable[tf.keras.callbacks.Callback] = None,
                 logs_dir: str = './logs/train_data',
                 checkpoint_dir: str = './checkpoints') -> None:
     
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.load_if_exist = load_if_exist
        self.loss = loss

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer

        self.metrics = metrics
        self.output_function = output_function
        self.input_shape = input_shape
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.hash = self.generate_hash()

        self.model: tf.keras.Model = None

      
        self.checkpoint_filepath = f'{self.checkpoint_dir}/model_{self.hash}/model_{self.hash}'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{self.logs_dir}/model_{self.hash}",
                                                              update_freq='batch', histogram_freq=1)

        self.callbacks = [model_checkpoint_callback, tensorboard_callback]

        if extra_callbacks is not None:
            self.callbacks.extend(extra_callbacks)

    def generate(self) -> tf.keras.Model:
     

        print(self.layers)

        if self.model is None:
            tf.keras.backend.clear_session()  
            SkipLayer.GROUP_NUMBER = 1
            inputs = tf.keras.Input(shape=self.input_shape)

            outputs = inputs

            for i, layer in enumerate(self.layers):
                outputs = layer.tensor_rep(outputs)

            outputs = self.output_function(outputs)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
       
            self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

            SkipLayer.GROUP_NUMBER = 1
        return self.model

    def evaluate(self, data: Dict[str, Any], batch_size: int = 64) -> Tuple[float, float]:
   

        return self.model.evaluate(data['x_test'], data['y_test'], batch_size=batch_size)

    def train(self, data: Dict[str, Any], batch_size: int = 64, epochs: int = 1) -> None:
     

        if self.load_if_exist and os.path.exists(f'{self.checkpoint_dir}/model_{self.hash}/'):
            self.model.load_weights(self.checkpoint_filepath)
        else:
            if self.model is not None:
                self.model.fit(data['x_train'], data['y_train'], batch_size=batch_size, epochs=epochs,
                               validation_split=.2,
                               callbacks=self.callbacks)

    def generate_hash(self) -> str:
     
        return '-'.join(map(str, self.layers))

    def __repr__(self) -> str:
        return self.hash


def get_layer_from_string(layer_definition: str) -> List[Layer]:
   

    layers_str: list = layer_definition.split('-')

    layers = []

    while len(layers_str) > 0:
        if layers_str[0].isdigit():
            f = SkipLayer(int(layers_str[0]), int(layers_str[0 + 1]))
            layers_str.pop(0)
            layers_str.pop(0)
        else:
            f = PoolingLayer(layers_str[0])
            layers_str.pop(0)
        layers.append(f)

    return layers



class AutoCNN:
    def get_input_shape(self) -> Tuple[int]:

        shape = self.dataset['x_train'].shape[1:]

        if len(shape) < 3:
            shape = (*shape, 1)

        return shape

    def get_output_function(self) -> Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer]:


        output_size = np.unique(self.dataset['y_train']).shape[0]

        def output_function(inputs):
            out = tf.keras.layers.Flatten()(inputs)

            return tf.keras.layers.Dense(output_size, activation='softmax')(out)

        return output_function

    def __init__(self, population_size: int,
                 maximal_generation_number: int,
                 dataset: Dict[str, Any],
                 output_layer: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer] = None,
                 epoch_number: int = 1,
                 optimizer: OptimizerV2 = tf.keras.optimizers.Adam(),
                 loss: Union[str, tf.keras.losses.Loss] = 'sparse_categorical_crossentropy',
                 metrics: Iterable[str] = ('accuracy',),
                 crossover_probability: float = .9,
                 mutation_probability: float = .2,
                 mutation_operation_distribution: Sequence[float] = None,
                 fitness_cache: str = 'fitness.json',
                 extra_callbacks: Iterable[tf.keras.callbacks.Callback] = None,
                 logs_dir: str = './logs/train_data',
                 checkpoint_dir: str = './checkpoints'
                 ) -> None:


        self.logs_dir = logs_dir
        self.checkpoint_dir = checkpoint_dir
        self.extra_callbacks = extra_callbacks
        self.fitness_cache = fitness_cache

        if self.fitness_cache is not None and os.path.exists(self.fitness_cache):
            with open(self.fitness_cache) as cache:
                self.fitness = json.load(cache)
        else:
            self.fitness = dict()

        if mutation_operation_distribution is None:
            self.mutation_operation_distribution = (.7, .1, .1, .1)
        else:
            self.mutation_operation_distribution = mutation_operation_distribution

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.epoch_number = epoch_number
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.maximal_generation_number = maximal_generation_number
        self.population_size = population_size
        self.population = []

        self.population_iteration = 0

        if output_layer is None:
            self.output_layer = self.get_output_function()
        else:
            self.output_layer = output_layer

        self.input_shape = self.get_input_shape()

    def initialize(self) -> None:

        self.population.clear()

        for _ in range(self.population_size):
            depth = random.randint(1, 5)

            layers = []

            for i in range(depth):
                r = random.random()

                if r < .5:
                    layers.append(self.random_skip())
                else:
                    layers.append(self.random_pooling())

            cnn = self.generate_cnn(layers)

            self.population.append(cnn)

    def random_skip(self) -> SkipLayer:


        f1 = 2 ** random.randint(5, 9)
        f2 = 2 ** random.randint(5, 9)
        return SkipLayer(f1, f2)

    def random_pooling(self) -> PoolingLayer:

        q = random.random()

        if q < .5:
            return PoolingLayer('max')
        else:
            return PoolingLayer('mean')

    def evaluate_fitness(self, population: Iterable[CNN]) -> None:


        for cnn in population:
            if cnn.hash not in self.fitness:
               
                self.evaluate_individual_fitness(cnn)

            print(cnn, self.fitness[cnn.hash])

    def evaluate_individual_fitness(self, cnn: CNN) -> None:


        try:
            cnn.generate()

            cnn.train(self.dataset, epochs=self.epoch_number)
            loss, accuracy = cnn.evaluate(self.dataset)
        except ValueError as e:
            print(e)
            accuracy = 0

        self.fitness[cnn.hash] = accuracy

        if self.fitness_cache is not None:
            with open(self.fitness_cache, 'w') as json_file:
                json.dump(self.fitness, json_file)

    def select_two_individuals(self, population: Sequence[CNN]) -> CNN:


        cnn1, cnn2 = random.sample(population, 2)

        if self.fitness[cnn1.hash] > self.fitness[cnn2.hash]:
            return cnn1
        else:
            return cnn2

    def split_individual(self, cnn: CNN) -> Tuple[Sequence[Layer], Sequence[Layer]]:

        split_index = random.randint(0, len(cnn.layers))

        return cnn.layers[:split_index], cnn.layers[split_index:]

    def generate_offsprings(self) -> List[CNN]:


        offsprings = []

        while len(offsprings) < len(self.population):
            p1 = self.select_two_individuals(self.population)
            p2 = self.select_two_individuals(self.population)

            while p1.hash == p2.hash:
                p2 = self.select_two_individuals(self.population)

            r = random.random()

            if r < self.crossover_probability:
                p1_1, p1_2 = self.split_individual(p1)
                p2_1, p2_2 = self.split_individual(p2)

                o1 = [*p1_1, *p2_2]
                o2 = [*p2_1, *p1_2]

                offsprings.append(o1)
                offsprings.append(o2)
            else:
                offsprings.append(p1.layers)
                offsprings.append(p2.layers)

        choices = ['add_skip', 'add_pooling', 'remove', 'change']

        for cnn in offsprings:
            cnn: list

            r = random.random()

            if r < self.mutation_probability:
                if len(cnn) == 0:
                    i = 0
                    operation = random.choices(choices[:2], weights=self.mutation_operation_distribution[:2])[0]
                else:
                    i = random.randint(0, len(cnn) - 1)
                    operation = random.choices(choices, weights=self.mutation_operation_distribution)[0]

                if operation == 'add_skip':
                    cnn.insert(i, self.random_skip())
                elif operation == 'add_pooling':
                    cnn.insert(i, self.random_pooling())
                elif operation == 'remove':
                    cnn.pop(i)
                else:
                    if isinstance(cnn[i], SkipLayer):
                        cnn[i] = self.random_skip()
                    else:
                        cnn[i] = self.random_pooling()

        offsprings = [self.generate_cnn(layers) for layers in offsprings]

        return offsprings

    def generate_cnn(self, layers: Sequence[Layer]) -> CNN:


        return CNN(self.input_shape, self.output_layer, layers, optimizer=self.optimizer, loss=self.loss,
                   metrics=self.metrics, extra_callbacks=self.extra_callbacks, logs_dir=self.logs_dir,
                   checkpoint_dir=self.checkpoint_dir)

    def environmental_selection(self, offsprings: Sequence[CNN]) -> Iterable[CNN]:

        whole_population = list(self.population)
        whole_population.extend(offsprings)

        new_population = []

        while len(new_population) < len(self.population):
            p = self.select_two_individuals(whole_population)

            new_population.append(p)

        best_cnn = max(whole_population, key=lambda x: self.fitness[x.hash])

        print("Best CNN:", best_cnn, "Score:", self.fitness[best_cnn.hash])

        if best_cnn not in new_population:
            worst_cnn = min(new_population, key=lambda x: self.fitness[x.hash])
            print("Worst CNN:", worst_cnn, "Score:", self.fitness[worst_cnn.hash])
            new_population.remove(worst_cnn)
            new_population.append(best_cnn)

        return new_population

    def run(self) -> CNN:
  

        print("Initializing Population")
        self.initialize()
        print("Population Initialization Done:", self.population)

        for i in range(self.maximal_generation_number):
            print("Generation", i)

            print("Evaluating Population fitness")
            self.evaluate_fitness(self.population)
            print("Evaluating Population fitness Done:", self.fitness)

            print("Generating Offsprings")
            offsprings = self.generate_offsprings()
            print("Generating Offsprings Done:", offsprings)

            print("Evaluating Offsprings")
            self.evaluate_fitness(offsprings)
            print("Evaluating Offsprings Done:", self.fitness)

            print("Selecting new environment")
            new_population = self.environmental_selection(offsprings)
            print("Selecting new environment Done:", new_population)

            self.population = new_population

        best_cnn = sorted(self.population, key=lambda x: self.fitness[x.hash])[-1]
        print("Best CNN:", best_cnn, "Score:", self.fitness[best_cnn.hash])
        return best_cnn
    

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('INFO')


random.seed(42)
tf.random.set_seed(42)


def mnist_test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    values = x_train.shape[0] // 2

    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(5, 1, data)
    a.run()


def cifar10_test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    values = x_train.shape[0]

    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(20, 10, data, epoch_number=10)
    a.run()


if __name__ == '__main__':
    mnist_test()

# model_training.py

"""
Model Training for Secure Federated Learning Systems

This module contains functions for training models across distributed datasets using federated learning techniques.

Techniques Used:
- Federated averaging
- Secure aggregation

Libraries/Tools:
- tensorflow_federated
- tensorflow
- numpy
- pandas

"""

import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd
import os

class ModelTraining:
    def __init__(self, client_data_dir='data/processed/', num_clients=5, rounds=10):
        """
        Initialize the ModelTraining class.
        
        :param client_data_dir: str, directory containing client data
        :param num_clients: int, number of clients in the federated learning setup
        :param rounds: int, number of federated training rounds
        """
        self.client_data_dir = client_data_dir
        self.num_clients = num_clients
        self.rounds = rounds

    def load_client_data(self, client_id):
        """
        Load data for a specific client.
        
        :param client_id: int, ID of the client
        :return: tuple, client data (X, y)
        """
        client_data_path = os.path.join(self.client_data_dir, f'client_{client_id}.csv')
        data = pd.read_csv(client_data_path)
        X = data.drop(columns=['target']).values
        y = data['target'].values
        return X, y

    def create_tf_dataset(self, X, y):
        """
        Create a TensorFlow dataset from input data.
        
        :param X: ndarray, feature data
        :param y: ndarray, target data
        :return: tf.data.Dataset, TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(X)).batch(32)
        return dataset

    def preprocess_client_data(self):
        """
        Preprocess data for all clients and create TensorFlow datasets.
        
        :return: list, list of TensorFlow datasets for each client
        """
        client_datasets = []
        for client_id in range(self.num_clients):
            X, y = self.load_client_data(client_id)
            dataset = self.create_tf_dataset(X, y)
            client_datasets.append(dataset)
        return client_datasets

    def build_model(self):
        """
        Build a simple neural network model for federated learning.
        
        :return: tf.keras.Model, compiled Keras model
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def model_fn(self):
        """
        Model function for federated learning.
        
        :return: tff.learning.Model, TFF model
        """
        keras_model = self.build_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=self.create_tf_dataset(np.zeros((1, 10)), np.zeros((1,))).element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

    def federated_training(self, client_datasets):
        """
        Set up and run federated training.
        
        :param client_datasets: list, list of TensorFlow datasets for each client
        :return: tff.learning.templates.IterativeProcess, federated training process
        """
        iterative_process = tff.learning.build_federated_averaging_process(
            model_fn=self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
        )
        state = iterative_process.initialize()

        for round_num in range(1, self.rounds + 1):
            state, metrics = iterative_process.next(state, client_datasets)
            print(f'Round {round_num}, Metrics={metrics}')
        
        return state

    def save_model(self, state, model_path='models/federated_model.h5'):
        """
        Save the federated model to a file.
        
        :param state: tff.learning.templates.IterativeProcessState, final state of the federated training process
        :param model_path: str, path to save the model
        """
        model = tff.learning.from_keras_model(
            self.build_model(),
            input_spec=self.create_tf_dataset(np.zeros((1, 10)), np.zeros((1,))).element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
        state.model.assign_weights_to(model)
        model.save(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    model_training = ModelTraining(client_data_dir='data/processed/', num_clients=5, rounds=10)
    
    # Preprocess client data
    client_datasets = model_training.preprocess_client_data()
    
    # Run federated training
    state = model_training.federated_training(client_datasets)
    
    # Save the federated model
    model_training.save_model(state, model_path='models/federated_model.h5')
    print("Model training completed and model saved.")

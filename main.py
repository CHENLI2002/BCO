from keras.src.initializers import initializers
from src.PlaceholderEnv import PlaceholderEnv
from src.TRPO_from_web.train import main
from src.BCO import BCO

if __name__ == "__main__":
    hyperparameter_list = {
        'num_hidden_layer': 2,
        'num_hidden_node': 128,
        'input_shape': (4,),
        'model_shape': (8,),
        'activation': 'tanh',
        'output_size': 2,
        'output_activation': 'softmax',
        'optimizer': 'adam',
        'learning_rate': 0.00001,
        'loss_func': 'categorical_crossentropy',
        'epochs': 10,
        'batch_size': 16,
        'num_iteration': 500,
        'initializer': 'he_uniform',
        'rep': 2,
    }

    env = PlaceholderEnv().env
    demoer = main()
    BCO = BCO(hyperparameter_list=hyperparameter_list, env=env, demoer=demoer)
    BCO.main_loop()
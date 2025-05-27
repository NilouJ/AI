
dataset_config = {
    'full_data_path': './data/full_data.csv', 
    'train_data_path': './data/train/', 
    'test_data_path': './data/test/', 
    'split_ratio': 0.8, 
    'shuffle': True,  
}

RF_config = {
    'hyperparam_ranges': {
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 5, 10], 
        'min_samples_split': [2, 4],  
        'min_samples_leaf': [1, 2], 
        'bootstrap': [True, False],  
    },
    'default_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
    }
}

CNN_config = {
    'architecture': {
        'n_layers': 2, 
        'filters': [32, 64],  
        'kernel_sizes': [3, 3],  
        'strides': [1, 1],  
        'dropout_rate': 0.2,  
        'fc_hidden_units': [64, 128], 
    },
    'default_params': {
        'batch_size': 16,
        'learning_rate': 1e-3,
        'epochs': 10,
    }
}

training_config = {
    'batch_size': 16, 
    'learning_rate': 1e-3, 
    'epochs': 10,  
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', 
}

logging_config = {
    'tensorboard': {
        'enabled': True,
        'log_dir': './logs/tensorboard/', 
    },
    'mlflow': {
        'enabled': True,
        'experiment_name': 'Metabolomics_Model_Tuning', 
    }
}

model_io_config = {
    'save_path': './models/', 
    'load_path': './models/',  
}

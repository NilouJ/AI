# ---------------------
# Dataset and Paths
# ---------------------
dataset_config = {
    'full_data_path': './data/full_data.csv',  # Full dataset path
    'train_data_path': './data/train/',  # Train split path
    'test_data_path': './data/test/',  # Test split path
    'split_ratio': 0.8,  # Train-test split ratio
    'shuffle': True,  # Shuffle dataset during training
}

# ---------------------
# Random Forest Model Hyperparameters
# ---------------------
RF_config = {
    'hyperparam_ranges': {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [None, 5, 10],  # Tree depth
        'min_samples_split': [2, 4],  # Min samples required to split a node
        'min_samples_leaf': [1, 2],  # Min samples required to form a leaf
        'bootstrap': [True, False],  # Bootstrap samples
    },
    'default_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
    }
}

# ---------------------
# 1D CNN Model Hyperparameters
# ---------------------
CNN_config = {
    'architecture': {
        'n_layers': 2,  # Fixed to 2 layers
        'filters': [32, 64],  # Filters for each layer
        'kernel_sizes': [3, 3],  # Kernel sizes for each layer
        'strides': [1, 1],  # Stride values for each layer
        'dropout_rate': 0.2,  # Dropout rate for regularization
        'fc_hidden_units': [64, 128],  # Fully connected layer sizes
    },
    'default_params': {
        'batch_size': 16,
        'learning_rate': 1e-3,
        'epochs': 10,
    }
}

# ---------------------
# Training Parameters
# ---------------------
training_config = {
    'batch_size': 16,  # Batch size for training
    'learning_rate': 1e-3,  # Learning rate for gradient-based optimization
    'epochs': 10,  # Total training epochs
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Training device
}

# ---------------------
# Logging Parameters
# ---------------------
logging_config = {
    'tensorboard': {
        'enabled': True,
        'log_dir': './logs/tensorboard/',  # TensorBoard log directory
    },
    'mlflow': {
        'enabled': True,
        'experiment_name': 'Metabolomics_Model_Tuning',  # MLflow experiment name
    }
}

# ---------------------
# Model Save/Load Paths
# ---------------------
model_io_config = {
    'save_path': './models/',  # Directory to save trained models
    'load_path': './models/',  # Directory to load pre-trained models
}

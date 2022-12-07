import argparse
import json
import keras
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

# Constants used to get different values from the config json object
TRAINING_DATA_FILE_PATH_KEY = "train_data_file_path"
TESTING_DATA_FILE_PATH_KEY  = "test_data_file_path"
PRED_VALUE_COL_NAME_KEY     = "pred_value_col_name"
MODEL_FILE_PATH_KEY         = "model_file_path"
MONOTONIC_INFO_KEY          = "monotonicity_parameters"
MONOTONIC_COLUMN_NAMES_KEY  = "monotonic_column_names"
MONOTONIC_DIRS_KEY          = "monotonicity_directions"
MONOTONIC_WEIGHT_KEY        = "monotonicity_weight"
TRAINING_PARAMS_KEY         = "training_params"
EPOCHS_KEY                  = "epochs"
BATCH_SIZE_KEY              = "batch_size"

def create_argument_parser():
    """
    Creates the parser used by this script.
    """
    parser = argparse.ArgumentParser(description='Monotonic Neural Network Training Framework')
    parser.add_argument('--config_file',
                        type=str,
                        help='configuration file')
    return parser

def parse_data_files(train_data_file_path, test_data_file_path, pred_value_col_name):
    """
    Parses the data files at the specified paths and returns the training and testing
    data and labels.
    """
    train_dataset = pd.read_csv(train_data_file_path, index_col=0)
    test_dataset = pd.read_csv(test_data_file_path, index_col=0)
    
    # pred_value_col_name is the column that will be predicted by the model and so
    # we pop it and use it as label data.
    train_labels = train_dataset.pop(pred_value_col_name)
    test_labels = test_dataset.pop(pred_value_col_name)
    
    # Pair each label with its index, i.e. convert [l0, l1, l2 ... ] to [[0, l0], [1, l1], [2, l2] ... ]
    # (This will be used to identify input data corresponding to the label while training)
    train_labels_indexed = np.asarray([([ind, label]) for ind, label in enumerate(train_labels)])
    test_labels_indexed = np.asarray([([ind, label]) for ind, label in enumerate(test_labels)])
    
    return train_dataset, train_labels, train_labels_indexed, test_dataset, test_labels, test_labels_indexed

def original_loss(original_loss_function):
    """
    Returns a loss function that will use the original_loss_function to calculate the loss.
    """
    def original_loss(y_true, y_pred):
        # We padded the y_labels with the indices of the labels so we remove those to use with the
        # original loss function.
        correct_y_true = tf.gather(y_true, [1], axis=1)
        original_loss = original_loss_function(correct_y_true, y_pred)
        return original_loss
    return original_loss

def monotonicity_enforcing_loss(model, train_input, monotonocity_info):
    """
    Returns a loss function that computes the monotonicity loss.
    """
    train_input_tensor = tf.convert_to_tensor(train_input, dtype=tf.float32)  
    
    def compute_monotonic_loss_for_deriv_values_row(deriv_values_row, monotonicity_direction, monotonicity_weight):
        """
        Compute the monotonicity loss for the specified derivative row, monotonicity direction and weight.

        Say, for 3 inputs and a feature i, we get a derivative value row as [-0.2, 0.1, -0.3] (these are the values
        of df/di on the 3 inputs). If monotonicity_direction is 1, it means we want the output to be increasing with
        i. So all positive values of derivatives are fine (and so we reduce them to 0). The negative values should be used
        for the loss (after taking the absolute and multiplying with the monotonicity_weight). If the specified
        monotonicity_weight is 10, this function would return [2 (-1 * -0.2 * 10), 0, 3 (-1 * -0.3 * 10)].
        A similar argument follows for the case when monotonicity_direction is 0.
        """
        if monotonicity_direction == 1:
            deriv_values_row = tf.map_fn(lambda deriv: 0.0 if deriv > 0 else -1 * monotonicity_weight * deriv, deriv_values_row)
        else:
            deriv_values_row = tf.map_fn(lambda deriv: 0.0 if deriv < 0 else monotonicity_weight * deriv, deriv_values_row)
        
        return deriv_values_row
    
    def monotonicity_enforcing_loss(y_true, y_pred):        
        # Get the indices of the training examples in this batch from the labels.
        # The labels are of the form [[ind_1, label_1], [ind_2, label_2], ... ]
        # and so getting the first column (indexed at 0) gives us the needed indices.
        batch_train_eg_indices = tf.gather(y_true, [0], axis=1)
        batch_train_eg_indices = tf.reshape(batch_train_eg_indices, [-1])
        
        # Get the indices of the monotonic columns in the training data
        monotonic_columns = monotonocity_info[MONOTONIC_COLUMN_NAMES_KEY]
        monotonic_columns_indices = [train_input.columns.get_loc(col) for col in monotonic_columns if col in train_input]
        
        # Use the gradient tape to get the gradient of f (output) with respect to the input
        with tf.GradientTape() as tape:
            tape.watch(train_input_tensor)
            output = model(train_input_tensor)
        DfDx = tape.gradient(output, train_input_tensor)

        # DfDx is over all the training samples and all the features.
        # Here we get DfDx for only the samples in this batch and only the values corresponding
        # to the monotonic columns.
        DfDx_batch_data = tf.map_fn(lambda ind: DfDx[int(ind)], batch_train_eg_indices)
        DfDx_batch_data_mon_cols = tf.gather(DfDx_batch_data, monotonic_columns_indices, axis = 1)

        # The transpose means that each row now corresponds to a monotonic feature.
        # Size of the row is equal to the samples in this batch and each value
        # is the derivative of output with respect to the monotonic feature of the
        # sample. That is, DfDx_batch_data_mon_rows[i][j] means  derivative of output with
        # respect to the ith monotonic feature of the jth training sample in this batch.
        DfDx_batch_data_mon_rows = tf.transpose(DfDx_batch_data_mon_cols)
        
        # Go over all the monotonic feature derivative rows, compute the corresponding losses
        # and sum them up
        monotonic_dirs = monotonocity_info[MONOTONIC_DIRS_KEY]
        for i in range(len(monotonic_dirs)):
            row_loss = compute_monotonic_loss_for_deriv_values_row(DfDx_batch_data_mon_rows[i],
                                                                   monotonic_dirs[i],
                                                                   monotonocity_info[MONOTONIC_WEIGHT_KEY])
            
            if i == 0:
                loss = row_loss
            else:
                loss += row_loss
        
        return loss
    return monotonicity_enforcing_loss

def joint_loss(model, train_input, original_loss_function, monotonocity_info):
    """
    Returns a loss function that combines the original loss and the monotonicity enforcing loss.
    """
    def loss_computer(y_true, y_pred):
        return original_loss(original_loss_function)(y_true, y_pred) + monotonicity_enforcing_loss(model, train_input, monotonocity_info)(y_true, y_pred)
    
    return loss_computer

def get_file_path_to_save_trained_model(loaded_model_file_path):
    """
    Returns the file path where we should save the trained model.
    """
    # Get the directory where loaded model file path is saved
    loaded_model_dir = os.path.dirname(loaded_model_file_path)

    # Get loaded model name
    loaded_model_name = os.path.basename(loaded_model_file_path).split(".")[0]

    # Get the new model name by appending "_mon" and the time
    timestr = time.strftime("%Y%m%d-%H:%M:%S")
    new_model_file_name = loaded_model_name + "_mon_" + timestr + ".h5"

    return os.path.join(loaded_model_dir, new_model_file_name)

def perform_monotonic_training(config):
    """
    Main method that performs the training based on the specified config.
    """
    # Get the training and testing data
    train_dataset, train_labels, train_labels_indexed, test_dataset, test_labels, test_labels_indexed = \
        parse_data_files(config[TRAINING_DATA_FILE_PATH_KEY], config[TESTING_DATA_FILE_PATH_KEY], config[PRED_VALUE_COL_NAME_KEY])
    
    # Load the model
    model = keras.models.load_model(config[MODEL_FILE_PATH_KEY])

    # Get the loss function and the optimizer used in the model
    original_loss_function = model.loss
    model_optimizer = model.optimizer

    # Compile the model using the joint loss (that optimizes both the original loss as well as the monotonicity loss)
    monotonicity_info = config[MONOTONIC_INFO_KEY]
    model.compile(model_optimizer,
                  loss = joint_loss(model, train_dataset, original_loss_function, monotonicity_info),
                  metrics = [original_loss(original_loss_function), monotonicity_enforcing_loss(model, train_dataset, monotonicity_info)])

    # Train the data according to the specified training parameters
    training_params = config[TRAINING_PARAMS_KEY]
    model.fit(train_dataset, train_labels_indexed, epochs=training_params[EPOCHS_KEY], batch_size=training_params[BATCH_SIZE_KEY])

    # Save the model for future use
    keras.models.save_model(model,  get_file_path_to_save_trained_model(config[MODEL_FILE_PATH_KEY]))

if __name__ == "__main__":
    # Create the parser
    parser = create_argument_parser()

    # Parse the args and get the specified config file
    args = parser.parse_args()
    config_file = args.config_file

    # Read the configuration and start the monotonic training process
    config = json.loads(open(config_file, "r").read())
    perform_monotonic_training(config)
    
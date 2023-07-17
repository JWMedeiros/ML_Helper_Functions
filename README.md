# ML_Helper_Functions
A repo containing helpful machine learning functions to be used with ML projects.

## Function Descriptions:

[Load and Prep Image](#lnpimg)
[Make Confusion Matrix](#mcm)
[Pred and Plot](#pnp)
[Create Tensorboard Callback](#ctc)
[Plot Loss Curves](#plc)
[Compare Histoys](#ch)
[Unzip Data](#ud)
[Walk Through Dir](#wtd)
[Calculate Results](#cr)
[Create Model](#cm)
[Preprocess Img](#pi)
[Predict and Calculate Results Binary](#pacrb)
[Pred Timer](#pt)
[Plot Time Series](#pts)
[Mean Absolute Scaled Error](#mase)
[Evaluate Preds](#ep)


## <a id="lnpimg"></a>Load and Prep Image

load_and_prep_image(filename, img_shape=224, scale=True):

Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Args:

  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True

## <a id="mcm"></a>Make Confusion Matrix

make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 

  Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)

## <a id="pnp"></a>Pred and Plot

pred_and_plot(model, filename, class_names):

  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.

## <a id="ctc"></a>Create Tensorboard Callback

create_tensorboard_callback(dir_name, experiment_name):

  Creates a TensorBoard callback instance to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)

## <a id="plc"></a>Plot Loss Curves

plot_loss_curves(history):

  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)

## <a id="ch"></a>Compare Historys

compare_historys(original_history, new_history, initial_epochs=5):

    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)

## <a id="ud"></a>Unzip Data

unzip_data(filename):

  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.

## <a id="wtd"></a>Walk Through Dir

walk_through_dir(dir_path):

  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory

## <a id="cr"></a>Calculate Results

calculate_results(y_true, y_pred):

  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.

## <a id="cm"></a>Create Model

create_model(model_url, num_classes=10):

  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in the output layer,
      should be equal to the number of target classes, default 10.
    
  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons.

## <a id="pi"></a>Preprocess Img

preprocess_img(image, label, img_shape=224):

  Converts image tensor from any datatype -> 'float32' and reshapes
  image to [img_shape, img_shape, color_channels]

## <a id="pacrb"></a>Predict and Calculate Results Binary

predict_and_calculate_results(model, validation_data, validation_labels):

  Uses a model to make predictions on data, then uses calculate results to generate the overall metrics for the performance of the model.
  (Used with binary classification models)

## <a id="pacrm"></a>Predict and Calculate Results Multiclass

predict_and_calculate_results_multiclass(model, validation_data, validation_labels):

  Uses a model to make predictions on data, then uses calculate results to generate the overall metrics for the performance of the model.
  (Used with multi-class classification models)


## <a id="pt"></a>Pred Timer

pred_timer(model, samples):

  Times how long a model takes to make predictions on samples.

## <a id="cbpd"></a>Combine Batch Prefetch Datasets

combine_batch_prefetch_datasets (sentences, characters, labels):

  Takes in two sets of data, a list of sentences and a list of characters and one-hot encoded labels
  Combines the two datasets and zips with the labels, then prefetches and batches the new dataset and returns it.
  The list of sentences and characters must share the same length, and labels (essentially from the same dataset.)


## <a id="pts"></a>Plot Time Series

plot_time_series(timesteps, values, format='.', start=0, end=None, label=None, ylabel='BTC Price'):

  Plots timesteps (a series of points in time) against values (a series of values across timesteps).

  Parameters
  -----------
  timesteps: array of timestep values
  values: array of values across time
  format: style of plot, default is '.'
  start: where to start the plot (setting a value will index from start of timesteps and values)
  end: where to end the plot (similar to start but for the end)
  label: label to show on plot about values

## <a id="mase"></a>Mean Absolute Scaled Error
mean_absolute_scaled_error(y_true, y_pred):

  Implement MASE (assuming no seasonality of data).

## <a id="ep"></a>Evaluate Preds
evaluate_preds (y_true, y_pred):

  Takes in y_true and y_pred for a regression problem, and returns all available metrics in dictionary format for evaluation purposes.

  Requires mean_absolute_scaled_error() helper function
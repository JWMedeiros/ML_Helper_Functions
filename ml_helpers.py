import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools


def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color channels)
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  img=img/255.
  return img

def view_random_image(target_dir,target_class):
  # Setup the target directory (to view images from)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/"+random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # Show the shape of the image
  
  return img

def plot_loss_curves(history):
  """
  Returns seperate loss curves for training and validation metrics.
  """
  loss = history.history['loss']
  val_loss=history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('accuracy')
  plt.xlabel('epochs')
  plt.legend()

def plot_predictions(train_data, 
                     train_labels,
                     test_data,
                    test_labels,
                    predictions):
    #Plots training data, test data and compares predictions to ground truth labels
    plt.figure(figsize=(7,5))
    # Plot the training data in blue
    plt.scatter(train_data, train_labels, c="b", label = "Training data")
    # Plot testing data in green
    plt.scatter(test_data, test_labels, c='g', label ='Testing data')
    # Plot models predictions in red
    plt.scatter(test_data, predictions, c='r', label='Predictions')
    # Show legend
    plt.legend();

def plot_decision_boundary(model, X, y):
    # Plots the decision boundary created by a model predicting on X.
    
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max=X[:,0].min()-0.1,X[:,0].max()+0.1
    y_min, y_max = X[:,1].min() -0.1, X[:,1].max()+0.1
    xx,yy=np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min,y_max, 100))
    
    # Create X value (we're going to make predictions on these)
    x_in=np.c_[xx.ravel(), yy.ravel()]
    
    # Make predictions
    y_pred=model.predict(x_in)
    
    # Check for multi-class
    if len(y_pred[0])>1:
        print("Doing multiclass classification")
        # We have to reshape our predictions
        y_pred=np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("Doing binary classification")
        y_pred=np.round(y_pred).reshape(xx.shape)
        
    # Plot the decision boundary
    plt.contourf(xx,yy,y_pred,cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:,0],X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_conf_matrix(y_true, y_pred, classes=None, figsize=(10,10), text_size=15):
    # Create the confusion matrix
    cm=confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis] # normalize out conf matrix
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # Set labels to be classes
    if classes:
        labels=classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title='Confusion Matrix',
          xlabel='Predicted Label',
          ylabel = "True Label",
          xticks = np.arange(n_classes),
          yticks=np.arange(n_classes),
          xticklabels=labels,
          yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set threshold for different colors
    threshold = (cm.max()+cm.min())/2.

    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,f"{cm[i,j]}({cm_norm[i,j]*100:.1f}%)",
                horizontalalignment='center',
                color='white' if cm[i,j]>threshold else'black',
                size=text_size)
        
# Reconfig pred_and_plot to work with multi-class functions
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction with model
  and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred=model.predict(tf.expand_dims(img, axis=0))

  # Add in logic for multi-class and get pred class_name
  if len(pred[0])>1:
    pred_class = class_names[tf.argmax(pred[0])]
  else:
    # Get the predicted class
    pred_class = class_names[int(tf.round(pred))]

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
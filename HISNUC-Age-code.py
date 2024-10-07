#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import logging
import datetime
import random
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Concatenate, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# In[2]:


def get_run_folder(learning_rate, target_genes, test_size, image_count, batch_size, epochs):
    args_str = (
        f"LR_{learning_rate}-"
        f"TrgtGenes{target_genes}-"
        f"TestSplit_{test_size}-"
        f"ImgCnt_{image_count}-"
        f"BatchSize_{batch_size}-"
        f"Epochs_{epochs}"
    )
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return f"run_{now_str}_{args_str}"

def create_path(parent_dir, child_dirs):
    path = parent_dir
    for child_dir in child_dirs:
        path = Path(path) / child_dir
        path.mkdir(exist_ok=True)
    return path

def initialize_experiment(learning_rate, target_genes, test_size, image_count, batch_size, epochs):
    run_folder = get_run_folder(learning_rate, target_genes, test_size, image_count, batch_size, epochs)
    current_path = create_path(Path(MAIN_DIR), ["data", run_folder])
    return current_path

def setup_logger(run_path, current_path):
    logger_name = f"logger_{run_path}"
    logger = logging.getLogger(logger_name)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if not logger.handlers:
        run_path = Path(run_path)
        file_handler = logging.FileHandler(current_path / "output.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    return logger


# In[3]:


learning_rate = 0.00005
target_genes = 1
test_size = 0.33
image_count = 100
batch_size = 40
epochs = 1000
seed = 40

IMG_DIR = 'path_to_compressed_image'
csv_file = 'path_to_expression_file_this_only_for_filter_out_image_file'
MAIN_DIR = "path_to_output_directery"
nuc_feature=["tissue_specific_nucleus_feature_included"]


# In[4]:


current_path = initialize_experiment(learning_rate, target_genes, test_size, image_count, batch_size, epochs)
run_folder = get_run_folder(learning_rate, target_genes, test_size, image_count, batch_size, epochs)
logger = setup_logger(run_folder, current_path)
logger.info(tf.__version__)
logger.info(tf.config.list_physical_devices('GPU'))
logger.info('Experiment setup complete.')


# In[6]:


# CREATE DATASETS 
data=pd.read_csv("/gpfs/gibbs/pi/gerstein/tu54/imaging_project/imaging-QTL-recal/imaging-QTL-skin-sun-exposed/qupath-feature-kde-IQR-Q1Q3.csv").set_index("Image")
first_n_rows_data = data.iloc[:,:]
image_dir_path=IMG_DIR
image_file = [i.split("_")[0] for i in os.listdir(image_dir_path)]
image_filenames_df = pd.DataFrame(image_file, columns=['image_file']).set_index('image_file')
merged_data = pd.concat([image_filenames_df, first_n_rows_data], axis=1, join="inner")
merged_data['individual ID'] = merged_data.index.str.split('-').str[:2].str.join('-')
merged_data['image_file']=merged_data.index


# In[11]:


#Encode and scale features
a=pd.read_csv("/gpfs/gibbs/pi/gerstein/rm2586/GTEx-imaging-prediction/GTEx-meta-032724.csv").set_index("SUBJID")
b=a
# Apply one-hot encoding
one_hot_encoded = pd.get_dummies(b['SEX'], prefix='Sex')
one_hot_encoded = one_hot_encoded.astype(int)

one_hot_encoded2 = pd.get_dummies(b['RACE'], prefix='RACE')
one_hot_encoded2 = one_hot_encoded2.astype(int)

one_hot_encoded3 = pd.get_dummies(b['DTHHRDY'], prefix='DTHHRDY')
one_hot_encoded3 = one_hot_encoded3.astype(int)

# Drop the original column and concatenate the one-hot encoded columns
b = pd.concat([b.drop('SEX', axis=1), one_hot_encoded], axis=1)
b = pd.concat([b.drop('RACE', axis=1), one_hot_encoded2], axis=1)
b = pd.concat([b.drop('DTHHRDY', axis=1), one_hot_encoded3], axis=1)

merged_data=merged_data.merge(b, how="inner", right_on=b.index, left_on="individual ID")

merged_data=merged_data[['HGHT', 'WGHT', 'BMI', 'TRDNISCH', 
       'drinkindex', 'smokeindex', 'TRVNTSR', 'Sex_1', 'Sex_2', 'RACE_1','RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99', 'DTHHRDY_0.0',
       'DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0',
                         'image_file', 'AGE']+nuc_feature]
feature_input_shape=len(['HGHT', 'WGHT', 'BMI', 'TRDNISCH', 
       'drinkindex', 'smokeindex', 'TRVNTSR', 'Sex_1', 'Sex_2', 'RACE_1','RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99', 'DTHHRDY_0.0',
       'DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0',
                         'image_file', 'AGE']+nuc_feature)-2


scale_feature=['HGHT', 'WGHT', 'BMI', 'TRDNISCH','drinkindex', 'smokeindex']+nuc_feature
for i in scale_feature:
    scaler_na = StandardScaler()
    scaler_na.fit(np.array(merged_data[i]).reshape(-1, 1))
    merged_data[i]=scaler_na.transform(np.array(merged_data[i]).reshape(-1, 1))
    
    
shuffled_data = merged_data.sample(frac=1, random_state=seed)
shuffled_data.to_csv(os.path.join(current_path, "data_with_genotype.csv"))


# In[17]:


features = shuffled_data[['image_file']]
labels = shuffled_data["AGE"]

nucleus_feature=shuffled_data[['HGHT', 'WGHT', 'BMI', 'TRDNISCH','drinkindex', 'smokeindex', 
                               'TRVNTSR', 'Sex_1', 'Sex_2', 'RACE_1','RACE_2', 'RACE_3', 'RACE_4', 'RACE_98', 'RACE_99', 
                               'DTHHRDY_0.0','DTHHRDY_1.0', 'DTHHRDY_2.0', 'DTHHRDY_3.0', 'DTHHRDY_4.0']+nuc_feature]


# Initial split into temp and test sets
X_temp, test_data, X_nucleus_temp, nucleus_test, y_temp, test_tag = train_test_split(features, nucleus_feature, labels, test_size=test_size, random_state=seed)

# Further split of the temp set into training and validation sets
train_data, val_data, nucleus_train, nucleus_val, train_tag, val_tag = train_test_split(X_temp, X_nucleus_temp, y_temp, test_size=0.2, random_state=seed)


# In[21]:


logger.info("################whole_sample_size")
logger.info(nucleus_train.shape)
logger.info(nucleus_val.shape)
logger.info(nucleus_test.shape)


# In[22]:


#Data augementation (optional)
import random
import numpy as np
from scipy.ndimage import gaussian_filter

# Rotation
def rot(x_old):
    x_new = np.rot90(x_old,1,axes=(0, 1))
    return x_new

# Flip
def flip(x_old):
    x_new = np.flip(x_old,1)
    return x_new

# Cutting pixels out and adding them into the other side of the image
def shiftwrap(x_old):
    x_new = x_old.copy()
    box = x_old[22:32, :, :]
    x_new[22:32, :, :] = x_new[0:10, :, :]  
    x_new[0:10, :, :] = box 
    return x_new


def dropout_pixels(image, dropout_rate=0.2):
    """
    Randomly drop out a percentage of pixels in the input image.

    Args:
        image (numpy.ndarray): A 3-dimensional numpy array representing an image
            with dimensions (width, height, depth).
        dropout_rate (float): The percentage of pixels to randomly drop out,
            expressed as a decimal between 0 and 1. Defaults to 0.2.

    Returns:
        A new numpy array with the same dimensions as the input image, but with
        a percentage of pixels randomly set to 0.
    """
    assert image.ndim == 3, "Input must be a 3-dimensional numpy array."
    assert 0 <= dropout_rate <= 1, "Dropout rate must be between 0 and 1."

    # Calculate the number of pixels to drop out.
    num_pixels = int(np.round(dropout_rate * image.shape[0] * image.shape[1]))

    # Choose random pixel indices to set to 0.
    indices = np.random.choice(image.shape[0] * image.shape[1], num_pixels, replace=False)

    # Set the selected pixels to 0.
    image_flat = image.reshape(-1, image.shape[2])
    image_flat[indices, :] = 0

    # Reshape the modified array to the original shape.
    return image_flat.reshape(image.shape)

def Gaussian_image(image, dropout_rate=0.2, sigma=1):
    """
    Randomly drop out a percentage of pixels in the input image and apply a
    Gaussian filter to the resulting image.

    Args:
        image (numpy.ndarray): A 3-dimensional numpy array representing an image
            with dimensions (width, height, depth).
        dropout_rate (float): The percentage of pixels to randomly drop out,
            expressed as a decimal between 0 and 1. Defaults to 0.2.
        sigma (float): The standard deviation of the Gaussian filter. Larger values
            result in more blurring. Defaults to 1.

    Returns:
        A new numpy array with the same dimensions as the input image, but with
        a percentage of pixels randomly set to 0 and a Gaussian filter applied.
    """
    assert image.ndim == 3, "Input must be a 3-dimensional numpy array."
    assert 0 <= dropout_rate <= 1, "Dropout rate must be between 0 and 1."

    # Convert the input image to float32.
    image = image.astype(np.float32)

    # Apply dropout to the image.
    dropout_image = dropout_pixels(image, dropout_rate)

    # Apply a Gaussian filter to the image.
    filtered_image = gaussian_filter(dropout_image, sigma=sigma)

    return filtered_image


# Random augmentation
def selection(x_old,num):

    if num ==1:
        return Gaussian_image(rot(x_old), dropout_rate=0.2, sigma=0.5)
    if num ==2:
        return Gaussian_image(flip(x_old), dropout_rate=0.2, sigma=0.5)
    if num ==3:
        return flip(rot(x_old))
    if num ==4:
        return Gaussian_image(shiftwrap(x_old), dropout_rate=0.2, sigma=0.5)
    if num == 5:
        return rot(shiftwrap(x_old))
    if num == 6:
        return flip(shiftwrap(x_old))
    if num == 7:
        return rot(flip(shiftwrap(x_old)))
    if num == 8:
        return Gaussian_image(rot(flip(shiftwrap(x_old))),sigma=0.5,dropout_rate=0.2)

        


# In[23]:


def is_valid_patch(arr):
    return arr.shape[0] == 32 and arr.shape[1] == 32 and arr.shape[2] == 128 and np.isnan(arr[:,:,1]).sum() < 450 #32

def augment_patch(arr,nucleus_label, y, sum_x, sum_nucleus, sum_y, i, Image_ID,occup, is_train_data):
    if is_valid_patch(arr):
        arr1 = np.nan_to_num(arr)
        sum_x.append(arr1)
        sum_nucleus.append(nucleus_label)
        sum_y.append(y)
        Image_ID.append(i)
        occup.append(np.isnan(arr[:,:,1]).sum()/(32*32))

def process_slide(i, nucleus_label, y_label, sum_x ,sum_nucleus, sum_y, Image_ID, occup, is_train_data):
    a = np.load(i + "_features.npy")
    b = np.swapaxes(a, 0, 2)
    unit_list = []
    
    for j in range(128):
        c = b[:,:,j][~np.isnan(b[:,:,j]).all(axis=1)]
        d = (c.T[~np.isnan(c.T).all(axis=1)]).T
        unit_list.append(d)
        
    e = np.array(unit_list)
    f = np.swapaxes(e, 0, 2)
    
    f_len = f.shape[0]
    f_width = f.shape[1]
    f_len_int = f_len // 16
    f_width_int = f_width // 16
    
    for k in range(f_len_int - 2):
        for p in range(f_width_int - 2):
            patch100 = f[k * 16:(k * 16 + 32), p * 16:(p * 16 + 32),:]
            augment_patch(patch100,nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)
        
 
        patch100 = f[k * 16:(k * 16 + 32), -33:-1,:]
        augment_patch(patch100,nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)

    for p in range(f_width_int-2):
        patch100 = f[-33:-1,p * 16:(p * 16 + 32),:]
        augment_patch(patch100,nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)

    patch100 = f[-33:-1,-33:-1,:]
    augment_patch(patch100,nucleus_label, y_label, sum_x, sum_nucleus, sum_y, i, Image_ID, occup, is_train_data)
    


def process_all_slides(data, nucleus, labels, is_train_data):
    sum_x = []
    sum_nucleus=[]
    sum_y = []
    Image_ID=[]
    occup=[]

    os.chdir(IMG_DIR)
    if is_train_data:
        logger.info("Processing all slides: training")
    else:
        logger.info("Processing all slides: testing")
        

    for index, i in enumerate(data.iterrows()):

        y_label = labels.iloc[index] # access label using integer index
        image_file = data.iloc[index]['image_file']
        nucleus_label= nucleus.iloc[index]
        process_slide(image_file, nucleus_label, y_label, sum_x, sum_nucleus, sum_y, Image_ID, occup, is_train_data)

    logger.info("Finished processing.")
    return sum_x, sum_nucleus, sum_y, Image_ID, occup



def stack_data(sum_x,sum_nucleus, sum_y, Image_ID, occup):
  
    image = np.stack(sum_x)
    nucleus=np.stack(sum_nucleus)
    label = np.array(sum_y).reshape(-1)
    occup = np.array(occup)
    Image_ID=np.array(Image_ID)
    
    
    image_dic = {i: image[i] for i in range(len(Image_ID))}
    nucleus_dic= {i: nucleus[i] for i in range(len(Image_ID))}
    # Create a DataFrame for images
    image_df = pd.DataFrame({"Label": label, "Image_ID": Image_ID, "Occupancy": occup})
    #print(image_df)
    # Sort the DataFrame by Image_ID and Occupancy
    image_df.sort_values(by=["Image_ID", "Occupancy"], ascending=[True, False], inplace=True)

    # Group by Image_ID and select the top two images with the highest occupancy
    top_images = image_df.groupby("Image_ID").head(8)


    values_list = [image_dic[key] for key in top_images.index]
    nucleus_list= [nucleus_dic[key] for key in top_images.index]
    x=np.stack(values_list)
    nuc=np.stack(nucleus_list)#.reshape(-1, 5)
    print(nuc.shape)
    y=np.array(top_images["Label"]).reshape(-1, 1)
    Image_ID_sum=top_images["Image_ID"]
    Occup=np.array(top_images["Occupancy"])
    return x, nuc, y, Image_ID_sum, Occup


# In[24]:


logger.info("################whole_sample_size")
logger.info(nucleus_train.shape)
logger.info(nucleus_val.shape)
logger.info(nucleus_test.shape)


# In[25]:


sum_x_train, sum_nucleus_train, sum_y_train, Image_ID_train, occup_train = process_all_slides(train_data, nucleus_train, train_tag, True)
sum_x_val, sum_nucleus_val, sum_y_val, Image_ID_val, occup_val = process_all_slides(val_data, nucleus_val, val_tag, False)
sum_x_test, sum_nucleus_test, sum_y_test, Image_ID_test, occup_test = process_all_slides(test_data, nucleus_test, test_tag, False)


# In[26]:


logger.info("###################whole_sample_size_inuse_before_stacking")
logger.info(len(set(Image_ID_train)))
logger.info(len(set(Image_ID_val)))
logger.info(len(set(Image_ID_test)))


# In[27]:


os.chdir(current_path)


# In[28]:


train_image, train_nucleus, train_labels, Image_ID_train_final, occup_train_final = stack_data(sum_x_train, sum_nucleus_train, sum_y_train, Image_ID_train, occup_train)
val_image, val_nucleus, val_labels, Image_ID_val_final, occup_val_final= stack_data(sum_x_val, sum_nucleus_val, sum_y_val, Image_ID_val, occup_val)
test_image, test_nucleus, test_labels, Image_ID_test_final, occup_test_final = stack_data(sum_x_test, sum_nucleus_test, sum_y_test, Image_ID_test, occup_test)


# In[29]:


logger.info("################sample_size")
logger.info(train_image.shape)
logger.info(train_nucleus.shape)
logger.info(test_image.shape)
logger.info("Processing all slides: training 64*64, smaller than 1800, top4, without amplification, shift 32, decay_steps=2000, decay_rate=0.9")


# In[30]:


logger.info("###################whole_sample_size_inuse_after_stacking")
logger.info(len(set(Image_ID_train_final)))
logger.info(len(set(Image_ID_val_final)))
logger.info(len(set(Image_ID_test_final)))


# In[31]:


input_shape = train_image.shape[1:]
output_units = train_labels.shape[1]
input_shape, output_units


# In[32]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.layers import Dropout
import random
from keras.regularizers import l2
from keras.layers import Concatenate # Add this line
from tensorflow.keras.layers import Input

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(20)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
tf.config.experimental.enable_op_determinism()

def build_model(output_units: int, input_shape: tuple, feature_input_shape: int):
    random.seed(10)
    model = keras.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D())
    model.add(layers.Dropout(0.2, seed=20))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Dropout(0.2, seed=20))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Dropout(0.2, seed=20))
    model.add(layers.Flatten())


    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))

    age_input = keras.layers.Input(shape=(feature_input_shape,), name='nucleus_input')
    merged = Concatenate()([model.output, age_input])
    merged1= layers.Dense(units=64, activation='relu')(merged)
    
    merged2=layers.Dense(units=16, activation='relu')(merged1)
    
    model_output = layers.Dense(units=output_units, activation='linear')(merged2)

    model = keras.Model(inputs=[model.input, age_input], outputs=model_output)

    return model


# In[33]:


model = build_model(output_units, input_shape, feature_input_shape)
logger.info("model summary")
logger.info(model.summary())
os.chdir(current_path)


# In[34]:


#learning_rate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=10000,
    decay_rate=0.95,
    staircase=True)

callbacks = [
    ModelCheckpoint(
        filepath="noise_mnist.keras",
        save_best_only=True,
        monitor="val_loss"),
    EarlyStopping(monitor='val_loss', patience=30)
]

opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(optimizer=opt, loss="mse")

history = model.fit([train_image, train_nucleus], train_labels, epochs=epochs, batch_size=batch_size, validation_data=([val_image,val_nucleus], val_labels), callbacks=callbacks)
y_pred = model.predict([test_image,test_nucleus])
results = model.evaluate([test_image,test_nucleus], test_labels)


# In[36]:


import os
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def save_model(model: tf.keras.Model,
               history: tf.keras.callbacks.History,
               current_path: str,
               target_dir: str,
               model_name: str,
               y_pred: np.ndarray,
               test_label: np.ndarray):
    target_dir_path = Path(current_path) / target_dir
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    assert model_name.endswith(".h5"), "model_name should end with '.h5'"
    model_save_path = target_dir_path / model_name
    
    logger.info("[INFO] Saving model to: %s", model_save_path)
    model.save(model_save_path)
    
    with open(target_dir_path / 'basic_history.pickle', 'wb') as f:
        pickle.dump(history.history, f)
    
    np.save(target_dir_path / "y_pred.npy", y_pred)
    np.save(target_dir_path / "test_label.npy", test_label)


# In[37]:


save_model(model=model,
           history=history,
           current_path=current_path,
           target_dir="models",
           model_name="basic.h5",
           y_pred=y_pred,
           test_label=test_labels)


# In[38]:


#%%writefile plots.py

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
def plot_loss(history, current_path):
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(current_path / "mse_validation_loss.png", bbox_inches="tight")
    plt.show()

def plot_log_loss(history, current_path):
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.title('model loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(current_path / "mse_validation_loss-log.png",bbox_inches="tight")
    plt.show()
    
    
def aggregate_value(y_pred, test_label, Image_ID):
    
    value_to_indices = {}
    for i, value in enumerate(Image_ID):
        if value not in value_to_indices:
            value_to_indices[value] = [i]
        else:
            value_to_indices[value].append(i)

    # Initialize an empty array to store the grouped data
    grouped_array_label = np.empty((len(value_to_indices), 1))
    grouped_array_pred = np.empty((len(value_to_indices), 1))

    # Calculate means and populate the grouped array
    for i, value in enumerate(value_to_indices.keys()):
        indices = value_to_indices[value]
        grouped_row_label = np.mean(test_label[indices], axis=0)
        grouped_row_pred = np.mean(y_pred[indices], axis=0)
        grouped_array_label[i] = grouped_row_label
        grouped_array_pred[i] = grouped_row_pred
    np.save("y_pred_wsi.npy", grouped_row_pred)
    np.save("test_label_wsi.npy", grouped_row_label)
    
    return grouped_array_pred, grouped_array_label


def calculate_correlation(y_pred_wsi, test_label_wsi):
    corr=[]
    corr_df=pd.DataFrame(np.nan, index=range(1), columns=range(1))
    p_df=pd.DataFrame(np.nan, index=range(1), columns=range(1))
    mae_df = pd.DataFrame(np.nan, index=range(1), columns=range(2))
    for i in range(1):

        j=i
        a=stats.pearsonr(y_pred_wsi[:,j], test_label_wsi[:,i])#spearmanr, axis=0, nan_policy='propagate', alternative='two-sided')  
        #print(a)
        a=list(a)
        corr=a[0]
        corr_df.iloc[i,0]=corr
        p=a[1]
        p_df.iloc[i,0]=p
        
        
        #mae = np.sqrt(mean_squared_error(y_pred_wsi[:,i], test_label_wsi[:,i]))#
        mae=median_absolute_error(y_pred_wsi[:,i], test_label_wsi[:,i])
        mae_df.iloc[i,0] = mae
        mean_age_label = np.mean(test_label_wsi[:,i])
        mae_df.iloc[i,1]=mean_age_label

    final_corr005=pd.DataFrame(np.nan, index=range(1), columns=range(1))
    final_corr005["corr_p_005"]=corr_df[p_df<0.05]
    final_corr005["index"]=range(1)
    final_corr005["raw_corr"]=corr_df
    final_corr005["raw_p"]=p_df
    multi=multipletests(np.array(final_corr005["raw_p"]) , alpha=0.05, method='hs', is_sorted=False, returnsorted=False)

    final_corr005["multi"]=multi[1]
    final_corr005['corr_multi_005'] = final_corr005['corr_p_005'].where(final_corr005['multi'] < 0.05)
    final_corr005.to_excel("aging_prediction_corr_005_genes.xlsx")
    mae_df.to_excel("aging_prediction_mae_genes.xlsx")

    return final_corr005["corr_multi_005"], multi, mae

def plot_corr_distribution(data, current_path):
    data=data[~np.isnan(data)]
    fig = plt.figure(figsize =(2, 6))
    plt.style.use("fast")
    plt.boxplot(data)
    plt.ylim(0,1.0)
    plt.tick_params(labelsize=14)
    plt.savefig(current_path / "correlation-distribution.png",bbox_inches="tight")
    plt.show()
    return None

def plot_scatter(y_pred_wsi, test_label_wsi, mae):


    # Plot the two arrays as a scatter plot
    plt.scatter(test_label_wsi, y_pred_wsi, c='blue', label='Scatter Plot')
    plt.xlabel('label')
    plt.ylabel('pred')
    plt.title('Scatter Plot of Two Arrays')
    plt.legend()
    plt.grid(True)
    corr,p=stats.pearsonr(np.array(y_pred_wsi).reshape(-1), np.array(test_label_wsi).reshape(-1))   
    plt.text(0.03, 0.85, f'Pearson Correlation: {corr:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.03, 0.8, f'P value: {p:.2e}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.03, 0.7, f'mae: {mae:.2f}', fontsize=12, transform=plt.gca().transAxes)

    plt.savefig('scatter_plot_WSI.png')
    
    return None


def all_plots(history, y_pred, test_label, Image_ID_test, current_path):

    plot_loss(history, current_path)
    plot_log_loss(history, current_path)

    y_pred_wsi, test_label_wsi=aggregate_value(y_pred, test_label, Image_ID_test)
    data, multi, mae = calculate_correlation(y_pred_wsi, test_label_wsi)
    plot_corr_distribution(data, current_path)

    np.save("y_pred_wsi.npy",y_pred_wsi)
    np.save("test_label_wsi.npy",test_label_wsi)
    
    plot_scatter(y_pred_wsi, test_label_wsi, mae)


    return None


# In[39]:


logger.info("Making plots")
all_plots(history, y_pred, test_labels, Image_ID_test_final, current_path)
logger.info("Done.")


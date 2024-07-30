import pandas as pd       
import matplotlib as mat
from keras.models import load_model
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
import random
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import sem, t
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, Xception, VGG16, MobileNetV3Large, MobileNet, MobileNetV2, DenseNet121, InceptionV3, VGG19, InceptionResNetV2, NASNetLarge, DenseNet201, EfficientNetV2S
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.layers import BatchNormalization, GlobalMaxPooling2D
from keras.models import clone_model
from tensorflow.keras.models import Sequential
from sklearn.metrics import recall_score
import visualkeras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from skimage.filters import gaussian
from skimage import exposure, feature
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
from tensorflow.keras.callbacks import ModelCheckpoint
import shutil
from sklearn.metrics import f1_score
import warnings
from sklearn.utils import shuffle
import gc
warnings.filterwarnings('ignore')
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.metrics import precision_score
from tensorflow.keras.metrics import Recall, Precision
from sklearn.metrics import classification_report
from sklearn.utils import resample

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



main_path = "../Data/labeledData"

fractured = glob.glob(os.path.join(main_path, "test", "Fractured", "*.jpg"))
non_fractured = glob.glob(os.path.join(main_path, "test", "Non_Fractured", "*.jpg"))

df_train = pd.DataFrame(np.concatenate([['Fractured'] * len(fractured), ['Non_Fractured'] * len(non_fractured)]), columns=['class'])
df_train['image'] = [x for x in fractured] + [x for x in non_fractured]
df_train['name'] = df_train['image'].apply(os.path.basename)

print("Echantillon des donnees avant le pretraitement :")
print(df_train[['name', 'class']].head())


df_train['original_image'] = df_train['image']


print("\nDataFrame :")
print(df_train.head())

main_path = "../Data/unlabeledData"

images = glob.glob(os.path.join(main_path, "*.jpg"))

df_non_labelised = pd.DataFrame({'image': images})



taille_image = 224

def redimensionner_image_cv(image, taille=(taille_image, taille_image)):
    image_redim = cv2.resize(image, taille, interpolation=cv2.INTER_AREA)
    
    return image_redim

def supprimer_bords_noirs(image):
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gris, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    image_recadree = image[y:y+h, x:x+w]

    return image_recadree


def normaliser_image(image):
    image_normalisee = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image_normalisee

def egaliser_histogramme_adaptatif(image):
    image_egalisee = exposure.equalize_adapthist(image, clip_limit=0.03)
    
    return image_egalisee


def pipeline_pretraitement(image_path):
    image = cv2.imread(image_path)
    sans_bord = supprimer_bords_noirs(image)
    image_histo = egaliser_histogramme_adaptatif(sans_bord)
    image_normalisee = normaliser_image(image_histo)
    redim = redimensionner_image_cv(image_normalisee)

    return redim


datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.4,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
)




    
def build_model_xception():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(taille_image, taille_image, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.6)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=SGD(learning_rate=5e-04, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    model.model_type = 'Xception'
    return model
    
    
    
def build_model_InceptionResNetV2():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(taille_image, taille_image, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.6)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.model_type = 'InceptionResNetV2'
    return model






df_train['image'] = df_train['image'].apply(pipeline_pretraitement)



X_all = np.stack(df_train['image'].values)
y_all = df_train['class'].map({'Fractured': 1, 'Non_Fractured': 0}).values




df_non_labelised_clean = pd.DataFrame(columns=['image'])
df_non_labelised_clean['image'] = df_non_labelised['image'][:2500].apply(pipeline_pretraitement)

X_train_non_labelised = np.stack(df_non_labelised_clean['image'].values)




X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=14, stratify=y_all, random_state=5)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=5, stratify=y_temp, random_state=5)


augmented_images = []
augmented_labels = []

for i in range(len(X_train)):
   x = X_train[i]
   y = y_train[i]
   x = x.reshape((1,) + x.shape)  
   
   aug_iter = datagen.flow(x, batch_size=1)
   for _ in range(20):
       aug_image = next(aug_iter)[0]
       augmented_images.append(aug_image)
       augmented_labels.append(y)

X_train_augmented = np.array(augmented_images)
y_train_augmented = np.array(augmented_labels)

n_models = 6
models = []


for i in range(n_models):
   X, y = resample(X_train_augmented, y_train_augmented, replace=True, n_samples=len(X_train_augmented))
   
   train_generator = datagen.flow(X, y, batch_size=4)
   
   if i % 2 == 0:
     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4, min_lr=1e-6)
 
     model = build_model_xception()
     model.fit(train_generator, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], class_weight={0:1, 1:1.5})
     models.append(model)
     
   else:
     model = build_model_InceptionResNetV2()
     early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
     model.fit(train_generator, epochs=100, validation_data=(X_val, y_val),  callbacks=[early_stopping], class_weight={0:1, 1:2.5})
     models.append(model)
     
     
for j, model in enumerate(models):
 model.save(f"modeles/modelEnsemble_model{j+1}.h5")
 
 

def ensemble_predictions(models, X):
   predictions = []
   for model in models:
       predictions.append(model.predict(X))
       gc.collect()
   predictions_mean = np.mean(predictions, axis=0)
   return predictions_mean, predictions
    
    
    
def majority_vote(predictions):
    predictions_classes = (np.array(predictions) > 0.5).astype(int)
    majority_vote_preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=predictions_classes)
    return majority_vote_preds

y_pred_mean, predictions = ensemble_predictions(models, X_test)
y_pred_classes_mean = (y_pred_mean > 0.5).astype(int)
y_pred_classes_majority = majority_vote(predictions)

accuracy_mean = accuracy_score(y_test, y_pred_classes_mean)
accuracy_majority = accuracy_score(y_test, y_pred_classes_majority)

print(f'Ensemble Test Accuracy (Mean): {accuracy_mean:.4f}')
print(f'Ensemble Test Accuracy (Majority Vote): {accuracy_majority:.4f}')

conf_matrix_mean = confusion_matrix(y_test, y_pred_classes_mean)
print("Matrice de confusion (Moyenne des predictions) :\n", conf_matrix_mean)

conf_matrix_majority = confusion_matrix(y_test, y_pred_classes_majority)
print("Matrice de confusion (Vote majoritaire) :\n", conf_matrix_majority)



from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score

loss_mean = log_loss(y_test, y_pred_mean)
print("Log Loss (Mean Predictions):", loss_mean)

auc_roc_mean = roc_auc_score(y_test, y_pred_mean)
print("AUC ROC (Mean Predictions):", auc_roc_mean)





del predictions,  y_pred_mean, y_pred_classes_mean, X_train_augmented, y_train_augmented, X, y, train_generator





def balance_classes(X, y, sample_weight=None):
    unique, counts = np.unique(y, return_counts=True)
    min_class = unique[np.argmin(counts)]
    max_class = unique[np.argmax(counts)]
    
    X_minority = X[y == min_class]
    y_minority = y[y == min_class]
    
    if sample_weight is not None:
        weights_minority = sample_weight[y == min_class]
    else:
        weights_minority = np.ones(len(y_minority))
    
    augmented_images = []
    augmented_labels = []
    augmented_weights = []
    
    total_augmentations = counts[max_class] - counts[min_class]
    augmentations_per_example = total_augmentations // len(X_minority)
    remaining_augmentations = total_augmentations % len(X_minority)

    for i in range(len(X_minority)):
        x = X_minority[i].reshape((1,) + X_minority[i].shape) 
        num_augmentations = augmentations_per_example + (1 if i < remaining_augmentations else 0)
        
        aug_iter = datagen.flow(x, batch_size=1)
        for _ in range(num_augmentations):
            aug_image = next(aug_iter)[0]
            augmented_images.append(aug_image)
            augmented_labels.append(min_class)
            augmented_weights.append(weights_minority[i])
    
    X_augmented = np.concatenate((X, np.array(augmented_images)), axis=0)
    y_augmented = np.concatenate((y, np.array(augmented_labels)), axis=0)
    if sample_weight is not None:
        sample_weight = np.concatenate((sample_weight, np.array(augmented_weights)), axis=0)
    
    return X_augmented, y_augmented, sample_weight

    
seuil = 0.92
    
batch_size = 1



from scipy.stats import mode

    
    
def ensemble_predictions_mean(models, X, batch_size=16):
    predictions = []
    for model in models:
        model_predictions = []
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            model_predictions.append(model.predict(X_batch))
            del X_batch
            gc.collect()
        model_predictions = np.concatenate(model_predictions, axis=0)
        predictions.append(model_predictions)
        del model_predictions  
        gc.collect()
    predictions_mean = np.mean(predictions, axis=0)
    gc.collect()
    return predictions_mean, predictions
    

for iteration in range(7):
    gc.collect()
    
    y_pred_non_labelised_mean = []
    
    for i in range(0, len(X_train_non_labelised), batch_size):
        X_batch = X_train_non_labelised[i:i + batch_size]
        y_pred_batch_mean, _ = ensemble_predictions(models, X_batch, batch_size=batch_size)
        y_pred_non_labelised_mean.append(y_pred_batch_mean)

        del X_batch, y_pred_batch_mean
        gc.collect()
        
    y_pred_non_labelised_mean = np.concatenate(y_pred_non_labelised_mean, axis=0)
    

    
    confident_indices = np.where((y_pred_non_labelised_mean >= seuil) | (y_pred_non_labelised_mean <= 1 - seuil))[0]
    confident_indices = np.unique(confident_indices)
    


    confident_X = X_train_non_labelised[confident_indices]
    confident_y = (y_pred_non_labelised_mean[confident_indices] > 0.5).astype(int).flatten()
    
    # Ajout des poids bases sur la confiance des modeles
    confident_weights = np.where(confident_y == 1, y_pred_non_labelised_mean[confident_indices], 1 - y_pred_non_labelised_mean[confident_indices])
    
    del y_pred_non_labelised_mean
    gc.collect()
    
    X_train_combined = np.concatenate((X_train, confident_X), axis=0)
    y_train_combined = np.concatenate((y_train, confident_y), axis=0)
    weights_combined = np.concatenate((np.ones(len(y_train)), confident_weights), axis=0)
    
    X_train_combined, unique_indices = np.unique(X_train_combined, axis=0, return_index=True)
    y_train_combined = y_train_combined[unique_indices]
    weights_combined = weights_combined[unique_indices]
    
    X_train_balanced, y_train_balanced, weights_balanced = balance_classes(X_train_combined, y_train_combined, sample_weight=weights_combined)
    
    augmented_images = []
    augmented_labels = []
    augmented_weights = []
    
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        w = 1  
        x = x.reshape((1,) + x.shape) 
        y = np.array([y])            
        w = np.array([w])    
        
        aug_iter = datagen.flow(x, batch_size=1)
        for _ in range(5):
            aug_image = next(aug_iter)[0]
            augmented_images.append(aug_image)
            augmented_labels.append(y[0]) 
            augmented_weights.append(w[0])
    
    X_train_balanced = np.array(augmented_images)
    y_train_balanced = np.array(augmented_labels)
    weights_balanced = np.array(augmented_weights)
    
    del confident_X, confident_y, confident_weights, X_train_combined, y_train_combined, weights_combined, unique_indices, augmented_images, augmented_labels, augmented_weights
    gc.collect()
    
    print(f"Iteration {iteration+1}: Nombre d'images ajoutees = {len(confident_indices)}")
    
    if len(confident_indices) == 0:
        print(f"Iteration {iteration+1}: Aucune image ajoutee, recommencer l'iteration.")
        iteration -= 1  
        seuil = seuil - 0.05
        print(f"Seuil :  {seuil}")
        continue 
    
    new_models = []
    for i, model in enumerate(models):
        positive_indices = np.where(y_train_balanced == 1)[0]

        adjusted_weights = np.copy(weights_balanced).astype(np.float64)
        
        adjusted_weights[positive_indices] *= 1.5
        
        X_resampled, y_resampled, w_resampled = resample(X_train_balanced, y_train_balanced, adjusted_weights,
                                                         replace=True, n_samples=len(X_train_balanced))

        train_data = tf.data.Dataset.from_tensor_slices((x, y, w)).batch(64)

        
        if i % 2 == 0:
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4, min_lr=1e-6)
            model.compile(optimizer=SGD(learning_rate=5e-04, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(train_data, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])
            new_models.append(model)
            
        else:
            early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(train_data, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
            new_models.append(model)
            
    models = new_models

            
            
            
            

            
    del confident_indices, X_train_balanced, y_train_balanced, x, y, train_data
    tf.keras.backend.clear_session()
    gc.collect()
    
    
    from sklearn.metrics import log_loss

    
    
    
    y_pred_mean_new, predictions_new = ensemble_predictions_mean(models, X_test)
    _, predictions_new = ensemble_predictions(models, X_test)
    y_pred_classes_mean_new = (y_pred_mean_new > 0.5).astype(int)
    y_pred_classes_majority_new = majority_vote(predictions_new)
    
    accuracy_mean_new = accuracy_score(y_test, y_pred_classes_mean_new)
    accuracy_majority_new = accuracy_score(y_test, y_pred_classes_majority_new)
    
    print(f'New Ensemble Test Accuracy (Mean): {accuracy_mean_new:.4f}')
    print(f'New Ensemble Test Accuracy (Majority Vote): {accuracy_majority_new:.4f}')
    
    conf_matrix_mean_new = confusion_matrix(y_test, y_pred_classes_mean_new)
    print("New Matrice de confusion (Moyenne des predictions) :\n", conf_matrix_mean_new)
    
    conf_matrix_majority_new = confusion_matrix(y_test, y_pred_classes_majority_new)
    print("New Matrice de confusion (Vote majoritaire) :\n", conf_matrix_majority_new)
    
    loss_mean_new = log_loss(y_test, y_pred_mean_new)
    print("New Log Loss (Mean Predictions):", loss_mean_new)
    
    auc_roc_mean_new = roc_auc_score(y_test, y_pred_mean_new)
    print("New AUC ROC (Mean Predictions):", auc_roc_mean_new)
    
    
    del predictions_new, y_pred_mean_new, y_pred_classes_mean_new, y_pred_classes_majority_new, accuracy_mean_new, accuracy_majority_new, conf_matrix_mean_new, conf_matrix_majority_new
    
    
    import matplotlib.cm as cm

    original_images = df_train.loc[df_train.index[X_temp.shape[0]:], 'original_image'].values
    
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
            x = layer(x)
        classifier_model = keras.Model(classifier_input, x)
    
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
    
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        
        
        del last_conv_layer, last_conv_layer_model, classifier_input, x, pooled_grads
        
        return heatmap
    
    def save_gradcam(img_array, heatmap, idx, label, output_dir='gradCamEnsemble', alpha=0.4):
        # Normalize the heatmap between 0 and 1
        heatmap = np.uint8(255 * heatmap)
        
        # Apply the colormap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        
        # Resize heatmap to match the image size
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        
        # Convert original image to uint8
        original_img_uint8 = np.uint8(img_array)
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + original_img_uint8
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"iteration{iteration}_test_image_{idx}_{label}.png")
        superimposed_img.save(output_path)
        print(f"Saved averaged Grad-CAM image at: {output_path}")
        
        
        del superimposed_img
    
    def generate_avg_gradmac_for_models(X_test, original_images, y_test, models, output_dir='gradCamEnsemble'):
      for idx, (img_array, original_img, label) in enumerate(zip(X_test, original_images, y_test)):
          img_array_expanded = np.expand_dims(img_array, axis=0)
          heatmaps = []
          target_shape = None
          
          for model in models:
              if hasattr(model, 'model_type'):
                  if model.model_type == 'Xception':
                      last_conv_layer_name = 'block14_sepconv2_act'
                  else:
                      last_conv_layer_name = 'conv_7b_ac'
              else:
                  print(f"Warning: Model at index {idx} does not have 'model_type' attribute")
                  continue
  
              heatmap = make_gradcam_heatmap(img_array_expanded, model, last_conv_layer_name)
              if heatmap is not None and heatmap.size != 0:
                  if target_shape is None:
                      target_shape = heatmap.shape
                  else:
                      heatmap = np.resize(heatmap, target_shape)  # Resize heatmap to target shape
                  heatmaps.append(heatmap)
              else:
                  print(f"Warning: No heatmap generated for model {model.model_type} at index {idx}")
  
          if heatmaps:
              try:
                  avg_heatmap = np.mean(np.stack(heatmaps), axis=0)
              except Exception as e:
                  print(f"Error averaging heatmaps: {e}, at index {idx}")
                  continue
  
              if isinstance(original_img, str):
                  original_img = np.array(Image.open(original_img))
  
              save_gradcam(original_img, avg_heatmap, idx, 'Fractured' if label == 1 else 'Non_Fractured', output_dir=output_dir)
          else:
              print(f"No heatmaps to average at index {idx}")
  
    generate_avg_gradmac_for_models(X_test, original_images, y_test, models)
    
    
    del original_images
    
    seuil = seuil - 0.02
    
    
    

import pandas as pd       
import matplotlib as mat
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
import visualkeras
from skimage.filters import gaussian
from skimage import exposure, feature
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
from tensorflow.keras.callbacks import ModelCheckpoint
import shutil
from sklearn.metrics import f1_score
import warnings
from sklearn.utils import shuffle
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



taille_image = 299

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

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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



def build_model_xception_opt(optimizer):
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

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def build_model_InceptionResNetV2_opt(optimizer):
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

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
        
models = [
    {'name': 'Xception', 'build_fn': build_model_xception_opt},
    {'name': 'InceptionResNetV2', 'build_fn': build_model_InceptionResNetV2_opt},
]
        
# Taix d'apprentissage à tester
taux_apprentissage = [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5]

def get_optimizer_SGD(lr):
    return SGD(learning_rate=lr, momentum=0.9)

def get_optimizer_Adam(lr):
    return Adam(learning_rate=lr)

n_splits = 4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {}

for lr in taux_apprentissage:

    for model_info in models:
        model_name = model_info['name']
        build_fn = model_info['build_fn']
        optimiseur = get_optimizer_SGD(lr) if model_name == 'Xception' else get_optimizer_Adam(lr)

        if model_name not in results:
            results[model_name] = {'val_loss_history': {}, 'val_accuracy_history': {}}

        results[model_name]['val_loss_history'][lr] = []
        results[model_name]['val_accuracy_history'][lr] = []

        df_clean = pd.DataFrame(columns=['image'])
        df_clean['image'] = df_train['image'].apply(pipeline_pretraitement)
        df_clean['class'] = df_train['class']

        X_train_chwapi = np.stack(df_clean['image'].values)
        y_train_chwapi = df_clean['class'].map({'Fractured': 1, 'Non_Fractured': 0}).values

        for train_index, test_index in skf.split(X_train_chwapi, y_train_chwapi):
            X_train, X_test = X_train_chwapi[train_index], X_train_chwapi[test_index]
            y_train, y_test = y_train_chwapi[train_index], y_train_chwapi[test_index]


            if model_name in ['Xception']:
                nb_augment = 20
                batch = 32
                optimiseur = get_optimizer_SGD(lr)
                opt_nom = "SGD"

            else:
                nb_augment = 10
                batch = 100
                optimiseur = get_optimizer_Adam(lr)
                opt_nom = 'Adam'

            augmented_images = []
            augmented_labels = []

            for image, label in zip(X_train, y_train):
                augmented_images.append(image)
                augmented_labels.append(label)

            for i in range(len(X_train)):
                image = X_train[i:i+1]
                label = y_train[i:i+1]

                for _ in range(nb_augment):
                    augmented_image, augmented_label = next(datagen.flow(image, label, batch_size=1))
                    augmented_images.append(augmented_image[0])
                    augmented_labels.append(augmented_label[0])

            X_augmented = np.array(augmented_images)
            y_augmented = np.array(augmented_labels)

            model = build_fn(optimiseur)
            history = model.fit(X_augmented, y_augmented, validation_data=(X_test, y_test), epochs=50, batch_size=batch, verbose=1)

            results[model_name]['val_loss_history'][lr].append(history.history['val_loss'])
            results[model_name]['val_accuracy_history'][lr].append(history.history['val_accuracy'])

for model_name, metrics in results.items():
    plt.figure(figsize=(7, 6))  
    for lr, val_loss_histories in metrics['val_loss_history'].items():
        val_loss_avg = np.mean(val_loss_histories, axis=0)
        plt.plot(range(1, len(val_loss_avg) + 1), val_loss_avg, label=f'Taux d\'apprentissage={lr}')
    plt.title(f'{model_name} - Perte de validation moyenne\n par epochs pour différents taux d\'apprentissage')
    plt.xlabel('Epochs')
    plt.ylabel('Perte de validation')
    plt.legend()
    plt.ylim(0, 3) 
    plt.tight_layout() 
    plt.savefig(f'graphiques/{model_name}_LR_loss.png')  
    plt.close()  

    plt.figure(figsize=(7, 6))  
    for lr, val_accuracy_histories in metrics['val_accuracy_history'].items():
        val_accuracy_avg = np.mean(val_accuracy_histories, axis=0)
        plt.plot(range(1, len(val_accuracy_avg) + 1), val_accuracy_avg, label=f'Taux d\'apprentissage={lr}')
    plt.title(f'{model_name} - Accuracy de validation moyenne\n par epochs pour différents taux d\'apprentissage')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy de validation')
    plt.legend()
    plt.tight_layout() 
    plt.savefig(f'graphiques/{model_name}_LR_accuracy.png') 
    plt.close() 

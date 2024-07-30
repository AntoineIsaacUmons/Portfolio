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
from sklearn.metrics import recall_score
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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.metrics import precision_score


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



# ################## POIDS #####################

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

    return model, base_model


def build_model_InceptionResNetV2_opt(optimizer):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(taille_image, taille_image, 3))
    
    regularizer = l2(0.005)
    
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

    return model, base_model


def unfreeze_model_layers(base_model, n_layers):
    if n_layers > 0:
        for layer in base_model.layers[-n_layers:]:
            layer.trainable = True

models = [
    {'name': 'Xception', 'build_fn': build_model_xception_opt, 'unfreeze_layers_list': [0, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 125, 132]},
    {'name': 'InceptionResNetV2', 'build_fn': build_model_InceptionResNetV2_opt, 'unfreeze_layers_list': [0, 162, 505, 740, 780]},
]

def get_optimizer_SGD(lr):
    return SGD(learning_rate=lr, momentum=0.9)

def get_optimizer_Adam(lr):
    return Adam(learning_rate=lr)

n_splits = 4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {}

for model_info in models:
    model_name = model_info['name']
    build_fn = model_info['build_fn']
    for unfreeze_layers in model_info['unfreeze_layers_list']:
        key = f"{model_name}_Unfreeze{unfreeze_layers}"
        results[key] = {'accuracies': [], 'auc_rocs': [], 'losses': []}

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
                optimiseur = get_optimizer_SGD(0.0005)
            else:
                nb_augment = 10
                batch = 100
                optimiseur = get_optimizer_Adam(0.0001)

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


            model, base_model = build_fn(optimiseur)
            unfreeze_model_layers(base_model, unfreeze_layers)
            model.compile(optimizer=optimiseur, loss='binary_crossentropy', metrics=['accuracy'])


            model.fit(X_augmented, y_augmented, epochs=12, batch_size=batch, verbose=1)

            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            accuracy = accuracy_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, model.predict(X_test))
            loss, _ = model.evaluate(X_test, y_test, verbose=1)

            results[key]['accuracies'].append(accuracy)
            results[key]['auc_rocs'].append(auc_roc)
            results[key]['losses'].append(loss)

print("\nRÉSULTATS MOYENS APRÈS TOUS LES PLIS")
for key, metrics in results.items():
    print(f"\nModèle et Niveau de Dégel: {key}")
    for metric_name, values in metrics.items():
        moyenne = np.mean(values)
        ic = sem(values) * t.ppf((1 + 0.95) / 2, len(values) - 1)
        print(f"{metric_name}: Moyenne={moyenne:.3f}, IC 95%={moyenne - ic:.3f} à {moyenne + ic:.3f}")

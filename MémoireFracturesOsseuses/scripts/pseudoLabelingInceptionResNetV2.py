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
from tensorflow.keras.metrics import Recall, Precision


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



df_clean = pd.DataFrame(columns=['image'])
df_clean['image'] = df_train['image'].apply(pipeline_pretraitement)
df_clean['class'] = df_train['class']

X_train_chwapi = np.stack(df_clean['image'].values)
y_train_chwapi = df_clean['class'].map({'Fractured': 1, 'Non_Fractured': 0}).values


indices_train = [64, 16, 1, 19, 43, 4, 28, 8, 58, 63, 65, 31, 14, 21, 39, 56, 18, 40, 44, 73, 34, 7, 32, 48, 54, 33, 60, 36, 57, 12, 49, 53, 71, 10, 9, 46, 26, 25, 50, 38, 62, 20, 74, 0, 51, 5, 6, 15, 2, 35, 67, 22, 66, 70, 41, 45]
indices_val = [24, 42, 30, 72, 37]
indices_test = [47, 69, 29, 52, 27, 13, 68, 55, 17, 59, 61, 11, 23, 3]

X_train = X_train_chwapi[indices_train]
y_train = y_train_chwapi[indices_train]

X_val = X_train_chwapi[indices_val]
y_val = y_train_chwapi[indices_val]

X_test = X_train_chwapi[indices_test]
y_test = y_train_chwapi[indices_test]



df_non_labelised_clean = pd.DataFrame(columns=['image'])
df_non_labelised_clean['image'] = df_non_labelised['image'].apply(pipeline_pretraitement)

X_train_non_labelised = np.stack(df_non_labelised_clean['image'].values)




def evaluate_model(model, X_val, y_val):
    y_pred_proba = model.predict(X_val).ravel()
    y_val = tf.cast(y_val, dtype=tf.float32)
    loss = tf.keras.losses.binary_crossentropy(y_val, y_pred_proba)
    loss_scalar = tf.reduce_mean(loss).numpy()
    
    return loss_scalar

def redimensionner_images(images, taille=(taille_image, taille_image)):
    images_redimensionnees = []
    for image in images:
        
        image_redim = cv2.resize(image, taille, interpolation=cv2.INTER_AREA)

        images_redimensionnees.append(image_redim)

    return np.array(images_redimensionnees)



train_accuracy = []
val_accuracy = []
train_recall = []
val_recall = []
train_precision = []
val_precision = []
train_f1_scores = []
val_f1_scores = []


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity




seuil_confidence = 0.99

original_label_weight = 1.0

epochs = 100
batch = 100

chemin_model = 'modeles/InceptionResNetV2_model_75_8.h5'

model_with_pseudo_label = build_model_InceptionResNetV2_opt(Adam(learning_rate=0.005))
model_with_pseudo_label.load_weights(chemin_model)


model_with_pseudo_label.compile(optimizer=Adam(learning_rate=5e-03), loss='binary_crossentropy',  metrics=['accuracy', Recall(name='recall'), Precision(name='precision')])

all_X_train_with_pseudo_label = np.array(X_train)
all_y_train_with_pseudo_label = np.array(y_train)

images_non_labelisees_restant = df_non_labelised_clean['image'].tolist()
X_non_labelised = redimensionner_images(images_non_labelisees_restant)


best_train_loss = []
val_loss = []




predictions_non_labelisees = model_with_pseudo_label.predict(X_non_labelised)

iteration = 1

while iteration <= 3:

    indices_confiance_fractures = [i for i, p in enumerate(predictions_non_labelisees) if p >= 0.5]
    indices_confiance_non_fractures = [i for i, p in enumerate(predictions_non_labelisees) if p < 0.5]

    indices_fractures_tries = sorted(indices_confiance_fractures, key=lambda x: predictions_non_labelisees[x], reverse=True)
    indices_non_fractures_tries = sorted(indices_confiance_non_fractures, key=lambda x: 1-predictions_non_labelisees[x], reverse=True)

    indices_ajoutes_fractures = [i for i in indices_fractures_tries if predictions_non_labelisees[i] >= seuil_confidence]
    indices_ajoutes_non_fractures = [i for i in indices_non_fractures_tries if predictions_non_labelisees[i] < (1 - seuil_confidence)]


    confiance_moyenne_fractures = np.mean(predictions_non_labelisees[indices_ajoutes_fractures]) if indices_ajoutes_fractures else 0
    confiance_moyenne_non_fractures = np.mean(1 - predictions_non_labelisees[indices_ajoutes_non_fractures]) if indices_ajoutes_non_fractures else 0

    print(f"Confiance moyenne pour les images avec fractures ajoutees : {confiance_moyenne_fractures:.4f}")
    print(f"Confiance moyenne pour les images sans fractures ajoutees : {confiance_moyenne_non_fractures:.4f}")


    all_X_train_with_pseudo_label = np.array(X_train)
    all_y_train_with_pseudo_label = np.array(y_train)

    sample_weights = np.ones(len(all_X_train_with_pseudo_label)) * original_label_weight

    ################ Augmentation uniquement images de base #####################
    augmented_images, augmented_labels, augmented_weights = [], [], []

    for i in range(len(all_X_train_with_pseudo_label)):
        image = all_X_train_with_pseudo_label[i:i+1] 
        label = all_y_train_with_pseudo_label[i:i+1]  
        source_weight = sample_weights[i]

        for _ in range(4):  
            augmented_image, _ = next(datagen.flow(image, label, batch_size=1))
            augmented_images.append(augmented_image[0])
            augmented_labels.append(label[0])
            augmented_weights.append(source_weight)



    all_X_train_with_pseudo_label = np.concatenate([all_X_train_with_pseudo_label, np.array(augmented_images)])
    all_y_train_with_pseudo_label = np.concatenate([all_y_train_with_pseudo_label, np.array(augmented_labels)])
    sample_weights = np.concatenate([sample_weights, np.array(augmented_weights)])


    for indice in indices_ajoutes_fractures + indices_ajoutes_non_fractures:
        pseudo_label = 1 if indice in indices_ajoutes_fractures else 0
        all_X_train_with_pseudo_label = np.append(all_X_train_with_pseudo_label, [X_non_labelised[indice]], axis=0)
        all_y_train_with_pseudo_label = np.append(all_y_train_with_pseudo_label, [pseudo_label])
        
        confiance = predictions_non_labelisees[indice] if pseudo_label == 1 else 1 - predictions_non_labelisees[indice]
        poids_modifie = 1 * confiance
        sample_weights = np.append(sample_weights, [poids_modifie])

    # Calcul du desequilibre des classes apres ajout
    nombre_fractures = np.sum(all_y_train_with_pseudo_label == 1)
    nombre_non_fractures = np.sum(all_y_train_with_pseudo_label == 0)

    print(f"Avant augmentation : Fractures : {len(indices_ajoutes_fractures)} Non fractures : {len(indices_ajoutes_non_fractures)}")


    ################ Oversampling #####################
    if nombre_fractures != nombre_non_fractures:
        class_to_augment = 1 if nombre_fractures < nombre_non_fractures else 0
        images_to_generate = abs(nombre_fractures - nombre_non_fractures)
        X_class = all_X_train_with_pseudo_label[all_y_train_with_pseudo_label == class_to_augment]

        X_generated = []
        sample_weights_generated = [] 

        original_weights = sample_weights[all_y_train_with_pseudo_label == class_to_augment]

        for _ in range(images_to_generate):
            index_to_use = _ % len(X_class)
            X_batch = datagen.flow(X_class[index_to_use:index_to_use+1], batch_size=1, seed=42)
            X_generated.append(X_batch[0][0])
            sample_weights_generated.append(original_weights[index_to_use])

        X_generated = np.array(X_generated)
        y_generated = np.ones(len(X_generated)) * class_to_augment

        all_X_train_with_pseudo_label = np.concatenate([all_X_train_with_pseudo_label, X_generated])
        all_y_train_with_pseudo_label = np.concatenate([all_y_train_with_pseudo_label, y_generated])
        sample_weights = np.concatenate([sample_weights, sample_weights_generated])


    if(len(all_X_train_with_pseudo_label) > 1000):
        batch = 128
    else:
        batch = 100
    
        
    print(f"Nombre d'images ajoutees: {indices_ajoutes_fractures + indices_ajoutes_non_fractures}")
    print(f"Nombre total d'images: {len(all_X_train_with_pseudo_label)}")

    nombre_fractures = np.sum(all_y_train_with_pseudo_label == 1)
    nombre_non_fractures = np.sum(all_y_train_with_pseudo_label == 0)
    print(f"Nombre d'images avec fractures : {nombre_fractures}")
    print(f"Nombre d'images sans fractures : {nombre_non_fractures}")

    print(f"Nombre total d'images non labellisees restantes : {len(X_non_labelised)}")


    


    ################ Augmentation pour toutes #####################
    augmented_images, augmented_labels, augmented_weights = [], [], []
    for i in range(len(all_X_train_with_pseudo_label)):
        image = all_X_train_with_pseudo_label[i:i+1]
        label = all_y_train_with_pseudo_label[i:i+1]
        source_weight = sample_weights[i] 

        for _ in range(2):  
            augmented_image, _ = next(datagen.flow(image, label, batch_size=1))
            augmented_images.append(augmented_image[0])
            augmented_labels.append(label[0])
            augmented_weights.append(source_weight)  

    X_augmented = np.concatenate([all_X_train_with_pseudo_label, np.array(augmented_images)])
    y_augmented = np.concatenate([all_y_train_with_pseudo_label, np.array(augmented_labels)])
    sample_weights_augmented = np.concatenate([sample_weights, np.array(augmented_weights)])



    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        
    model_with_pseudo_label.compile(optimizer=Adam(learning_rate=5e-04), loss='binary_crossentropy', metrics=['accuracy', Recall(name='recall'), Precision(name='precision')])


    class_weights = {0: 1, 1: 2.5}
    sample_weights_adjusted = np.array([class_weights[y] for y in y_augmented]) * sample_weights_augmented

    history = model_with_pseudo_label.fit(X_augmented, y_augmented,
              validation_data=(X_val, y_val), 
              epochs=epochs, 
              verbose=1, 
              batch_size=batch,
              callbacks=[early_stopping],
              sample_weight=sample_weights_adjusted
              )
    
    val_performance = evaluate_model(model_with_pseudo_label, X_val, y_val)


    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_rec = history.history['recall'][-1]
    val_rec = history.history['val_recall'][-1]
    train_prec = history.history['precision'][-1]
    val_prec = history.history['val_precision'][-1]

    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)
    train_recall.append(train_rec)
    val_recall.append(val_rec)
    train_precision.append(train_prec)
    val_precision.append(val_prec)

    train_f1 = f1_score(train_prec, train_rec)
    val_f1 = f1_score(val_prec, val_rec)
    train_f1_scores.append(train_f1)
    val_f1_scores.append(val_f1)


    best_train_loss.append(min(history.history['loss']))
    val_loss.append(val_performance)


    
    y_pred = (model_with_pseudo_label.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    
    results = model_with_pseudo_label.evaluate(X_test, y_test, verbose=1)
    loss = results[0]
    accuracy = results[1]
    auc_roc = roc_auc_score(y_test, model_with_pseudo_label.predict(X_test))
    sensitivity = recall_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(precision, sensitivity)
    
    print(f"Affichage a chaque etape comme backup iteration : {iteration}")
    print("Matrice de Confusion :")
    print(cm)
    print(f"Perte : {loss:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC-ROC : {auc_roc:.4f}")
    print(f"Sensibilite : {sensitivity:.4f}")
    print(f"Specificite : {specificity:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"f1 : {f1:.4f}")


    if seuil_confidence > 0.95:
      seuil_confidence = seuil_confidence - 0.01
    
    if iteration > 2:
      model_with_pseudo_label.save(f"modelesFinaux/model_with_pseudo_label4_Inception_98_002_iteration{iteration}.h5")


    iteration += 1



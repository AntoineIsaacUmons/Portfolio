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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, Dropout
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



taille_image = 256

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

    return model


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

    return model





df_train['image'] = df_train['image'].apply(pipeline_pretraitement)

X_test = np.stack(df_train['image'].values)
y_test = df_train['class'].map({'Fractured': 1, 'Non_Fractured': 0}).values




df_non_labelised_clean = pd.DataFrame(columns=['image'])
df_non_labelised_clean['image'] = df_non_labelised['image'][:100].apply(pipeline_pretraitement)

X_train_non_labelised = np.stack(df_non_labelised_clean['image'].values)



input_shape = (taille_image, taille_image, 3)


input_img = Input(shape=input_shape)

def unet_autoencoder(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)

    u6 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(concatenate([u6, c4]))
    u7 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(concatenate([u7, c3]))
    u8 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(concatenate([u8, c2]))
    u9 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(concatenate([u9, c1]))

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
    
    
autoencoder = unet_autoencoder()
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


X_train, X_val = train_test_split(X_train_non_labelised, test_size=0.2)

train_generator = datagen.flow(X_train, X_train, batch_size=64)
val_generator = datagen.flow(X_val, X_val, batch_size=32)



history = autoencoder.fit(train_generator,
                epochs=100,
                shuffle=True,
                validation_data=val_generator)
                
                
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.plot(loss, label='Perte d\'entrainement')
plt.plot(val_loss, label='Perte de validation')
plt.xlabel('Epoques')
plt.ylabel('Perte')
plt.title('Courbes de Perte d\'Entrainement et de Validation')
plt.legend()
plt.savefig('graphiques/auto.png')
plt.close()
                
                
X_val, X_final_test, y_val, y_final_test = train_test_split(X_test, y_test, test_size=0.9)


decoded_imgs_val = autoencoder.predict(X_val)
reconstruction_error_val = np.mean(np.square(X_val - decoded_imgs_val), axis=(1, 2, 3))

def find_best_threshold(reconstruction_error, y_true):
    percentiles = np.percentile(reconstruction_error, np.arange(0, 101, 2))
    scores = []
    best_percentile = 0
    
    for i, threshold in enumerate(percentiles):
        y_pred = reconstruction_error > threshold
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        scores.append((threshold, f1, precision, recall))
        if f1 == max(scores, key=lambda x: x[1])[1]:
            best_percentile = np.arange(0, 101, 2)[i]
    
    best_threshold, best_f1, _, _ = max(scores, key=lambda x: x[1])
    
    for threshold, f1, precision, recall in scores:
        print(f"Seuil: {threshold:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Rappel: {recall:.4f}")
    
    return best_threshold, best_f1, best_percentile

best_threshold, best_f1, best_percentile = find_best_threshold(reconstruction_error_val, y_val)
print(f"Meilleur seuil: {best_threshold:.4f}, avec F1-score: {best_f1:.4f}, correspondant au percentile: {best_percentile}")



decoded_imgs_test = autoencoder.predict(X_final_test)
reconstruction_error_test = np.mean(np.square(X_final_test - decoded_imgs_test), axis=(1, 2, 3))

y_pred_test = reconstruction_error_test > best_threshold

conf_matrix = confusion_matrix(y_final_test, y_pred_test)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_final_test, y_pred_test)
print("Classification Report:\n", class_report)



for i in range(3): 
    plt.figure(figsize=(5, 5))
    plt.imshow(np.squeeze(decoded_imgs_test[i]), cmap='gray')
    plt.axis('off')
    plt.savefig(f'images/predicted_image_{i + 1}.png')
    plt.close()
    



threshold = np.percentile(reconstruction_error_test, 98)

print("threshold : ", threshold)

y_pred_test = reconstruction_error_test > threshold

conf_matrix = confusion_matrix(y_final_test, y_pred_test)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_final_test, y_pred_test)
print("Classification Report:\n", class_report)










X_val_normal = X_val[y_val == 0]
X_val_anomaly = X_val[y_val == 1]

augmentations_per_image = 5

def augment_data(X_data, y_label, augmentations_per_image):
    X_data_augmented = []
    y_data_augmented = []
    for i in range(len(X_data)):
        x = X_data[i]
        y = y_label
        x = np.expand_dims(x, 0) 
        j = 0
        for batch in datagen.flow(x, batch_size=1):
            X_data_augmented.append(batch[0])
            y_data_augmented.append(y)
            j += 1
            if j >= augmentations_per_image:
                break
    return np.array(X_data_augmented), np.array(y_data_augmented)

X_val_normal_augmented, y_val_normal_augmented = augment_data(X_val_normal, 0, augmentations_per_image)
X_val_anomaly_augmented, y_val_anomaly_augmented = augment_data(X_val_anomaly, 1, augmentations_per_image)

X_val_augmented = np.concatenate([X_val_normal_augmented, X_val_anomaly_augmented])
y_val_augmented = np.concatenate([y_val_normal_augmented, y_val_anomaly_augmented])

print(f"Nombre total d'images augmentees : {len(X_val_augmented)}")





X_val_augmented, y_val_augmented = shuffle(X_val_augmented, y_val_augmented, random_state=42)

def custom_loss(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    
    anomaly_penalty = 10.0 
    loss = tf.where(tf.equal(y_true[:, 0, 0, 0], 0), reconstruction_loss, anomaly_penalty * reconstruction_loss)
    return tf.reduce_mean(loss)

autoencoder.compile(optimizer='adam', loss=custom_loss)

autoencoder.fit(X_val_augmented, X_val_augmented,
                                      epochs=20,
                                      batch_size=32,
                                      shuffle=True)



decoded_imgs_test = autoencoder.predict(X_final_test)
reconstruction_error_test = np.mean(np.square(X_final_test - decoded_imgs_test), axis=(1, 2, 3))



threshold = np.percentile(reconstruction_error_test, 98)

print("threshold : ", threshold)

y_pred_test = reconstruction_error_test > threshold

conf_matrix = confusion_matrix(y_final_test, y_pred_test)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_final_test, y_pred_test)
print("Classification Report:\n", class_report)









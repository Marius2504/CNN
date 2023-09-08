import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)

# ct = 0 - citim si etichetele
# ct = 1 - nu citim etichetele
def citire(text, ct):
    train_image = []
    images = []
    labels = []

    linii = np.loadtxt(text, 'str')

    for linie in linii:
        if ct == 0:
            linie = linie.split(",")
            images.append(linie[0])
            labels.append(linie[1])
        else:
            images.append(linie)
            labels.append(0)

    images = images[1:]  # eliminam primul element deoarece in acest caz este 'id'
    labels = labels[1:]
    n = len(labels)

    # transformam labels in lista de intregi
    for i in range(n):
        labels[i] = int(labels[i])

    for imagine in images:
        if ct == 0:
            img = cv2.imread(f'train+validation\{imagine}')
        else:
            img = cv2.imread(f'test\{imagine}')
        train_image.append(img)

    train_image = np.array(train_image).astype('float32')
    labels = np.array(labels)
    return train_image, labels


def AI():
    # construim un model CNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1), #renunt la 10% din neuroni
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        # numarul de filtre este ales in functie de acuratetea obtinuta

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(7, activation='softmax')
        # numarul de unitati din layer-ul final coincide cu numarul de clase
        # folosesc functia de activare 'relu' deoarece acuratetea rezultata este mult mai buna ca in cazul altor functii
    ])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
    # functia de early stopping permite programului sa se opreasca in cazul in care nu evoluaza dpdv al acuratetii

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-5),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # hiperparametrii lr si decay au fost alesi astfel incat acuratetea este maxima
    model.fit(it, steps_per_epoch=250, epochs=20)
    # modelul primeste datele generate

    model.fit(pixelList_train, labels_train, epochs=10, validation_data=(pixelList_validation, labels_validation),
              callbacks=[early_stopping])
    # setez datele de validare

    score, acc = model.evaluate(pixelList_validation, labels_validation)

    pred = model.predict(pixelList_validation)
    global cnn_predictions
    for lista in pred:
        val = max(lista)
        for i in range(len(lista)):
            if lista[i] == val:
                cnn_predictions.append(i)
    return model.predict(pixelList_test)


def rewrite():
    # functie care are rolul de a suprascrie documentul "sample_submission"
    f = open("sample_submission.txt")
    linii = f.readlines()
    linii_noi = ['id,label\n']

    for i in range(1, len(pixelList_test) + 1):
        linie = linii[i]
        linie = linie.split(",")
        linii_noi.append(linie[0] + f",{predicted_labels[i - 1]}\n")

    f.close()
    f = open("sample_submission.txt", "w")
    f.write("".join(linii_noi))
    f.close()


def SVM_model():
    svm_model = SVC()
    pixelList_train_flat = pixelList_train.reshape(pixelList_train.shape[0], -1)
    pixelList_validation_flat = pixelList_validation.reshape(pixelList_validation.shape[0], -1)
    svm_model.fit(pixelList_train_flat, labels_train)
    return svm_model.predict(pixelList_validation_flat)


def RandomForest_model():
    rf_model = RandomForestClassifier()
    pixelList_train_flat = pixelList_train.reshape(pixelList_train.shape[0], -1)
    pixelList_validation_flat = pixelList_validation.reshape(pixelList_validation.shape[0], -1)
    rf_model.fit(pixelList_train_flat, labels_train)
    return rf_model.predict(pixelList_validation_flat)


def KNN_model():
    knn_model = KNeighborsClassifier(n_neighbors=5)  # Inițializăm modelul KNN cu numărul de vecini dorit
    pixelList_train_flat = pixelList_train.reshape(pixelList_train.shape[0], -1)
    pixelList_validation_flat = pixelList_validation.reshape(pixelList_validation.shape[0], -1)
    knn_model.fit(pixelList_train_flat, labels_train)
    return knn_model.predict(pixelList_validation_flat)


def ResNet_model():
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = Sequential()
    model.add(resnet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # Numărul de clase (7 în acest caz)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    pixelList_train_resize = resize_images(pixelList_train,(32,32))
    pixelList_validation_resize = resize_images(pixelList_validation,(32,32))
    model.fit(pixelList_train_resize, labels_train, epochs=10, batch_size=32,validation_data=(pixelList_validation_resize, labels_validation))
    return model.predict(pixelList_validation_resize)


def resize_images(images, size):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, size)
        resized_images.append(resized_image)
    return np.array(resized_images)


pixelList_train, labels_train = citire("train.txt", 0)
pixelList_test, labels_test = citire("test.txt", 1)
pixelList_validation, labels_validation = citire("validation.txt", 0)

# standardizare
train_std = np.std(pixelList_train, axis=0)
train_mean = np.mean(pixelList_train, axis=0)
test_std = np.std(pixelList_test, axis=0)
test_mean = np.mean(pixelList_test, axis=0)
valid_std = np.std(pixelList_validation, axis=0)
valid_mean = np.mean(pixelList_validation, axis=0)

# scadem media si impartim la valorile standardizate
pixelList_train = (pixelList_train - train_mean) / train_std
pixelList_test = (pixelList_test - test_mean) / test_std
pixelList_validation = (pixelList_validation - valid_mean) / valid_std

# generam valori aditionale prin ImageDataGenerator
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator()
it = datagen_train.flow(pixelList_train, labels_train, batch_size=32)
datagen_train.fit(pixelList_train)

# validare-calculare acuratete
cnn_predictions = []
svm_predictions = []
rf_predictions = []
resnet_predictions = []
cnn_accuracy = 0
svm_accuracy= 0
rf_accuracy= 0
knn_accuracy= 0
resnet_accuracy= 0
#resnet_predictions = ResNet_model()
#resnet_accuracy = np.mean(resnet_predictions == labels_validation) * 100
#print("Acuratețea modelului ResNet: {:.2f}%".format(resnet_accuracy))
#print("aici2")



# rewrite()

def initialize():
    global svm_predictions, rf_predictions, knn_predictions, resnet_predictions
    global cnn_accuracy,svm_accuracy,rf_accuracy,knn_accuracy,resnet_accuracy
    predicted_labels = AI()
    pp = []
    # obtinem label-ul cu predicitia maxima
    for lista in predicted_labels:
        val = max(lista)
        for i in range(len(lista)):
            if lista[i] == val:
                pp.append(i)

    svm_predictions = SVM_model()
    rf_predictions = RandomForest_model()
    knn_predictions = KNN_model()
    #resnet_predictions = ResNet_model()
    # Acuratețea modelului CNN
    cnn_accuracy = np.mean(cnn_predictions == labels_validation) * 100
    # Acuratețea modelului SVM
    svm_accuracy = np.mean(svm_predictions == labels_validation) * 100
    # Acuratețea modelului RandomForest
    rf_accuracy = np.mean(rf_predictions == labels_validation) * 100
    # Acuratetea modelului KNN
    knn_accuracy = np.mean(knn_predictions == labels_validation) * 100
    # Acuratetea modelului ResNet
    #resnet_accuracy = np.mean(resnet_predictions == labels_validation) * 100


def bestModel():
    # Comparăm acuratețea modelelor
    global best
    best = max(cnn_accuracy, svm_accuracy, rf_accuracy, knn_accuracy)
    best_model = ""
    if cnn_accuracy == best:
        best_model = "CNN"
    elif svm_accuracy == best:
        best_model = "SVM"
    elif rf_accuracy == best:
        best_model = "Random Forest"
    else:
        best_model = "KNN"
    return "Cel mai bun model este: " + best_model


@app.route("/cnn_accuracy")
def CNN_ACC():
    return "Acuratețea modelului CNN: {:.2f}%".format(cnn_accuracy), 200


@app.route("/svm_accuracy")
def SVM_ACC():
    return "Acuratețea modelului SVM: {:.2f}%".format(svm_accuracy)


@app.route("/rf_accuracy")
def RF_ACC():
    return "Acuratețea modelului RandomForest: {:.2f}%".format(rf_accuracy)


@app.route("/knn_accuracy")
def KNN_ACC():
    return "Acuratețea modelului KNN: {:.2f}%".format(knn_accuracy)


@app.route("/resnet_accuracy")
def resnet_ACC():
    return "Acuratețea modelului ResNet: {:.2f}%".format(resnet_accuracy)



@app.route("/bestModel")
def BEST_ACC():
    return bestModel(), 200



@app.route("/conf_matrix_cnn")
def conf_matrix_cnn():
    return json.dumps(confusion_matrix(labels_validation, cnn_predictions).tolist())


@app.route("/conf_matrix_svm")
def conf_matrix_svm():
    return json.dumps(confusion_matrix(labels_validation, svm_predictions).tolist())


@app.route("/conf_matrix_rf")
def conf_matrix_rf():
    return json.dumps(confusion_matrix(labels_validation, rf_predictions).tolist())


@app.route("/conf_matrix_knn")
def conf_matrix_knn():
    return json.dumps(confusion_matrix(labels_validation, knn_predictions).tolist())



if __name__ == "main":
    initialize()
    app.run(debug=True)
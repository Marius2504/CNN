import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

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

    images = images[1:]  #eliminam primul element deoarece in acest caz este 'id'
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
    #construim un model CNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        #numarul de filtre este ales in functie de acuratetea obtinuta

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(7, activation='softmax')
        #numarul de unitati din layer-ul final coincide cu numarul de clase
        #folosesc functia de activare 'relu' deoarece acuratetea rezultata este mult mai buna ca in cazul altor functii
    ])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
    #functia de early stopping permite programului sa se opreasca in cazul in care nu evoluaza dpdv al acuratetii

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #hiperparametrii lr si decay au fost alesi astfel incat acuratetea este maxima
    model.fit_generator(it, steps_per_epoch=250,epochs=20)
    #modelul primeste datele generate
    model.fit(pixelList_train, labels_train, epochs=80,validation_data=(pixelList_validation,labels_validation), callbacks=[early_stopping])
    #setez datele de validare

    score, acc = model.evaluate(pixelList_validation, labels_validation)

    pred = model.predict(pixelList_validation)
    pp = []
    for lista in pred:
        val = max(lista)
        for i in range(len(lista)):
            if lista[i] == val:
                pp.append(i)
    print(confusion_matrix(labels_validation, pp))
    return model.predict(pixelList_test)


def rewrite():
    #functie care are rolul de a suprascrie documentul "sample_submission"
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


pixelList_train, labels_train = citire("train.txt", 0)
pixelList_test, labels_test = citire("test.txt", 1)
pixelList_validation, labels_validation = citire("validation.txt", 0)

#standardizare
train_std = np.std(pixelList_train, axis=0)
train_mean = np.mean(pixelList_train, axis=0)
test_std = np.std(pixelList_test, axis=0)
test_mean = np.mean(pixelList_test, axis=0)
valid_std = np.std(pixelList_validation,axis=0)
valid_mean = np.mean(pixelList_validation,axis=0)

#scadem media si impartim la valorile standardizate
pixelList_train = (pixelList_train - train_mean) / train_std
pixelList_test = (pixelList_test - test_mean) / test_std
pixelList_validation = (pixelList_validation - valid_mean) / valid_std

#generam valori aditionale prin ImageDataGenerator
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator()
it = datagen_train.flow(pixelList_train,labels_train,batch_size=32)
datagen_train.fit(pixelList_train)

predicted_labels = AI()
pp = []
for lista in predicted_labels:
    val = max(lista)
    for i in range(len(lista)):
        if lista[i] == val:
            pp.append(i)
#obtinem label-ul cu predicitia maxima
predicted_labels = pp
#print(confusion_matrix(labels_validation, predicted_labels) )
# print(acuratete())
rewrite()

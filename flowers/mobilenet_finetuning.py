import os
from keras.applications import MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# テキストから分類クラスを読み込む
def make_classes_data(label_file):
    with open(label_file, "r") as f:
        data = f.read()
    classes = data.split('\n')
    if '' in classes:
        remove_emptyclass(classes)
    return classes

# 分類クラスのテキストの最後に改行が入っていた場合空のクラスができるので削除
def remove_emptyclass(classes):
    if '' in classes:
        classes.remove('')
        remove_emptyclass(classes)

# 学習用画像が入っているフォルダから分類クラスのテキストを生成
def make_classes_fromdir(train_data_dir):
    classes = []
    p = Path(train_data_dir)
    folders = list(p.glob(u'**/'))
    folders.pop(0)
    for f in folders:
        dir_name = os.path.basename(f)
        classes.append(dir_name)
    return classes

# 損失の履歴をプロット
def plot_history_loss(history):
    plt.figure()
    plt.plot(history.history['loss'],"o-",color='b',ms=1.3,lw=0.5,label="loss",)
    plt.plot(history.history['val_loss'],"o-",color='r',ms=1.3,lw=0.5,label="val_loss",)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    # plt.show()

# 精度の履歴をプロット
def plot_history_acc(history):
    plt.figure()
    plt.plot(history.history['acc'],"o-",color='b',ms=1.3,lw=0.5,label="acc",)
    plt.plot(history.history['val_acc'],"o-",color='r',ms=1.3,lw=0.5,label="val_acc",)
    plt.title('model acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    # plt.show()

# 学習の履歴をテキストに保存
def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as f:
        f.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            f.write("%d\t%f\t%f\t%f\t%f\n" % (i+1, loss[i], acc[i], val_loss[i], val_acc[i]))

if __name__ == '__main__':
    # 学習用データとテスト用データ
    train_data_dir = 'dataset/train_images'
    test_data_dir = 'dataset/test_images'

    #分類クラスの読み込み
    if os.path.exists('classes.txt'):
        classes = make_classes_data('classes.txt')
    else:
        classes = make_classes_fromdir(train_data_dir)
        with open('classes.txt', 'w') as f:
            for c in classes:
                f.write('{}\n'.format(c))

    # 結果保存用フォルダ
    dt_now = datetime.now()
    result_dir = 'results/{}'.format(dt_now.strftime('%Y%m%d_%H%M'))
    os.makedirs(result_dir, exist_ok = True)

    # 学習用データとテスト用データのリスト
    train_p = Path(train_data_dir)
    train_pictures = list(train_p.glob(u'**/*.jpg'))
    test_p = Path(test_data_dir)
    test_pictures = list(test_p.glob(u'**/*.jpg'))

    # サンプル数
    nb_train_samples = len(train_pictures)
    nb_val_samples = len(test_pictures)
    # エポック数
    nb_epoch = 100

    # バッチサイズ
    batch_size = 32
    # 分類クラス数
    nb_classes = len(classes)

    # 画像の情報
    img_rows, img_cols = 150, 150
    channels = 3

    # VGG16モデルと学習済み重みをロード
    # 今回はfine-tuningなのでinclude_top=False (学習済みの全結合層を含まない)
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    mobilenet = MobileNet(include_top=False, weights='imagenet', input_shape = None)
    # mobilenet.summary()

    # 全結合層の新規構築
    x = mobilenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(nb_classes, activation = 'softmax')(x)

    # MobileNetとFCを接続
    model = Model(inputs = mobilenet.input, outputs = predictions)

    # 72層までをfreeze
    for layer in model.layers[:72]:
        layer.trainable = False

        # Batch Normalization の freeze解除
        if "bn" in layer.name:
            layer.trainable = True

    #73層以降は学習させる
    for layer in model.layers[72:]:
        layer.trainable = True

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.Adam(),
                  metrics = ['accuracy'])

    # Callback
    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        verbose = 1
    )

    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 3,
        verbose = 1
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.,
        zoom_range=0.,
        horizontal_flip=False)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_rows, img_cols),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    # Fine-tuning
    history = model.fit_generator(
        train_generator,
        #samples_per_epoch=nb_train_samples,
        steps_per_epoch=nb_train_samples/batch_size,
        epochs=nb_epoch,
        validation_data=test_generator,
        #nb_val_samples=nb_val_samples
        validation_steps=nb_val_samples/batch_size,
        callbacks = [early_stopping, reduce_lr]
        )

    plot_history_loss(history)
    plt.savefig(os.path.join(result_dir, 'loss_fig.png'))

    plot_history_acc(history)
    plt.savefig(os.path.join(result_dir, 'acc_fig.png'))

    model.save_weights(os.path.join(result_dir, 'mobilenet_flowers_finetuning.h5'))
    save_history(history, os.path.join(result_dir, 'mobilenet_flowers_history.txt'))


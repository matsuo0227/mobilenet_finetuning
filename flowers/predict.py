# -*- coding: utf-8 -*-
import os
import sys
from keras.applications import MobileNet
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.preprocessing import image
import numpy as np

if len(sys.argv) != 2:
    print("usage: python predict.py [filename]")
    sys.exit(1)

filename = sys.argv[1]
print('input:', filename)

result_dir = 'results'

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

if __name__ == '__main__':
    train_data_dir = 'dataset/train_images'

    #分類クラスの読み込み
    if os.path.exists('classes.txt'):
        classes = make_classes_data('classes.txt')
    else:
        classes = make_classes_fromdir(train_data_dir)
        with open('classes.txt', 'w') as f:
            for c in classes:
                f.write('{}\n'.format(c))

    nb_classes = len(classes)

    img_rows, img_cols = 150, 150
    channels = 3

    # MobileNetのモデル作成
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    mobilenet = MobileNet(include_top=False, weights='imagenet', input_shape = None)
    # mobilenet.summary()

    # 全結合層の新規構築
    x = mobilenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(nb_classes, activation = 'softmax')(x)

    # VGG16とFCを接続
    model = Model(inputs = mobilenet.input, outputs = predictions)

    # 学習済みの重みをロード
    model.load_weights(os.path.join(result_dir, 'mobilenet_flowers_finetuning.h5'))

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.Adam(),
                  metrics = ['accuracy'])
    # model.summary()

    # 画像を読み込んで4次元テンソルへ変換
    img = image.load_img(filename, target_size=(img_rows, img_cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
    # これを忘れると結果がおかしくなるので注意
    x = x / 255.0

    # クラスを予測
    # 入力は1枚の画像なので[0]のみ
    pred = model.predict(x)[0]
    print(pred)

     #予測確率が高いトップ5を出力
    top = nb_classes
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    for x in result:
        print(x)

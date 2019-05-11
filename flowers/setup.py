import shutil
import requests
import os
import random
import csv

# 連番画像の保存先
PIC_DIR = 'jpg'
# 学習用データとテスト用データの移動先
TRAIN_DIR = 'dataset/train_images'
TEST_DIR = 'dataset/test_images'

# ディレクトリ作成
os.makedirs(TRAIN_DIR, exist_ok = True)
os.makedirs(TEST_DIR, exist_ok = True)

# ラベル(クラス)と画像番号が書かれたcsv
f = open('labels.csv', 'r')

# csvのreader
reader = csv.reader(f)

for row in reader:
    start_num = int(row[0])
    end_num = int(row[1])
    class_name = row[2]

    class_dir_train = os.path.join(TRAIN_DIR, class_name)
    os.makedirs(class_dir_train, exist_ok = True)

    for n in range(start_num, end_num + 1):
        img_name = 'image_{}.jpg'.format(str(n).zfill(4))
        img_dir_src = os.path.join(PIC_DIR, img_name)
        img_dir_dst = os.path.join(class_dir_train, img_name)
        shutil.move(img_dir_src, img_dir_dst)

    # テスト用画像をランダムに10枚選択
    files = os.listdir(class_dir_train)
    random.shuffle(files)

    class_dir_test = os.path.join(TEST_DIR, class_name)
    os.makedirs(class_dir_test, exist_ok = True)

    for f in files[:10]:
        img_dir_src = os.path.join(class_dir_train, f)
        img_dir_dst = os.path.join(class_dir_test, f)
        shutil.move(img_dir_src, img_dir_dst)

# shutil.rmtree(PIC_DIR)

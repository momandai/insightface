import datetime
import os
import glob
import shutil
import tqdm
import cv2
import numpy as np
import json
from retinaface import RetinaFace

gen_type = os.environ.get("gen_type")

root = "/media/test/data/coco"
in_file = os.path.join(root, 'trainvalno5k.txt') if gen_type == "train" else os.path.join(root, '5k.txt')
output_json_file = os.path.join(root, 'train_datalist.json') if gen_type == "train" else os.path.join(root, 'test_datalist.json')
print("in file: {0}, out put json file: {1}".format(in_file, output_json_file))

thresh = 0.8
gpuid = 0
input_shape = 1024

def letterbox(img, new_shape=416, color=(128, 128, 128), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratiow, ratioh, dw, dh


if __name__ == '__main__':
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    datalist = []
    with open(in_file, 'r') as f:
        img_files = f.read().splitlines()
        for img_file in tqdm.tqdm(img_files):
            label_file = img_file.replace('images', 'labels').replace(os.path.splitext(img_file)[-1], '.txt')
            if os.path.isfile(label_file):
                with open(label_file, 'r') as f_label_file:
                    labels = []
                    boxes = []
                    x = np.array([x.split() for x in f_label_file.read().splitlines()], dtype=np.float32)
                    y = None
                    find_person = False
                    img = cv2.imread(img_file)
                    width = img.shape[1]
                    height = img.shape[0]
                    for i in range(x.shape[0]):
                        if x[i, 0] == 0:
                            find_person = True
                            person_cx = x[i, 1] * width
                            person_cy = x[i, 2] * height
                            person_width = x[i, 3] * width
                            person_height = x[i, 4] * height
                            person_x0 = int(person_cx - person_width/2)
                            person_y0 = int(person_cy - person_height/2)
                            person_x1 = int(person_cx + person_width/2)
                            person_y1 = int(person_cy + person_height/2)
                            labels.append(1)
                            boxes.append([person_x0, person_y0, person_x1, person_y1])
                    if find_person:
                        input_img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=input_shape, mode='square')
                        faces, landmarks = detector.detect(input_img, thresh, scales=[1.0], do_flip=False)
                        if faces.shape[0] != 0:
                            print('find', faces.shape[0], 'faces')
                            for j in range(faces.shape[0]):
                                # print('score', faces[i][4])
                                box = faces[j]
                                x0 = int((box[0]-padw)/ratiow)
                                x1 = int((box[2]-padw)/ratiow)
                                y0 = int((box[1]-padh)/ratioh)
                                y1 = int((box[3]-padh)/ratioh)
                                labels.append(2)
                                boxes.append([x0, y0, x1, y1])
                                # cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                                # cv2.rectangle(img, (person_x0, person_y0), (person_x1, person_y1), (0, 255, 255), 2)
                                # cv2.imshow("cam", img)
                                # cv2.waitKey(-1)
                    if boxes and labels:
                        datalist.append({"image_file": img_file, 'labels': labels, 'boxes': boxes})
    json.dump(datalist, open(os.path.join(root, output_json_file), 'w'))
    print("label json file generated, total image: {0}".format(len(datalist)))





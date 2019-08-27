import datetime
import os
import glob
import shutil
import tqdm
import cv2
import numpy as np
from retinaface import RetinaFace


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

gen_type = os.environ.get("gen_type")


in_file = "/home/neptune/github/coco/trainvalno5k.txt" if gen_type == "train" else "/home/neptune/github/coco/5k.txt"
output_label_dir = "/home/neptune/github/coco/person_face_labels" if gen_type == "train" else "/home/neptune/github/coco/person_face_labels_valid"
output_image_dir = "/home/neptune/github/coco/person_face_images" if gen_type == "train" else "/home/neptune/github/coco/person_face_images_valid"
output_abstract_file = "./coco_person_face.txt" if gen_type == "train" else "./coco_person_face_valid.txt"
output_abstract_file_np = None
if os.path.exists(output_label_dir):
    shutil.rmtree(output_label_dir)
os.makedirs(output_label_dir)

if os.path.exists(output_image_dir):
    shutil.rmtree(output_image_dir)
os.makedirs(output_image_dir)

thresh = 0.8
gpuid = 0
input_shape = 1024
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
with open(in_file, 'r') as f:
    img_files = f.read().splitlines()
    for img_file in tqdm.tqdm(img_files[0:100]):
        label_file = img_file.replace('images', 'labels').replace(os.path.splitext(img_file)[-1], '.txt')
        if os.path.isfile(label_file):
            with open(label_file, 'r') as f_label_file:
                x = np.array([x.split() for x in f_label_file.read().splitlines()], dtype=np.float32)
                y = None
                has_person = False
                for i in range(x.shape[0]):
                    if x[i, 0] == 0:
                        has_person = True
                        y = np.vstack((y, [x[i, :]])) if y is not None else np.array([x[i, :]])
                if has_person:
                    img = cv2.imread(img_file)
                    width = img.shape[1]
                    height = img.shape[0]
                    input_img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=input_shape, mode='square')
                    faces, landmarks = detector.detect(input_img, thresh, scales=[1.0], do_flip=False)
                    if faces is not None:
                        print('find', faces.shape[0], 'faces')
                        for i in range(faces.shape[0]):
                            # print('score', faces[i][4])
                            box = faces[i]
                            x0 = (box[0]-padw)/ratiow
                            x1 = (box[2]-padw)/ratiow
                            y0 = (box[1]-padh)/ratioh
                            y1 = (box[3]-padh)/ratioh
                            Cx = (x0 + x1)/2/width
                            Cy = (y0 + y1)/2/height
                            normalized_width = (x1 - x0) / width
                            normalized_height = (y1 - y0) / height
                            y = np.vstack((y, np.array([[1, Cx, Cy, normalized_width, normalized_height]])))
                    #     for i in range(y.shape[0]):
                    #         if y[i, 0] == 0:
                    #             color = (0, 0, 255)
                    #         else:
                    #             color = (255, 0, 0)
                    #         x0 = (y[i, 1] - y[i, 3]/2) * width
                    #         y0 = (y[i, 2] - y[i, 4]/2) * height
                    #         x1 = (y[i, 1] + y[i, 3]/2) * width
                    #         y1 = (y[i, 2] + y[i, 4]/2) * height
                    #         cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
                    # cv2.imshow("cam", img)
                    # cv2.waitKey(-1)

                    output_label_file = os.path.join(output_label_dir, os.path.basename(label_file))
                    output_image_file = os.path.join(output_image_dir, os.path.basename(img_file))
                    shutil.copy(img_file, output_image_file)
                    np.savetxt(output_label_file, y, fmt='%.6f')
                    output_abstract_file_np = np.vstack((output_abstract_file_np, [output_image_file])) \
                        if output_abstract_file_np is not None else np.array([output_image_file])

np.savetxt(output_abstract_file, output_abstract_file_np, fmt='%s')


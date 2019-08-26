import datetime
import os
import glob
import shutil
import tqdm
import cv2
import numpy as np
from retinaface import RetinaFace

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
                    scales = [1024, 1980]
                    img = cv2.imread(img_file)
                    width = img.shape[1]
                    height = img.shape[0]
                    im_shape = img.shape
                    target_size = scales[0]
                    max_size = scales[1]
                    im_size_min = np.min(im_shape[0:2])
                    im_size_max = np.max(im_shape[0:2])
                    # im_scale = 1.0
                    # if im_size_min>target_size or im_size_max>max_size:
                    im_scale = float(target_size) / float(im_size_min)
                    # prevent bigger axis from being more than max_size:
                    if np.round(im_scale * im_size_max) > max_size:
                        im_scale = float(max_size) / float(im_size_max)
                    scales = [im_scale]
                    flip = False
                    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
                    if faces is not None:
                        print('find', faces.shape[0], 'faces')
                        for i in range(faces.shape[0]):
                            # print('score', faces[i][4])
                            box = faces[i]
                            Cx = (box[0] + box[2])/2/width
                            Cy = (box[1] + box[3])/2/height
                            normalized_width = (box[2] - box[0]) / width
                            normalized_height = (box[3] - box[1]) / height
                            y = np.vstack((y, np.array([[1, Cx, Cy, normalized_width, normalized_height]])))
                        # for i in range(y.shape[0]):
                        #     if y[i, 0] == 0:
                        #         color = (0, 0, 255)
                        #     else:
                        #         color = (255, 0, 0)
                        #     x0 = (y[i, 1] - y[i, 3]/2) * width
                        #     y0 = (y[i, 2] - y[i, 4]/2) * height
                        #     x1 = (y[i, 1] + y[i, 3]/2) * width
                        #     y1 = (y[i, 2] + y[i, 4]/2) * height
                        #     cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
                    # cv2.imshow("cam", img)
                    # cv2.waitKey(-1)

                    output_label_file = os.path.join(output_label_dir, os.path.basename(label_file))
                    output_image_file = os.path.join(output_image_dir, os.path.basename(img_file))
                    shutil.copy(img_file, output_image_file)
                    np.savetxt(output_label_file, y, fmt='%.6f')
                    output_abstract_file_np = np.vstack((output_abstract_file_np, [output_image_file])) \
                        if output_abstract_file_np is not None else np.array([output_image_file])

np.savetxt(output_abstract_file, output_abstract_file_np, fmt='%s')


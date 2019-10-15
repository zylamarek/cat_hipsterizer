import cv2
import os
import pandas as pd
import numpy as np

dirname = 'test'
bbs_model_name = '2019_10_14_22_16_22'
base_path = '../cat-dataset/data/clean/%s' % dirname

img_size = 224
file_list = [f for f in sorted(os.listdir(base_path)) if f.endswith('.cat')]

dataset = {
    'imgs': [],
    'lmks': [],
    'landmarks_original': [],
    'img_max_size': [],
    'bbs': [],
    'transform': []
}


def resize_img(im):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    return new_im, ratio, top, left


data_bbs = np.load('dataset/%s.npy' % dirname, allow_pickle=True)
bbs_transforms = np.array(data_bbs.item().get('transform'))
data_pred_bbs = np.load('dataset/predicted_bbox_%s_%s.npy' % (bbs_model_name, dirname), allow_pickle=True)
predicted_bbs = np.array(data_pred_bbs.item().get('bbs'))

mses = []
for f, predicted_bb, bbs_transform in zip(file_list, predicted_bbs, bbs_transforms):
    # read landmarks
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
    landmarks = (pd_frame.as_matrix()[0][1:((pd_frame.shape[1] - 1) // 2) * 2 + 1]).reshape((-1, 2))
    landmarks_original = landmarks.copy()
    bb_original = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

    # transform bb to original coordinates
    bb = predicted_bb.reshape((-1, 2))
    bb -= np.array(bbs_transform[1:])
    bb = bb / bbs_transform[0]

    center = np.mean(bb, axis=0)
    face_size = np.max(np.diff(bb, axis=0))
    new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
    ]).astype(np.int)
    new_bb = np.clip(new_bb, 0, 99999)
    new_landmarks = landmarks - new_bb[0]

    # load image
    img_filename, ext = os.path.splitext(f)
    img = cv2.imread(os.path.join(base_path, img_filename))
    img_max_size = np.max(img.shape[:2])
    new_img = img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]

    mses.append(np.mean(np.square(bb - bb_original)))

    # resize image and relocate landmarks
    img, ratio, top, left = resize_img(new_img)
    new_landmarks = ((new_landmarks * ratio) + np.array([left, top])).astype(np.int)

    dataset['imgs'].append(img)
    dataset['lmks'].append(new_landmarks.flatten())
    dataset['landmarks_original'].append(landmarks_original.flatten())
    dataset['img_max_size'].append(img_max_size)
    dataset['transform'].append((ratio, left, top, new_bb[0]))

    # for l in new_landmarks:
    #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    # cv2.imshow('img', img)
    # if cv2.waitKey(0) == ord('q'):
    #   sys.exit(1)

print('bbox mse', np.mean(mses))
np.save('dataset/lmks_pred_%s_%s.npy' % (bbs_model_name, dirname), np.array(dataset))

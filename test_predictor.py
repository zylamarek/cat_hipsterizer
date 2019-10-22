import cv2
import os
import numpy as np
from keras.models import load_model
import datetime
from collections import defaultdict

dirname = 'test'
bbs_model_name = '2019_10_14_22_16_22'
lmks_model_name = '2019_10_14_23_50_08'
base_path = '../cat-dataset/data/clean/%s' % dirname

img_size = 224
file_list = [f for f in sorted(os.listdir(base_path)) if f.endswith('.cat')]
output_dir = os.path.join('output', datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
os.makedirs(output_dir)

data_lmks = np.load('dataset/lmks_pred_%s_%s.npy' % (bbs_model_name, dirname), allow_pickle=True)
x_lmks = np.array(data_lmks.item().get('imgs')).astype('float32') / 255.
y_lmks = np.array(data_lmks.item().get('lmks'))
landmarks_originals = np.array(data_lmks.item().get('landmarks_original'))
lmks_transforms = np.array(data_lmks.item().get('transform'))

lmks_model = load_model('models/%s.h5' % lmks_model_name)
predicted_lmks = lmks_model.predict(x_lmks, verbose=1)

metrics = defaultdict(list)
for landmarks_predicted, lmks_transform, landmarks_original, filename in \
        zip(predicted_lmks, lmks_transforms, landmarks_originals, file_list):
    landmarks_predicted = landmarks_predicted.reshape((-1, 2)) - np.array(lmks_transform[1:3])
    landmarks_predicted /= lmks_transform[0]
    landmarks_predicted += lmks_transform[3]

    landmarks_original = landmarks_original.reshape((-1, 2))
    bb_original = np.concatenate([np.min(landmarks_original, axis=0), np.max(landmarks_original, axis=0)])
    face_size = np.max(np.diff(bb_original.reshape((-1, 2)), axis=0))


    def get_mape(a, b):
        return np.mean(np.abs((landmarks_predicted[a: b + 1] - landmarks_original[a: b + 1]) / face_size * 100.))


    err = landmarks_predicted - landmarks_original
    metrics['mae'].append(np.mean(np.abs(err)))
    metrics['mse'].append(np.mean(np.square(err)))
    metrics['mspe'].append(np.mean(np.square(err / face_size * 100.)))
    mape = np.mean(np.abs(err / face_size * 100.))
    metrics['mape'].append(mape)
    metrics['mape eyes'].append(get_mape(0, 1))
    metrics['mape mouth'].append(get_mape(2, 2))
    metrics['mape ears'].append(get_mape(3, 8))

    img = cv2.imread(os.path.join(base_path, filename[:-4]))
    for i_l, (l_p, l_o) in enumerate(zip(landmarks_predicted.astype('int'), landmarks_original)):
        cv2.circle(img, center=tuple(l_p), radius=1, color=(255, 0, 0), thickness=2)
        cv2.putText(img, str(i_l), tuple(l_p), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.circle(img, center=tuple(l_o), radius=1, color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, '%.9f_%s' % (mape, filename[:-4])), img)

for name, vals in metrics.items():
    print('%s:\t%.2f' % (name, np.mean(vals)))
    if name.startswith('ms'):
        print('r%s:\t%.2f' % (name, np.sqrt(np.mean(vals))))

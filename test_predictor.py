import cv2
import os
import numpy as np
from keras.models import load_model
from collections import defaultdict

dirname = 'test'
bbs_model_name = '2019_10_14_22_16_22'
lmks_model_name = '2019_10_14_23_50_08'
base_path = '../cat-dataset/data/clean/%s' % dirname

img_size = 224
file_list = [f for f in sorted(os.listdir(base_path)) if f.endswith('.cat')]
os.makedirs('output', exist_ok=True)

data_lmks = np.load('dataset/lmks_pred_%s_%s.npy' % (bbs_model_name, dirname), allow_pickle=True)
x_lmks = np.array(data_lmks.item().get('imgs')).astype('float32') / 255.
y_lmks = np.array(data_lmks.item().get('lmks'))
landmarks_originals = np.array(data_lmks.item().get('landmarks_original'))
lmks_transforms = np.array(data_lmks.item().get('transform'))
img_max_sizes = np.array(data_lmks.item().get('img_max_size'))

lmks_model = load_model('models/%s.h5' % lmks_model_name)
predicted_lmks = lmks_model.predict(x_lmks, verbose=1)
mse_eval = lmks_model.evaluate(x_lmks, y_lmks, verbose=1)

mses = defaultdict(list)
for landmarks_predicted, lmks_transform, landmarks_original, img_max_size, filename in \
        zip(predicted_lmks, lmks_transforms, landmarks_originals, img_max_sizes, file_list):
    landmarks_predicted = landmarks_predicted.reshape((-1, 2)) - np.array(lmks_transform[1:3])
    landmarks_predicted /= lmks_transform[0]
    landmarks_predicted += lmks_transform[3]
    landmarks_predicted = landmarks_predicted
    landmarks_original = landmarks_original.reshape((-1, 2))


    def get_mse_normalized(a, b):
        return np.mean(np.square((landmarks_predicted[a: b + 1] - landmarks_original[a: b + 1]) / img_max_size))


    mse = np.mean(np.square(landmarks_predicted - landmarks_original))
    mses['all'].append(mse)
    mses['all normalized'].append(get_mse_normalized(0, 8))
    mses['eyes normalized'].append(get_mse_normalized(0, 1))
    mses['mouth normalized'].append(get_mse_normalized(2, 2))
    mses['ears normalized'].append(get_mse_normalized(3, 8))

    img = cv2.imread(os.path.join(base_path, filename[:-4]))
    for i_l, (l_p, l_o) in enumerate(zip(landmarks_predicted.astype('int'), landmarks_original)):
        cv2.circle(img, center=tuple(l_p), radius=1, color=(255, 0, 0), thickness=2)
        cv2.putText(img, str(i_l), tuple(l_p), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.circle(img, center=tuple(l_o), radius=1, color=(0, 255, 0), thickness=2)
    cv2.imwrite('output/%.5f.png' % mse, img)

print('mse all: %.7f' % np.mean(mses['all']))
for name, vals in mses.items():
    print('rmse %s: %.7f' % (name, np.sqrt(np.mean(vals))))
print('mse eval', mse_eval)
print('rmse eval', np.sqrt(mse_eval))

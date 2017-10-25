import cv2
from scipy import misc
import numpy as np

from face_database import face_process, get_face_vec, get_face_img

query_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/known_people_manually/Hillary/000010.jpg'

img = misc.imread(query_img_path)

misc.imshow(img)

img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

mini_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset//unknown_people/mini.jpg'
misc.imsave(mini_path, img)
misc.imshow(img)

args_image_size = 160

query_face_closeup, _, _ = get_face_img(np.atleast_1d(mini_path))
query_processed_face_ = face_process(query_face_closeup, False, False, args_image_size)
query_face_vector = get_face_vec(query_processed_face_)


import pickle
import os

classifier_filename_exp = os.path.expanduser('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/political_classifier.pkl')
print('Testing classifier')
with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)

print('Loaded classifier model from file "%s"' % classifier_filename_exp)

predictions = model.predict_proba(query_face_vector)
best_class_indices = np.argmax(predictions, axis=1)
best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

for i in range(len(best_class_indices)):
    print('%s: %.3f' % (class_names[best_class_indices[i]], best_class_probabilities[i]))

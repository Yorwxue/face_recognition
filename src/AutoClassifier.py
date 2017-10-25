# import cv2
from scipy import misc
import numpy as np
import pickle
import os
from face_database import face_process, get_face_vec, get_face_img, ImageClass, get_image_paths_and_labels
import facenet
import tensorflow as tf
import math
from shutil import copyfile


# automate class all crawler images into directory of people

classifier_filename_exp = os.path.expanduser('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/political_classifier.pkl')
model = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/20170512-110547.pb'
image_size = 160
seed = 666
batch_size = 1000
# --

images_directory = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/crawlar_img'
closeups_directory = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/mtcnnpy_160/crawlar_mtcnnpy_160'
target_directory = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/character'
root_directory ='/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master'

# face processing
# align
# os.system('export PYTHONPATH=/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src')
# os.system('for N in {1..4}; do python3.6 %s/src/align/align_dataset_mtcnn.py %s %s --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done' % (root_directory, images_directory, closeups_directory))
# exit()


#
with tf.Graph().as_default():
    with tf.Session() as sess:
        np.random.seed(seed=seed)

        # get dataset
        dataset = facenet.get_dataset(closeups_directory)

        # Check that there are at least one training image per class
        for cls in dataset:
            assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

        # label
        paths, labels = get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(model)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        # load classify model
        print('Classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for i in range(len(best_class_indices)):
            print(paths[i].split('/')[-2], paths[i].split('/')[-1],
                  '%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)
        # copyfile(src, dst)

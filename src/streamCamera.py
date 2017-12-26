from __future__ import print_function

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
import pickle

import align.detect_face
import facenet
from PIL import Image, ImageDraw, ImageFont
from face_database import faceDB, realTimeFaceRecognition, drawBoundaryBox

# python3.6 src/streamcamera.py {model_path}
# ex: python3.6 src/streamCamera.py /media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/20170512-110547.pb

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 32
image_size = 160
buffer_size = 1
# classifier_filename_exp = os.path.expanduser('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/political_classifier.pkl')
db_name = 'sir_db'

# font style
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 255, 255)
lineType = 2

# test_video = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/action_recognition_dataset/UCF101/v_Archery_g01_c07.avi'
test_video = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/unknown_people/1-woPojJbd6lT7CFZ9lHRVDw.gif'


def puttext_in_chinese(img, text, location):
    # cv2 to pil
    cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img)

    # text
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
    draw.text(location, text, (0, 0, 255), font=font)  # third parameter is color

    # pil to cv2
    cv2_text_im = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_text_im


def main(args):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        # load the align model
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        with tf.Session() as sess:

            # load the detect model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # load the classify model
            # with open(classifier_filename_exp, 'rb') as infile:
            #     (model, class_names) = pickle.load(infile)
            db = faceDB(db_name)

            # -------------------------------------------------------------------------
            cap = cv2.VideoCapture(test_video)

            # frame_buffer = list()
            # stack = 0
            marked_frameList = list()

            while True:
                ret, frame = cap.read()

                if not ret:  # end of video
                    # show frame
                    for marked_frame in marked_frameList:
                        cv2.imshow("image", marked_frame[0])
                        cv2.waitKey(1)
                    exit()

                # frame_buffer.append(frame)
                # stack += 1

                bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    images = list()
                    face_location = list()

                    for det_no in range(nrof_faces):
                        each_det = np.squeeze(det[det_no])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(each_det[0] - margin / 2, 0)  # left Bound
                        bb[1] = np.maximum(each_det[1] - margin / 2, 0)  # upper Bound
                        bb[2] = np.minimum(each_det[2] + margin / 2, img_size[1])  # right Bound
                        bb[3] = np.minimum(each_det[3] + margin / 2, img_size[0])  # lower Bound

                        # draw boundary box
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        face_location.append(bb)

                        # face cropping
                        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                        images.append(scaled)

                    # face vector
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    # face recognition
                    # predictions = model.predict_proba(emb_array)
                    # best_class_indices = np.argmax(predictions, axis=1)
                    # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    nameList, distanceList = realTimeFaceRecognition(emb_array, db)

                    # mark in image
                    # for i in range(len(best_class_indices)):
                    #     # person_name = '%s: %.3f' % (class_names[best_class_indices[i]], best_class_probabilities[i])
                    #     face = face_location[i]
                    #     # cv2.putText(frame, person_name, (face[0], face[3]), font, fontScale, fontColor, lineType)
                    #     frame = puttext_in_chinese(frame, person_name, (face[0], face[3]))
                    marked_frame = drawBoundaryBox([frame]*len(emb_array), face_location, nameList, distanceList)

                    marked_frameList.append(marked_frame)

                if len(marked_frameList)>=buffer_size:
                    # show frame
                    for marked_frame in marked_frameList:
                        cv2.imshow("image", marked_frame[0])
                        cv2.waitKey(1)

                    del marked_frameList

                    marked_frameList = list()

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
    #                     help='Indicates if a new classifier should be trained or a classification ' +
    #                          'model should be used for classification', default='CLASSIFY')
    # parser.add_argument('data_dir', type=str,
    #                     help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('classifier_filename',
    #                     help='Classifier model file name as a pickle (.pkl) file. ' +
    #                          'For training this is the output and for classification this is an input.')
    # parser.add_argument('--use_split_dataset',
    #                     help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
    #                          'Otherwise a separate test set can be specified using the test_data_dir option.',
    #                     action='store_true')
    # parser.add_argument('--test_data_dir', type=str,
    #                     help='Path to the test data directory containing aligned images used for testing.')
    # parser.add_argument('--batch_size', type=int,
    #                     help='Number of images to process in a batch.', default=90)
    # parser.add_argument('--image_size', type=int,
    #                     help='Image size (height, width) in pixels.', default=160)
    # parser.add_argument('--seed', type=int,
    #                     help='Random seed.', default=666)
    # parser.add_argument('--min_nrof_images_per_class', type=int,
    #                     help='Only include classes with at least this number of images in the dataset', default=20)
    # parser.add_argument('--nrof_train_images_per_class', type=int,
    #                     help='Use this number of images from each class for training and the rest for testing',
    #                     default=10)

    return parser.parse_args(argv)


if __name__ == '__main__':
    model_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/20170512-110547.pb'

    main(parse_arguments(np.atleast_1d(model_path)))
    # main(parse_arguments(sys.argv[1:]))

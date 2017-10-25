import numpy as np
# import cv2
#
# dataset_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/LFW/'
# mtcnn_directory = 'lfw_mtcnnpy_160/'
# person_name = 'Aaron_Eckhart/'
# file_name = 'Aaron_Eckhart_0001.png'
#
# picture = cv2.imread(dataset_path+mtcnn_directory+person_name+file_name)
#
# # Our operations on the frame come here
# cv2.imshow('image', picture)
# cv2.waitKey(1000)
#
# mini_picture = cv2.resize(picture, (45, 45), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('image', mini_picture)
# cv2.waitKey(1000)
#
# # When everything done, release the capture
# cv2.destroyAllWindows()


# import os
# realpath = os.path.realpath(__file__)
# print(realpath)
# split = os.path.split(realpath)
# print(split)
# splitext = os.path.splitext(split[1])
# print(splitext[0])


# ------------------------------------------------------------

import tensorflow as tf
from scipy import misc
import align.detect_face
import facenet
import os
import cv2
import time
from PIL import Image, ImageDraw, ImageFont

args_margin = 32
args_image_size = 160
args_batch_size = 1000
args_seed = 666
args_use_split_dataset = False

# get the path of this program file
src_path, _ = os.path.split(os.path.realpath(__file__))
args_model = os.path.expanduser(src_path + '/pre_train_models/20170512-110547.pb')

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


def img_read(image_path):
    try:
        img = cv2.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
    else:
        # make sure that all images are normal
        # ----------------------------------------------
        if img.ndim < 2:  # an normal image should has at least two dimension(width and high)
            print('Unable to align "%s"' % image_path)
            return
        if img.ndim == 2:  # an image which has only one channel
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]
        # ----------------------------------------------
    return img


def get_face_vec(face_imgs):
    # input a img of face closeup
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args_seed)

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args_model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            feed_dict = {images_placeholder: face_imgs, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            return emb_array


def get_image_paths_and_labels(dataset):
    image_paths_flat = list()
    labels_flat = list()
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def get_face_img(img_paths):
    # input a list of paths of images
    # return
    #     (1) close-ups of faces
    #     (2) source
    #     (3) locations
    face_closeups = list()
    face_source = list()
    face_locations = list()

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        for path in img_paths:
            img = img_read(path)
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]

                for det_no in range(nrof_faces):
                    each_det = np.squeeze(det[det_no])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(each_det[0] - args_margin / 2, 0)  # left Bound
                    bb[1] = np.maximum(each_det[1] - args_margin / 2, 0)  # upper Bound
                    bb[2] = np.minimum(each_det[2] + args_margin / 2, img_size[1])  # right Bound
                    bb[3] = np.minimum(each_det[3] + args_margin / 2, img_size[0])  # lower Bound
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    scaled = misc.imresize(cropped, (args_image_size, args_image_size), interp='bilinear')

                    face_closeups.append(scaled)
                    face_source.append(path)
                    face_locations.append(bb)

    return face_closeups, face_source, face_locations


def face_process(face_closeups, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    # input a list of faces
    # return a nd-array of pre-processed faces
    nrof_samples = len(face_closeups)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = face_closeups[i]
        try:
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            if do_prewhiten:
                img = facenet.prewhiten(img)
            img = facenet.crop(img, do_random_crop, image_size)
            img = facenet.flip(img, do_random_flip)
            images[i, :, :, :] = img
        except:
            continue
    return images


def cal_euclidean(x, y):
    # x, y must be matrices
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    diff = np.subtract(x, y)
    dist = np.sum(np.square(diff), 1)
    return dist


# -----------------------------------------------------------------------------------------------------------------------

# find the three most likely person
def faceDB(img_path, update=False):
    if update or \
            not os.path.exists('face_vectors.npy') or \
            not os.path.exists('face_source.npy') or \
            not os.path.exists('face_locations.npy'):
        people_list = os.listdir(img_path)

        print('preparing image paths')
        image_paths = list()
        for person in people_list:
            image_paths += [img_path+'%s/' % (person) + image for image in os.listdir(img_path+'%s/' % person)]

        print('loading images')
        face_closeups, face_source, face_locations = get_face_img(image_paths)

        print('processing images')
        processed_face_closeups = face_process(face_closeups, False, False, args_image_size)

        print('calculate face vectors')
        face_vectors = get_face_vec(processed_face_closeups)

        np.save('face_vectors.npy', face_vectors)
        np.save('face_source.npy', face_source)
        np.save('face_locations.npy', face_locations)

        # show
        # for count in range(len(face_vectors)):
        #     misc.imshow(face_closeups[count])
        #     misc.imshow(misc.imread(face_source[count]))
        #     bb = face_locations[count]
        #     misc.imshow(misc.imread(face_source[count])[bb[1]:bb[3], bb[0]:bb[2], :])
    else:
        face_vectors = np.load('face_vectors.npy')
        face_source = np.load('face_source.npy')
        face_locations = np.load('face_locations.npy')

    return face_vectors, face_source, face_locations

# ----------------------------------------------------------------------------------------


def faceRecognition(known_img_path, query_img_path):
    query_face_closeup, query_face_source, query_face_locations = get_face_img(np.atleast_1d(query_img_path))

    query_processed_face_ = face_process(query_face_closeup, False, False, args_image_size)
    query_face_vector = get_face_vec(query_processed_face_)

    face_vectors, face_source, _ = faceDB(known_img_path, update=False)

    faceInfo = list()

    for face_no in range(len(query_face_vector)):
        dist = cal_euclidean(query_face_vector[face_no], face_vectors)
        # indices = dist.argsort()[:3]  # find the indices of the 3 lower number
        index = dist.argsort()[:1]  # the most similar
        person_name = str(face_source[index]).split('/')[-2]
        faceInfo.append([query_face_source[face_no], query_face_locations[face_no], person_name])

    return faceInfo


def drawBoundaryBox(faceInfo):
    # font style
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

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

    imgList = list()
    imgSource = ''

    for face_no in range(len(faceInfo)):
        face_source, face_locations, person_name = faceInfo[face_no]
        if imgSource != face_source:
            imgSource = face_source
            if face_no != 0:
                imgList.append(img)
            img = cv2.imread(imgSource)

        # draw boundary box
        cv2.rectangle(img, (face_locations[0], face_locations[1]), (face_locations[2], face_locations[3]), (0, 255, 0), 2)
        # cv2.putText(img, person_name, (face_locations[0], face_locations[3]), font, fontScale, fontColor, lineType)
        img = puttext_in_chinese(img, person_name, (face_locations[0], face_locations[3]))

        if face_no == len(faceInfo)-1:
            imgList.append(img)

    return imgList


if __name__ == '__main__':
    img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/mtcnnpy_160/known_mtcnnpy_160/'

    queryList = list()
    # query_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset//unknown_people/02586.jpg'
    # queryList.append('/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/unknown_people/02586.jpg')
    queryList.append('/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/unknown_people/125215.jpg')
    beginTime = time.time()
    faceInfo = faceRecognition(img_path, queryList)
    print('%f' % (time.time()-beginTime))

    imgList = drawBoundaryBox(faceInfo)
    for img in imgList:
        # show result
        # img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('contours', img)
        cv2.waitKey(10000)

# ---------- testing accuracy---------
# import pickle
# classifier_filename_exp = os.path.expanduser('/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/facenet-master/src/pre_train_models/my_classifier.pkl')
# print('Testing classifier')
# with open(classifier_filename_exp, 'rb') as infile:
#     (model, class_names) = pickle.load(infile)
#
# print('Loaded classifier model from file "%s"' % classifier_filename_exp)
#
# predictions = model.predict_proba(face_vec)
# best_class_indices = np.argmax(predictions, axis=1)
# best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
#
# for i in range(len(best_class_indices)):
#     print('%s: %.3f' % (class_names[best_class_indices[i]], best_class_probabilities[i]))


# --------- euclidean ---------

# face_vec = np.concatenate((face_vec1, face_vec2))
#
# correlation_table = np.zeros((len(face_vec), len(face_vec)))
# euclidean_table = np.zeros((len(face_vec), len(face_vec)))
# pred_table = np.zeros((len(face_vec), len(face_vec)))
#
#
# print('euclidean_table')
# for i in range(len(face_vec)):
#     euclidean_table[i] = cal_euclidean(face_vec[i], face_vec)


# ---------- graph --------------

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.mlab as mlab
#
#
# def P(X, Y):
#     return mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# jet = plt.get_cmap('jet')
#
# langth = len(face_vec)
#
# x = np.linspace(0, langth, langth)
# y = np.linspace(0, langth, langth)
# X, Y = np.meshgrid(x, y)
# Z = euclidean_table
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=jet, linewidth=0)
# ax.set_zlim3d(0, Z.max())
#
# plt.show()

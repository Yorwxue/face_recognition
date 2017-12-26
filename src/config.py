import os
from easydict import EasyDict as edict
import json

easy_config = edict()
easy_config.TRAIN = edict()

## Adam
easy_config.TRAIN.batch_size = 16
easy_config.TRAIN.lr_init = 1e-4
easy_config.TRAIN.beta1 = 0.9

## initialize G
easy_config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
easy_config.TRAIN.n_epoch = 2000
easy_config.TRAIN.lr_decay = 0.1
easy_config.TRAIN.decay_every = int(easy_config.TRAIN.n_epoch / 2)

## train set location
easy_config.TRAIN.hr_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/DIV2K/DIV2K_train_HR/'
easy_config.TRAIN.lr_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/DIV2K/DIV2K_train_LR_bicubic/X4/'

easy_config.VALID = edict()
## test set location
# config.VALID.hr_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/DIV2K/DIV2K_valid_HR/'
# config.VALID.lr_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/DIV2K/DIV2K_valid_LR_bicubic/X4/'
easy_config.VALID.hr_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/pic_of_sir/pic_of_person_hr/'
easy_config.VALID.lr_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/pic_of_sir/pic_of_person_lr/'

easy_config.checkpoint_dir = "checkpoint"


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")

# ------------------------

# camera_size = [480, 640, 3]

src_path, _ = os.path.split(os.path.realpath(__file__))
args_model = os.path.expanduser(src_path + '/pre_train_models/20170512-110547.pb')
args_margin = 32
args_image_size = 160
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # default: [0.6, 0.7, 0.7] # three steps's threshold
factor = 0.9  # default: 0.709  # scale factor
args_seed = 666
nrof_register = 10

face_threshold = 0.6

unknown = '肩膀上的朋友'

db_name = 'sir_db'
# known_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/mtcnnpy_160/known_mtcnnpy_160/'
# known_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/mtcnnpy_160/pic_of_sir_mtcnnpy_160/'
known_img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/face_database/'
update = False

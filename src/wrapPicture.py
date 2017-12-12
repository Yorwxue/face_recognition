import os
from shutil import copyfile


def wrap_directory(path):
    picList = os.listdir(path)
    for pic in picList:
        if not os.path.exists(os.path.join(path, pic.split('.')[0])):
            os.mkdir(os.path.join(path, pic.split('.')[0]))
        # os.rename(os.path.join(path, pic), os.path.join(path, pic.split('.')[0], pic))
        copyfile(os.path.join(path, pic), os.path.join(path, pic.split('.')[0], pic))
        # os.rename(os.path.join(path, pic.split('.')[0], pic), os.path.join(path, pic.split('.')[0], path.split('/')[-2]+pic))


def un_wrap(path, target_dir):

    itemList = os.listdir(path)
    for item in itemList:
        if item != 'all':
            if os.path.isfile(os.path.join(path, item)):
                # os.rename(os.path.join(path, item), os.path.join(target_dir, item))
                copyfile(os.path.join(path, item), os.path.join(target_dir, item))
                os.rename(os.path.join(target_dir, item), os.path.join(target_dir, item))
            else:
                un_wrap(os.path.join(path, item), target_dir)


if __name__ == '__main__':
    path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/pic_of_sir'
    # path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/LFW/raw/lfw'
    # path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/crawlar_img'
    # path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/face_recognition_dataset/character'
    # wrap_directory(path)

    if not os.path.exists(os.path.join(path, 'all')):
        os.mkdir(os.path.join(path, 'all'))
    un_wrap(path, os.path.join(path, 'all'))

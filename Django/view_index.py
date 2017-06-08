import os


def getProjectRootPath():
    pwd = os.getcwd()
    return os.path.abspath(os.path.dirname(pwd)) + '/Django'


import sys

sys.path.append(getProjectRootPath())

import cStringIO
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.shortcuts import render_to_response
from django import forms
import time
from PIL import Image
# import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
from networks.factory import get_network


class UserForm(forms.Form):
    # username = forms.CharField()
    headImg = forms.FileField()


def hello(request):
    # img = request.POST[]
    if request.method == "POST":
        uf = UserForm(request.POST, request.FILES)
        if uf.is_valid():
            img = uf.files['headImg']
            filename = saveImage(img)
            return render_to_response('show_image.html', {'images': [filename]})
    else:
        uf = UserForm()
    return render_to_response('index.html', {'uf': uf})
    # return render_to_response('index.html')


from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


def saveImage(image):
    pwd = os.getcwd()
    filetype = image.content_type.split('/')[1]
    filePath = os.path.abspath(os.path.dirname(pwd)) + '/Django/image/' + str(time.time()) + '.' + filetype

    if isinstance(image, TemporaryUploadedFile):
        temp_file = open(image.temporary_file_path(), 'rb+')
        content = cStringIO.StringIO(temp_file.read())
        image = Image.open(content)
        temp_file.close()

    # image.seek(0)
    # img = Image.open(image)
    # img.save(filePath)
    default_storage.save(filePath, ContentFile(image.read()))
    path = detectObject(filePath, filetype)
    # new_img = Image.open(path)
    fileName = path.split('/')[-1]
    return fileName
    # return HttpResponse("<img src="{% static img %}" style="width: 80%;alignment: center">")


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = path
    # im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)


# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='Faster R-CNN demo')
#     parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
#                         default=0, type=int)
#     parser.add_argument('--cpu', dest='cpu_mode',
#                         help='Use CPU mode (overrides --gpu)',
#                         action='store_true')
#     parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
#                         default='VGGnet_test')
#     parser.add_argument('--model', dest='model', help='Model path',
#                         default=' ')
#
#     args = parser.parse_args()
#
#     return args


def detectObject(path, filetype):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # args = parse_args()

    # if args.model == ' ':
    #     raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network('VGGnet_test')
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    model_path = getProjectRootPath() + '/Django/model/VGGnet_fast_rcnn_iter_70000.ckpt'
    saver.restore(sess, model_path)

    # sess.run(tf.initialize_all_variables())

    print
    '\n\nLoaded network {:s}'.format(model_path)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    #
    # for im_name in im_names:
    #     print
    #     '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #     print
    #     'Demo for data/demo/{}'.format(im_name)
    #     demo(sess, net, im_name)
    demo(sess, net, path)
    # plt.show()
    plt.savefig(path + '.detected.' + filetype)
    return path + '.detected.' + filetype

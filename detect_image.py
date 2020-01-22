import argparse
import os
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import torch

from skimage.color import gray2rgb

from sklearn import preprocessing

"""
Extension to Detect.py that can be used for detecting objects. 

"""
def clipping_stretching(timestep):
    """
    Applies a clipping and stretching, discarding everything outside the 5th and 95th percentile.
    timestep - The timestep to use.
    """
    min_max_scaler = preprocessing.MinMaxScaler()

    timestep[timestep > np.percentile(timestep, 95)] = np.percentile(timestep, 95)
    timestep[timestep < np.percentile(timestep, 5)] = np.percentile(timestep, 5)
    return min_max_scaler.fit_transform(timestep)

class LoadSingleImage:
    def __init__(self, images, img_size=416, half=False):
        self.images = images
        self.img_size = img_size
        self.mode = 'images'
        self.half = half
        self.nF = len(images)
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        img0 = gray2rgb(clipping_stretching(self.images[self.count]))

        self.count += 1

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32) # uint8 to fp16/fp32

        return img, img0, self.cap

class YoloV3:
    def __init__(self, weights_file, half=False, img_size=(416, 416), agnostic_nms=False, iou_thres=0.5, conf_thres=0.3,cfg='cfg/yolov3-spp.cfg', device='1', names_file='data/coco.names'):
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.agnostic_nms = agnostic_nms
        self.img_size = img_size
        self.weights_file = weights_file
        self.half = half
        self.cfg = cfg
        self.names_file = names_file

        # Select device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = Darknet(self.cfg, self.img_size)

        # Load weights
        # attempt_download(self.weights_file)
        if self.weights_file.endswith('.pt'):  # pytorch format
            print(self.weights_file)
            print(os.getcwd())
            self.model.load_state_dict(torch.load(self.weights_file, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, self.weights_file)

        # Eval mode
        self.model.to(self.device).eval()

        # Half precision
        half = half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        # Get names and colors
        self.names = load_classes(self.names_file)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def predict(self, images):
        # Create dataloader
        dataset = LoadSingleImage(images, img_size=self.img_size[0])

        t0 = time.time()
        predictions = []
        for img, im0s, vid_cap in dataset:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img)[0]

            if self.half:
                pred = pred.float()
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       agnostic=self.agnostic_nms)
            # Process detections
            for i, det in enumerate(pred): # detections per image
                prediction = []
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    for x, y, w, h, conf, cls in det.detach().numpy():
                        prediction.append({'class': cls.item(), 'x': x.item()/self.img_size[0], 'y': y.item()/self.img_size[0], 'width': (w.item()-x.item())/self.img_size[0], 'height': (h.item()-y.item())/self.img_size[0], 'conf': conf.item()})
                predictions.append(prediction)
        return predictions, t0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()

import time

import cv2
import torch
# import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#### This a modification based on the 'LoadImages' from 'utils.datasets'
class LoadSingleImage:

    def __init__(self, img_file, img_size=640, stride=32):
        self.img_file = img_file
        self.img_size = img_size
        self.stride = stride
        self.nf = 1

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # Decode image data as a numpy array
        image = np.frombuffer(self.img_file, np.uint8)
        # Decode image as a color image
        img0 = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x16x16
        img = np.ascontiguousarray(img)

        return img, img0

    def __len__(self):
        return self.nf  # number of files

class Initialization:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(
        self, 
        weights = 'rtrain-2.pt', 
        img_size = 640, 
        conf_thres = 0.25,
        iou_thres = 0.45,
        classes = None,
        agnostic_nms = False,
        augment = False,
        no_trace = False
        ):
        self.configure(img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes, agnostic_nms=agnostic_nms, augment=augment, no_trace=no_trace, weights=weights)

        

    def configure(
        self, 
        weights = 'rtrain-2.pt', 
        img_size = 640, 
        conf_thres = 0.25,
        iou_thres = 0.45,
        classes = None,
        agnostic_nms = False,
        augment = False,
        no_trace = False
        ):
        self.img_size=img_size
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.no_trace=no_trace
        self.weights=[weights]
        self.__setConfiguration()
    
    def __setConfiguration(self):
        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        
        self.stride = int(self.model.stride.max())  # model stride

        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size

        if not self.no_trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()


        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.img_size
        self.old_img_h = self.img_size
        self.old_img_b = 1
    
    #TODO: Improve the return.
    def detection(self, image):
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Set Dataloader
        dataset = LoadSingleImage(image,img_size=self.img_size, stride=self.stride)

        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections 
            for i, det in enumerate(pred):  # detections per image

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    result = ""

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        result = result + f"{n} {self.names[int(c)]}{'s' * (n > 1)},"
                        print(result)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # label format
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=4)
                    _, img_encoded = cv2.imencode('.jpg', im0s) # Encode image in JPG format
                    return img_encoded, result
            return 0,""
    
    
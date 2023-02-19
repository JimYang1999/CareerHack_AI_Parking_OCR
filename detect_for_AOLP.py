from pathlib import Path
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from model.inference import Scatter_Text_Recognizer
from PIL import Image
import os 
import csv

def detect(source='../dataset/AOLP_answer/Subset_RP/Image/' , weights='./best.pt' , img_size=1280 , iou_thres=0.5 , conf_thres=0.9 , detect_index = 1 , show_img = False):
    license_list = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize
    set_logging()
    device = select_device(device)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    index = 1439
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
        
        # Inference
        with torch.no_grad(): # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            all_location = []
            all_label = []
            p, im0, frame = path, im0s, getattr(dataset, 'frame', 0)
            img1 = im0.copy()
            p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    location = []
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    for xy in xyxy:
                        location.append(int(xy))
                    all_location.append(location)
                    all_label.append(label.split(' ')[0])

                if index%detect_index==0 and len(all_label)>0:
                    num=0
                    for location in all_location:
                        license = img1[int(location[1]):int(location[3]),int(location[0]):int(location[2])]
                        img_name = str(p).split('\\')[-1].split('.')[0]
                        print(p)
                        cv2.imwrite(f'../dataset/AOLP_answer/image/{str(index)}_{str(num)}.jpg' , license)
                        num+=1
                    license_list = read_txt(img_name , license_list , index)
        index+=1
    write_csv(license_list)

def write_csv(license_list):
    with open('../dataset/AOLP_answer/answer2.csv', 'w', newline='' , encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in license_list:
            index = i.split(':')[0]
            name = i.split(':')[1]
            print(index , name)
            writer.writerow([index,name])

def read_txt(img_name , license_list , index):
    path = '../dataset/AOLP_answer/Subset_RP/groundtruth_recognition/'
    f = open(path + img_name + ".txt", 'r')
    text = f.read()
    f.close()
    license_list.append(str(index) + ".jpg" + ":" + text.replace('\n', ''))
    return license_list
        

if __name__ == '__main__':
        detect()
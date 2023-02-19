from pathlib import Path
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import plot_one_box , check_text
from utils.torch_utils import select_device
from model.inference import Scatter_Text_Recognizer
from PIL import Image
from google.cloud import storage
from google.cloud import vision
from google.cloud.vision_v1 import types
import os
import io
import csv

def write_csv(license_list):
    with open('./reference_data_private_results.csv', 'w', newline='' , encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img_filename','detected_license_plate'])
        for i in license_list:
            index = i.split(':')[0]
            name = i.split(':')[1]
            print(index , name)
            writer.writerow([index,name])


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./bsid-user-group4-sa-key.json"

def detect(source='../dataset/AOLP/all_image/' , weights='./best.pt' , img_size=1280 , iou_thres=0.9 , conf_thres=0.8 , show_img = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize
    ocr_model = Scatter_Text_Recognizer()
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
    license_list = []
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
            print(p)
            img_name = str(p).split('\\')[-1].split('.')[0]
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
                    cv2.imwrite('./test.jpg' , im0)
                    print(location)
                    all_location.append(location)
                    all_label.append(label.split(' ')[0])
                if len(all_label)>0:   
                    max_shape = 0
                    for location in all_location:
                        shape = (int(location[3])-int(location[1]))*(int(location[2])-int(location[0]))
                        if  shape > max_shape:
                            license = img1[int(location[1]):int(location[3]),int(location[0]):int(location[2])]
                            max_shape = shape

                    text = recognize_ocr(license, ocr_model)
                    if text==None or text.count('-')==2:
                        text = detect_text(license)
                        print(f'None -> {text}')
                    text = check_text(text.replace('-',''))
                    license_list.append(f'{img_name}.jpg:{text}')
                
            else:
                text = detect_text(img1)
                text = check_text(text.replace('-',''))
                license_list.append(f'{img_name}.jpg:{text}')
    write_csv(license_list)

def recognize_ocr(img, ocr_model):
    img = Image.fromarray(img)
    text = ocr_model.predict(img)
    return text

def detect_text(img):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    success, encoded_image = cv2.imencode('.jpg', img)
    img = encoded_image.tobytes()
    img = types.Image(content=img)
    response = client.text_detection(image=img)
    texts = response.text_annotations
    text = texts[0].description
    if '\n' in text:
        print(text)
        text = text.split('\n')[-1]
    if len(texts)>0:
        return text 
    else: 
        return 'None'

if __name__ == '__main__':
        detect()
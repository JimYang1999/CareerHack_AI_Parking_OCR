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
from google.cloud import storage
from google.cloud import vision
from google.cloud.vision_v1 import types
import os
import requests
from datetime import datetime
import json
import shutil

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./bsid-user-group4-sa-key.json"

def detect(source='../dataset/Park_image/' , weights='./best.pt' , img_size=1280 , iou_thres=0.9 , conf_thres=0.7 , detect_index = 1 , show_img = False):
    mode = 'Exit' #'Entry' , 'Exit' , 'Park'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ocr_model = Scatter_Text_Recognizer()
    set_logging()
    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride) 
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
        
        with torch.no_grad(): 
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):  # detections per image
            all_location = []
            all_label = []
            p, im0, frame = path, im0s, getattr(dataset, 'frame', 0)
            img1 = im0.copy()
            p = Path(p)  # to Path
            print(p)
            img_name = str(p).split('\\')[-1].split('.')[0]
            
            ER_PLID = img_name[0]
            ER_ENTER_TIME = img_name.split('_')[1]
            ER_EXIT_TIME = img_name.split('_')[-1]
            PS_CODE = img_name.split('_')[0].split('-')[-1]
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

                if len(all_label)>0:
                    max_shape = 0
                    for location in all_location:
                        shape = (int(location[3])-int(location[1]))*(int(location[2])-int(location[0]))
                        if  shape > max_shape:
                            license = img1[int(location[1]):int(location[3]),int(location[0]):int(location[2])]
                            max_shape = shape

                    text = recognize_ocr(license, ocr_model)
                    if text == None or text.count('-')>1:
                        text = detect_text(license)
                    text = text.replace('-','')
                    
                    destination =f"{mode}_image/" + img_name + '_' + text + '.jpg'
                    try:
                        upload_blob(source_file_name=p  , destination_blob_name=destination)
                    except:
                        pass
                    
                    if mode =='Entry':
                        send_data_Entry(ER_PLID , text , ER_ENTER_TIME , destination , PS_CODE)
                        try:
                            upload_blob(source_file_name=p  , destination_blob_name=destination.replace('Entry' , 'Park')) #upload park_img
                        except:
                            pass
                        
                        try:
                            shutil.move(f'../dataset/Exit_image/{img_name}.jpg' , f'../dataset/Park_image/{img_name}.jpg')
                        except:
                            pass
                        
                    if mode=='Exit':
                        send_data_Exit(text , ER_EXIT_TIME)
                        try:
                            shutil.move(f'../dataset/Park_image/{img_name}.jpg' , f'../dataset/Exit_image/{img_name}.jpg')
                        except:
                            pass
                        
            else:
                text = detect_text(img1)
                text = text.replace('-','')
                destination =f"{mode}_image/" + img_name + '_' + text + '.jpg'
                try:
                    upload_blob(source_file_name=p  , destination_blob_name=destination)
                except:
                    pass
                if mode =='Entry':
                    send_data_Entry(ER_PLID , text , ER_ENTER_TIME , destination , PS_CODE)
                    try:
                        upload_blob(source_file_name=p  , destination_blob_name=destination.replace('Entry' , 'Park')) #upload park_img
                    except:
                        pass
                    try:
                        shutil.move(f'../dataset/Exit_image/{img_name}.jpg' , f'../dataset/Park_image/{img_name}.jpg')
                    except:
                        pass
                if mode=='Exit':
                    send_data_Exit(text , ER_EXIT_TIME)
                    try:
                        shutil.move(f'../dataset/Park_image/{img_name}.jpg' , f'../dataset/Exit_image/{img_name}.jpg')
                    except:
                        pass
                    
            if show_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        
def recognize_ocr(img, ocr_model):
    img = Image.fromarray(img)
    text = ocr_model.predict(img)
    # text = reader.readtext(img,detail = 0 , text_threshold = 0.7)[0].upper()
    print(f'license plate = {text}\n')
    return text

def upload_blob(bucket_name = 'tsmchack2023-bsid-grp4-public-write-bucket', source_file_name='', destination_blob_name=''):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

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
        return None
    
def send_data_Entry(PLID , LICENSE , ENTER_TIME ,destination ,PS_CODE):
    PLID_dict = {'A':'1','B':'2','C':'3','D':'4'}
    enter_time = datetime(2023,2,11,int(ENTER_TIME[:2]),int(ENTER_TIME[-2:])).__str__() #+ datetime(ENTER_TIME)
    params = {
    "ER_PLID" : f'PL_0000{PLID_dict[PLID]}', #A1B2C3D4
    "ER_LICENSE" : LICENSE,
    "ER_ENTER_TIME": enter_time, 
    "ER_IMAGE" : '/' + destination,
    }
    api = '/webapi/insert_entry_record'
    url = 'http://34.81.83.19:8000' + api
    html = requests.post(f'{url}', json.dumps(params)) # select
    PS_ID = get_data_PSID(PLID , PS_CODE)

    insert_park(params["ER_PLID"] , PS_ID , LICENSE , destination)

def send_data_Exit(LICENSE ,EXIT_TIME):
    exit_time = datetime(2023,2,11,int(EXIT_TIME[:2]),int(EXIT_TIME[-2:])).__str__() #+ datetime(ENTER_TIME)
    api = f'/webapi/update_ENTRY_RECORD_exit_time_by_ER_LICENSE/{LICENSE}/{exit_time}'
    url = 'http://34.81.83.19:8000' + api
    html = requests.put(f'{url}')
    print(html)

def get_data_PSID(PLID , PS_CODE):
    api = f'/appapi/get_PARKING_SPACES_info_by_PS_CODE/{PLID}{PS_CODE}'
    url = 'http://34.81.83.19:8000' + api
    html = requests.get(f'{url}')
    data = json.loads(html.text)["response"][0]["PS_ID"]
    return data

def insert_park(PA_PLID , PA_PSID , PA_LICENSE , PA_IMAGE):
    params = {
    "PA_PLID" : PA_PLID ,
    "PA_PSID" : PA_PSID ,
    "PA_LICENSE": PA_LICENSE, 
    "PA_IMAGE" : '/' + PA_IMAGE.replace('Entry' , 'Park')
    }
    api = '/webapi/insert_park'
    url = 'http://34.81.83.19:8000' + api
    html = requests.post(f'{url}', json.dumps(params)) # select

if __name__ == '__main__':
        detect()
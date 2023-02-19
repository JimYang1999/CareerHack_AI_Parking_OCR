import numpy as np
from model.model import Model
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i
            
    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    
class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.convert('L')
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class Model_Set:
    def __init__(self, num_class):
        self.imgH = 50
        self.imgW = 100
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.num_class = num_class
        self.hidden_size = 512
        self.batch_max_length = 10

class Scatter_Text_Recognizer:
    def __init__(self):
        self.converter = AttnLabelConverter('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        self.opt = Model_Set(len(self.converter.character))
        self.length_for_pred = torch.IntTensor([self.opt.batch_max_length])
        self.text_for_pred = torch.LongTensor(1, self.opt.batch_max_length).fill_(0)
        self.transform = ResizeNormalize((50, 100))
        
        self.model = Model(self.opt)
        self.model.load_state_dict(torch.load('./model/checkpoint/weight.pth', map_location='cpu'))
    def predict(self, img):
        answer = ""
        max_score = 0
        max_score_list = [0,0,0,0,0,0,0] #用不同weight要改7或6個0
        max_index_list = [0,0,0,0,0,0,0] #
        index = 0
        less_7 = 0
        empty = 0
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            predss = self.model(img, self.text_for_pred, is_train=False)[0]
            confidence_score_list = []
            pred_str_list = []
            for preds in predss:
                _, preds_index = preds.max(2)
                pred_str = self.converter.decode(preds_index, self.length_for_pred)[0]
                
                preds_prob = F.softmax(preds, dim=2)
                pred_max_prob, _ = preds_prob.max(dim=2)
                pred_EOS = pred_str.find('[s]')
                pred_str = pred_str[:pred_EOS]
                pred_max_prob = pred_max_prob[0][:pred_EOS]
                print(pred_max_prob.cumprod(dim=0))
                if len(pred_max_prob.cumprod(dim=0))==0:empty+=1 #如果陣列為空，就加1，如果五次都判斷為空，就回傳None
                if len(pred_max_prob.cumprod(dim=0)) < 7:less_7 += 1  #如果判斷到的車牌號碼少於7(6)碼，就+1
                for i in range(len(pred_max_prob.cumprod(dim=0))):
                    if max_score_list[i] < pred_max_prob.cumprod(dim=0)[i]:
                        max_score_list[i] = pred_max_prob.cumprod(dim=0)[i]
                        max_index_list[i] = index
                confidence_score = pred_max_prob.cumprod(dim=0)
                if confidence_score.size(0) != 0:
                    confidence_score_list.append(confidence_score[-1].item())
                else:
                    confidence_score_list.append(0)
                pred_str_list.append(pred_str)
                
                index+=1
            if empty ==5:
                return None
            confidence_score_list = np.array(confidence_score_list)
            pred_str_list = np.array(pred_str_list)
            best_pred_index = np.argmax(confidence_score_list, axis=0)
            best_pred_index = np.expand_dims(best_pred_index, axis=0)
            str_index = 0
            if less_7 < 5:
                for k in max_index_list:
                    answer += pred_str_list[k][str_index]
                    str_index+=1
            else:
                answer = pred_str_list[best_pred_index][0]
        return answer
        
if __name__ == '__main__':
        
    img = Image.open('./397.jpg')
    ocr = Scatter_Text_Recognizer()
    code = ocr.predict(img)
    print(code)
    pass

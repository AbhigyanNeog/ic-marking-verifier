import cv2
import torch
import numpy as np
from craft import CRAFT  # CRAFT model script (available at https://github.com/clovaai/CRAFT-pytorch)
from torch.autograd import Variable
import pytesseract
import torch.nn.functional as F

# Utilities to decode CRAFT output
from craft_utils import getDetBoxes, adjustResultCoordinates  # from official repo
from imgproc import resize_aspect_ratio, normalizeMeanVariance  # from official repo

from collections import OrderedDict

class CraftOCR:
    def __init__(self, craft_model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = CRAFT()
        # Load checkpoint with 'module.' prefix fix
        state_dict = torch.load(craft_model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]  # remove 'module.' prefix
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

        self.model.eval()
        self.model.to(self.device)

    def detect_text_boxes(self, image):
        # Preprocess image
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR)
        ratio_h = ratio_w = 1 / target_ratio
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            y, _ = self.model(x)
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        boxes, polys = getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4)
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        return boxes

    def ocr_boxes(self, image, boxes, whitelist=None):
        results = []
        for box in boxes:
            pts = box.astype(int)
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            roi = image[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = "--psm 7 --oem 3"
            if whitelist:
                config += f" -c tessedit_char_whitelist={whitelist}"
            text = pytesseract.image_to_string(thresh, config=config).strip()
            if text:
                results.append(text)
        return " ".join(results)

def run_ic_ocr(image, craft_model_path, whitelist=None, device='cpu'):
    detector = CraftOCR(craft_model_path, device=device)
    boxes = detector.detect_text_boxes(image)
    if boxes is None or len(boxes) == 0:
        return ""
    return detector.ocr_boxes(image, boxes, whitelist=whitelist)


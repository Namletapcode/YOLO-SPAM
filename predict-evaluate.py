import cv2
import torch
from ultralytics_custom import YOLO

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False  
    return _original_load(*args, **kwargs)
torch.load = _patched_load

if __name__ == '__main__':
    
    model = YOLO('data/weights/epoch52.pt')

    print("================== BẮT ĐẦU DỰ ĐOÁN ==================")
    results = model.predict(
        source='data/MP-IDB-YOLO/test.txt',  
        conf=0.43,         
        iou=0.45,         
        save=True,         
        show=False,     
        project='results',    
        name='predict',
        device='cpu' 
    )

    print("================== BẮT ĐẦU ĐÁNH GIÁ ==================")
    metrics = model.val(
        data='D:/Documents/NCKH/YOLO-SPAM/data/MP-IDB-YOLO/data.yaml',  
        batch=4,           
        imgsz=640,         
        conf=0.001,        
        iou=0.6,
        plots=True,        
        save_json=True,
        single_cls=False,   
        project='results', 
        name='evaluate',
        workers=0,         
        device='cpu'
    )
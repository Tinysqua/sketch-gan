'''
识别的流程：
1. 使用attempt load 函数去根据weights自动加载模型
2. 使用一个迭代器类去不断读取图片（比使用while 1更加高级）
3. 得到之后对图片使用letterbox进行resize 处理
4.  通过model得到pred结果
5. 通过non_max_suppression函数处理上一步得到的pred结果，并且根据调参可以调整阈值
6. 使用scalebox还原原来尺寸的xywh，然后根据切片就可以获得相应的信息。
'''
import argparse
import torch
from models.experimental import attempt_load
from Load_Stream import Load_Stream
from utils.general import check_img_size, non_max_suppression, scale_boxes
from models.Warm_up import warmup
def run(args):
    device = torch.device(f'cuda:{args.device}')
    model = attempt_load(args.weights, device=device, inplace=True, fuse=True)
    fp16 = args.fp16
    stride = stride = max(int(model.stride.max()), 32)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.half() if fp16 else model.float()
    names = model.names
    img_sz = check_img_size(args.imgsz, s=stride)

    dataset = Load_Stream(s=args.stream)
    warmup(model, imgsz=(1, 3, *img_sz) ,device=device)

    for im, im_read in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, args.conf, args.iou_thres, None, False, max_det=1000)

        for i, det in enumerate(pred):
            print(det.shape)
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im_read.shape).round()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='E:\server_using\yolo-v5\\runs\\train\exp7\weights\\best.pt')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--stream', type=int, default=0)
    parser.add_argument('--imgsz', type=list, default=[640, 640])
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    
    args = parser.parse_args()
    run(args)
    

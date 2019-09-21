import time, os
import json
import mmcv
from mmdet.apis import init_detector, inference_detector


def main():
    th = 0.05
    config_file = 'config.py'  # config file
    checkpoint_file = '../work_dirs/epoch_12.pth'  # checkpoint file

    test_path = '../data/guangdong1_round1_testB_20190919/'  # data path

    json_name = "result_" + "" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".json"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    for i, img_name in enumerate(img_list, 1):
        print(i)
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                image_name = img_name
                for bbox in bboxes:
                    if bbox[4] > th:
                        x1, y1, x2, y2, score = bbox.tolist()
                        x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                        result.append(

                            {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

    with open(json_name, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()
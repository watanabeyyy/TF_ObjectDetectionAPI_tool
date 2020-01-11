"""detection向けのモジュール
NMS, 検出結果の表示等
"""
import cv2
import numpy as np
from operator import itemgetter


def non_max_suppression(objects, overlap_thresh, img):
    """NMSを行う
    Args:
        objects : 検出した矩形のリスト
        overlap_thresh : IOUの閾値、これ以上の物は同一物体とみなす
        img : 検出に使った画像,boundingboxのrescaleに使う
    Returns:
        objects: NMSで残った矩形のリスト
    Note:
        info = [score, label, box ([xmin,ymin,xmax,ymax])]
        objects = [info, info, ...]
        [[0.7692556, 'white', [0.26800233, 0.2805628, 0.33868653, 0.34357908]], 
        [0.75813735, 'black', [0.093283266, 0.19705181, 0.15611862, 0.25550985]],
        [0.6268233, 'black', [0.21051638, 0.27619553, 0.2744451, 0.33593547]]]
    """
    if len(objects) < 2:
        return objects

    # creat box list
    boxes = []
    scores = []
    for ball in objects:
        scores.append(ball[0])
        boxes.append(ball[2])
    scores = np.array(scores, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    h, w, c = img.shape
    boxes = boxes * np.array([w, h,  w, h])

    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(scores)  # スコアを降順にソートしたインデックス一覧
    selected = []  # NMS により選択されたインデックス一覧

    # indices がなくなるまでループする。
    while len(indices) > 0:
        # indices は降順にソートされているので、一番最後の要素の値 (インデックス) が
        # 残っている中で最もスコアが高い。
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        overlap = (i_w * i_h) / area[remaining_indices]
        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(indices, np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0])))

    # 選択された短形の一覧を返す。
    if len(selected) > 1:
        return itemgetter(*selected)(objects)
    else:
        return [itemgetter(*selected)(objects)]


def overlay_result(objects, img):
    """boundingboxを画像に描画する
    Args:
        objects : 検出した矩形のリスト
        img : 検出に使った画像,boundingboxのrescaleに使う
    Returns:
        img: 検出枠を書き込んだ画像
    """
    img = np.copy(img)
    colors = [
        (0, 0, 255),
        (0, 255, 0),
    ]
    for ball in objects:
        detection_score = ball[0]
        label = ball[1]
        if label == "white":
            class_id = 0
        elif label == "black":
            class_id = 1
        # Define bounding box
        h, w, c = img.shape
        box = ball[2] * np.array([h, w,  h, w])
        box = box.astype(np.int)
        # Draw bounding box
        cv2.rectangle(img, (box[0], box[1]),
                      (box[2], box[3]), colors[class_id], 3)
        # Put label near bounding box
        information = label[0] + "," + str(int(detection_score * 100.0))
        cv2.putText(img, information, (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 1, cv2.LINE_AA)
    return img


if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    from detection import DetectionModel_PB, DetectionModel_TFLite, inference_tta
    detection_model = DetectionModel_PB()
    input_dir = "C:/home/workspace/BallRecognitionApp/backend/img"
    paths = glob.glob(input_dir + "/*.png")
    print(paths)
    for path in tqdm(paths):
        img = cv2.imread(path)
        objects = inference_tta(detection_model, img)
        out_img = overlay_result(objects, img)
        cv2.imshow("", out_img)
        cv2.waitKey(0)
        objects = non_max_suppression(objects, 0.5, img)
        out_img = overlay_result(objects, img)
        cv2.imshow("", out_img)
        cv2.waitKey(0)

    detection_model = DetectionModel_TFLite()
    input_dir = "C:/home/workspace/BallRecognitionApp/backend/img"
    paths = glob.glob(input_dir + "/*.png")
    print(paths)
    for path in tqdm(paths):
        img = cv2.imread(path)
        objects = inference_tta(detection_model, img)
        out_img = overlay_result(objects, img)
        cv2.imshow("", out_img)
        cv2.waitKey(0)
        objects = non_max_suppression(objects, 0.5, img)
        out_img = overlay_result(objects, img)
        cv2.imshow("", out_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

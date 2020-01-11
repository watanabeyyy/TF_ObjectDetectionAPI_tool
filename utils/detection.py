"""検出モデルで推論を行う
ProtocolBuffer,TFLiteのモデルに対応
"""
import cv2
import tensorflow as tf
import numpy as np


class DetectionModel_PB():
    def __init__(self, model_path="data/frozen_inference_graph.pb"):
        """model loader
        Args:
            model_path (str, optional): path to pb model. Defaults to "data/frozen_inference_graph.pb".
        Note:
            Tensorflow Object Detection API でのprotocolbufferモデル作成を想定しています
        """
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with self.detection_graph.as_default():
            self.tf_sess = tf.compat.v1.Session(config=tf_config)
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            self.image_tensor = tf.compat.v1.get_default_graph(
            ).get_tensor_by_name('image_tensor:0')

    def inference(self, img, score_thresh=0.5):
        """detection実行
        Args:
            img : 画像入力
            score_thresh : 検出スコアの閾値
        Returns:
            objects: 検出した矩形情報のリスト
        Note:
            info = [score, label, box ([xmin,ymin,xmax,ymax])]
            objects = [info, info, ...]
            [[0.7692556, 'white', [0.26800233, 0.2805628, 0.33868653, 0.34357908]], 
            [0.75813735, 'black', [0.093283266, 0.19705181, 0.15611862, 0.25550985]],
            [0.6268233, 'black', [0.21051638, 0.27619553, 0.2744451, 0.33593547]]]
        """
        img = np.copy(img)
        img = cv2.resize(img, (320, 320))
        # convert bgr to rgb
        img = img[:, :, ::-1]
        img = np.expand_dims(img, axis=0)
        # img = img.astype(np.float32)
        output_dict = self.tf_sess.run(self.tensor_dict,
                                       feed_dict={self.image_tensor: img})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        num_detections = int(output_dict['num_detections'][0])
        detection_classes = output_dict['detection_classes'][0].astype(
            np.int64)
        detection_boxes = output_dict['detection_boxes'][0]
        detection_scores = output_dict['detection_scores'][0]

        # 検出したobjectsを格納するリスト
        objects = []
        for i in range(num_detections):
            # get outputs
            class_id = int(detection_classes[i])
            if class_id == 1:
                label = 'white'
            elif class_id == 2:
                label = "black"
            box = []
            score = detection_scores[i]
            if score > score_thresh:
                #x > 配列列方向
                # y v 配列行方向
                box.append(detection_boxes[i][1])  # xmin upper left horizontal
                box.append(detection_boxes[i][0])  # ymin upper left vertical
                # xmax under right horizontal
                box.append(detection_boxes[i][3])
                box.append(detection_boxes[i][2])  # ymax under right vertical
                objects.append([score, label, box])
        return objects


class DetectionModel_TFLite():
    def __init__(self, model_path="data/tflite_graph.tflite"):
        """model loader
        Args:
            model_path (str, optional): path to tflite model. Defaults to "data/tflite_graph.tflite".
        Note:
            Tensorflow Object Detection API でのtfliteモデル作成を想定しています
        """
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.interpreter.allocate_tensors()

    def inference(self, img, score_thresh=0.5):
        """detection実行
        Args:
            img : 画像入力
            score_thresh : 検出スコアの閾値
        Returns:
            objects: 検出した矩形情報のリスト
        Note:
            info = [score, label, box ([xmin,ymin,xmax,ymax])]
            objects = [info, info, ...]
            [[0.7692556, 'white', [0.26800233, 0.2805628, 0.33868653, 0.34357908]], 
            [0.75813735, 'black', [0.093283266, 0.19705181, 0.15611862, 0.25550985]],
            [0.6268233, 'black', [0.21051638, 0.27619553, 0.2744451, 0.33593547]]]
        """
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # input_shape = input_details[0]['shape']

        img = np.copy(img)
        img = cv2.resize(img, (320, 320))
        # convert bgr to rgb
        img = img[:, :, ::-1]/127.
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        self.interpreter.set_tensor(input_details[0]['index'], img)

        # 実行
        self.interpreter.invoke()

        # get outputs
        boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(output_details[2]['index'])[0]
        num_detections = self.interpreter.get_tensor(
            output_details[3]['index'])[0]

        # 検出したobjectsを格納するリスト
        objects = []
        for i in range(int(num_detections)):
            class_id = int(classes[i])
            if class_id == 0:
                label = 'white'
            else:
                label = "black"
            box = []
            score = scores[i]
            if score > score_thresh:
                #x > 配列列方向
                # y v 配列行方向
                box.append(boxes[i][1])  # xmin upper left horizontal
                box.append(boxes[i][0])  # ymin upper left vertical
                box.append(boxes[i][3])  # xmax under right horizontal
                box.append(boxes[i][2])  # ymax under right vertical
                objects.append([score, label, box])
        return objects


def inference_tta(DetectionModel, img, score_thresh = 0.5):
    """ttaの推論を行う関数
    Args:
        img : 画像入力
        score_thresh : 検出スコアの閾値
    Returns:
        objects: 検出した矩形情報のリスト
    """
    img = np.copy(img)

    # normal
    objects = DetectionModel.inference(img, score_thresh)
    tta_objects = objects

    # reverse x
    img = img[:, ::-1, :]
    objects = DetectionModel.inference(img, score_thresh)
    img = img[:, ::-1, :]
    for ball in objects:
        ball[2][0] = 1 - ball[2][0]
        ball[2][2] = 1 - ball[2][2]
        ball[2][0], ball[2][2] = ball[2][2], ball[2][0]
    tta_objects.extend(objects)

    # reverse y
    img = img[::-1, :, :]
    objects = DetectionModel.inference(img, score_thresh)
    img = img[::-1, :, :]
    for ball in objects:
        ball[2][1] = 1 - ball[2][1]
        ball[2][3] = 1 - ball[2][3]
        ball[2][1], ball[2][3] = ball[2][3], ball[2][1]
    tta_objects.extend(objects)

    # reverse xy
    img = img[::-1, ::-1, :]
    objects = DetectionModel.inference(img, score_thresh)
    img = img[::-1, ::-1, :]
    for ball in objects:
        ball[2][0] = 1 - ball[2][0]
        ball[2][1] = 1 - ball[2][1]
        ball[2][2] = 1 - ball[2][2]
        ball[2][3] = 1 - ball[2][3]
        ball[2][0], ball[2][2] = ball[2][2], ball[2][0]
        ball[2][1], ball[2][3] = ball[2][3], ball[2][1]
    tta_objects.extend(objects)

    return tta_objects


if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    detection_model = DetectionModel_PB()
    input_dir = "C:/home/workspace/BallRecognitionApp/backend/img"
    paths = glob.glob(input_dir + "/*.png")
    print(paths)
    for path in tqdm(paths):
        img = cv2.imread(path)
        objects = inference_tta(detection_model, img)
    print(len(objects))

    detection_model = DetectionModel_TFLite()
    input_dir = "C:/home/workspace/BallRecognitionApp/backend/img"
    paths = glob.glob(input_dir + "/*.png")
    print(paths)
    for path in tqdm(paths):
        img = cv2.imread(path)
        objects = inference_tta(detection_model, img)
    print(len(objects))

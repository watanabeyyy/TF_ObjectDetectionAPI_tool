# TensorFlow Object Detection API Tools

# CAUTION
ssdlite_mobilenet_v3_smallしか試していない
-> 画像サイズ320x320等、コード内で直接指定している箇所があるため、他のモデルで不整合が生じる可能性あり

# References
- https://github.com/karaage0703/object_detection_tools

# 動作環境
python 3.6
tensorflow 1.14.0

# 事前準備

## フォルダ構成
```
models   <-https://github.com/tensorflow/models.git からclone
│
TF_ObjectDetectionAPI_tools
├─ config   <-configファイルを入れておく
│  └─ ssdlite_mobilenet_v3_small_320x320_coco.config
├─ models   <-finetuning用に学習済みモデルを保存しておく
├─ data   <-pascalvoc_to_tfrecord.pyによって生成されたdatasetが格納される
│  ├─ tf_label_map.pbtxt   <-labelのidとnameを紐づけるファイル、適切に中身を書き換えておく
│  ├─ train   <-空のフォルダを作っておく
│  └─ val   <-空のフォルダを作っておく
├─ utils
├─ pascalvoc_to_tfrecord.py
└─  inference.py
```



## TensorFlow Object Detection API のセットアップ

### git cloneとpath
```
git clone https://github.com/tensorflow/models.git
```
```
"PATH": "${env:PATH};${workspaceRoot};${workspaceRoot}/../models/research;${workspaceRoot}/../models/research/slim"
```

### protocolbufferのインストール
https://github.com/protocolbuffers/protobuf/releases
からprotocの適切なバージョンをとってくる。

protocで.protoのコンパイル
```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

### pycocotoolsのインストール
```
pip install cython
git clone https://github.com/cocodataset/cocoapi.git
python setup.py build_ext install
```
windowsではエラーが発生したので下記対応し再度実行。

#### warning回避
コンパイラのオプションを変更で回避できる。
引っかかるのはワーニングのオプションなのでcocoapi/PythonAPI/setup.py の extra_compile_args を以下に変更します。
```
ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=[],
    )
]
```

### Object Detection APIの動作確認

```
python object_detection/builders/model_builder_test.py
```
これでエラーがでなければOK

## 学習につかうTFrecordを作成する
※現時点ではpngのみ対応

学習用のデータセットとして、PascalVOC形式のxmlファイルからTFrecordを作成する。
./data/trainと./data/valに保存される。
```
python pascalvoc_to_tfrecord.py ^
--input_img_dir path_to_png_dir ^
--input_xml_dir path_to_xml_dir
```
アノテーションにはlabelimgの使用を推奨。

## ラベル情報のファイルを作成する
tf_label_map.pbtxtがラベル情報のファイル。train時に使うため適切に更新しておく。フォーマットは下記。
```
item {
 id: 1
 name: 'white'
}
item {
 id: 2
 name: 'black'
}
```

## 学習済みモデルのダウンロード
ここから好きなモデルをダウンロードして、modelsに保存
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
```
├─ models
│  └─ ssd_mobilenet_v3_small_coco_2019_08_14
│     ├─ checkpoint
│     ├─ frozen_inference_graph.pb
│     ├─ model.ckpt.data-00000-of-00001
│     ├─ model.ckpt.index
│     ├─ model.ckpt.meta
│     ├─ model.tflite
│     └─ pipeline.config
```


# 学習
```
python ../models/research/object_detection/model_main.py ^
--pipeline_config_path="./config/ssdlite_mobilenet_v3_small_320x320_coco.config" ^
--model_dir="./saved_model"
```
学習条件の設定はconfigファイルで行う。


## configファイルの詳細
この辺見ればtrain configの詳細が分かる
https://github.com/tensorflow/models/blob/master/research/object_detection/protos/train.proto

## error
trainの最後の処理のモデルの保存でエラーが発生した。
```
tensorflow.python.framework.errors_impl.NotFoundError: Failed to create a directory: ./saved_model\export\Servo\temp-b'1578107225'; No such file or directory
```
windows特有？
致命的ではないが、saved_modelフォルダに事前に下記を作っておけば回避可能。
```
mkdir saved_model\export\Servo
```

## tensorboard
```
tensorboard --logdir=./saved_model
```

# 推論
## 推論用モデルの作成
checkpointから推論用の.pbを作成する。出力名は`frozen_inference_graph.pb`
```
python ../models/research/object_detection/export_inference_graph.py ^
--input_type image_tensor ^
--pipeline_config_path ./config/ssdlite_mobilenet_v3_small_320x320_coco.config ^
--trained_checkpoint_prefix ./saved_model/model.ckpt-10000 ^
--output_directory ./exported_graphs 
```

## inference
※png画像のみ対応。
```
python inference.py ^
--model ./exported_graphs/frozen_inference_graph.pb ^
--input_dir path_to_png_dir
```

# TFLite向けのモデル出力
## export .pb for tflite

```
python ../models/research/object_detection/export_tflite_ssd_graph.py ^
--pipeline_config_path=./config/ssdlite_mobilenet_v3_small_320x320_coco.config ^
--trained_checkpoint_prefix=./saved_model/model.ckpt-10000 ^
--output_directory=./exported_graphs/tflite ^
--max_detections=100 ^
--add_postprocessing_op=true 
```

## quantize
これで.tfliteが作成できる
```
tflite_convert ^
--output_file=./exported_graphs/tflite/tflite_graph.tflite ^
--graph_def_file=./exported_graphs/tflite/tflite_graph.pb ^
--inference_type="FLOAT" ^
--input_arrays=normalized_input_image_tensor ^
--input_shapes="1,320,320,3" ^
--output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 ^
--default_ranges_min=0 ^
--default_ranges_max=6 ^
--mean_values=128 ^
--std_dev_values=128 ^
--allow_custom_ops
```

### output_arraysの意味
The outputs of the quantized model are named 
'TFLite_Detection_PostProcess',
'TFLite_Detection_PostProcess:1', 
'TFLite_Detection_PostProcess:2',
'TFLite_Detection_PostProcess:3' 
and represent four arrays:
detection_boxes, detection_classes, detection_scores, and num_detections.

### inference
※png画像のみ対応。
```
python inference.py ^
--model ./exported_graphs/tflite/tflite_graph.tflite ^
--input_dir path_to_png_dir
```

### 速度比較
画像3枚分測定
#### 環境
mobilenetv3
windows
cpu (core i5)

#### 結果
tflite
```
inference_time =  0.10678958892822266 [sec]
inference_time =  0.07995891571044922
inference_time =  0.08995890617370605
```

not tflite(.pb)
```
inference_time =  0.1238710880279541
inference_time =  0.08998894691467285
inference_time =  0.08995270729064941
```
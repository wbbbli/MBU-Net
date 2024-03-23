import cv2
import json
import random
import numpy as np
from pathlib import Path
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

# 语义分割的coco数据集
class CocoDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0, **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        self.split = split
        self.dataset_path = Path(dataset_path)
        self.stuff_prob = stuff_prob

        self.load_samples()

    def load_samples(self):
        annotation_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}2017.json'
        self.labels_path = self.dataset_path / 'annotations' / f'panoptic_{self.split}2017'
        self.images_path = self.dataset_path / f'{self.split}2017'

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        self.dataset_samples = annotation['annotations']  # 标注信息

        self._categories = annotation['categories']  # 所有的类别
        self._stuff_labels = [x['id'] for x in self._categories if x['isthing'] == 0]   # 遍历所有类别，分为stuff和非stuff类
        self._things_labels = [x['id'] for x in self._categories if x['isthing'] == 1]
        self._things_labels_set = set(self._things_labels)  # 输入可迭代对象，返回无序不重复元素集
        self._stuff_labels_set = set(self._stuff_labels)
        # print(self._things_labels_set)

    def get_sample(self, index) -> DSample:
        dataset_sample = self.dataset_samples[index]  # 根据下表来获取对应图片的标注

        image_path = self.images_path / self.get_image_name(dataset_sample['file_name'])  # annotation 里面的图片信息
        label_path = self.labels_path / dataset_sample['file_name']

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)  # label 的图片是什么
        label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]  # 为什么要这样

        instance_map = np.full_like(label, 0)
        things_ids = []
        stuff_ids = []

        for segment in dataset_sample['segments_info']:  # 遍历图片里面的每个标注
            # 'segments_info' 里面包含对图片里面所有实例对象的分割，会显示对象的id，可以根据这个自己处理成语义分割的标注
            class_id = segment['category_id']
            obj_id = segment['id']  # id 表示的是什么， 可能是一种约定俗成的转换格式？
            if class_id in self._things_labels_set:
                if segment['iscrowd'] == 1:  # iscrow为1表示一堆东西堆在一起，难以逐个分开（比如一车香蕉），所以使用了这个标注
                    continue  # 可能因为是实例分割，所以不加入这种难分的类
                things_ids.append(obj_id)
            else:
                stuff_ids.append(obj_id)  # 天空之类没有明确边界的类别，不存在 iscrowd

            instance_map[label == obj_id] = obj_id  # 看后面怎么处理 obj_id
        # obj_id可能和掩模里图片的颜色有关
        if self.stuff_prob > 0 and random.random() < self.stuff_prob:  # 看是否加入stuff类，或随机加入
            instances_ids = things_ids + stuff_ids
        else:
            instances_ids = things_ids

            for stuff_id in stuff_ids:  # 不加入stuff 时额外再处理
                instance_map[instance_map == stuff_id] = 0  # 0为背景类，相当于把stuff改为背景类

        return DSample(image, instance_map, objects_ids=instances_ids)  # 返回一个初始化好的类，类中记录了该index对应图片标注的信息

    # image 为图片，instance_map 为id构成的图，其中不同实例的id都不相同，背景为0，instances_ids为已有的实例对象的id

    @classmethod
    def get_image_name(cls, panoptic_name):
        return panoptic_name.replace('.png', '.jpg')

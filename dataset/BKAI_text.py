import re
import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from utils import strs
from dataset.data_util import pil_load_img
from dataset.dataload import Text_dataset, TextInstance
from util.io import readlines
from util.misc import norm2


class BKAIText(TextDataset):

    def __init__(self, data_path, is_training=True, transform=None, ignore_list=None):
        super(BKAIText, self).__init__(transform, is_training)
        self.data_path = data_path 
        self.data_list = self.export_data(self.data_path)
        self.is_training = is_training

        if is_training:
            self.train_list, self.val_list = train_test_split(self.data_list, shuffle=True,
                                                              test_size=0.25, random_state=2000)

        else:
            self.test_list = glob(data_path + "/*.jpg")

    @staticmethod
    def parse_text(gt_path):
        polygons = []
        with open(gt_path, "r") as F:
            contents = F.readlines()
            for line in contents:
                poly = []
                content = line.split(",")
                poly.extend([int(i) for i in content[:8]])
                text_label = line.replace(",".join(content[:8]) + ",", "")
                xx = [poly[2 * i] for i in range(4)]
                yy = [poly[2 * i + 1] for i in range(4)]
                pts = np.stack([xx, yy]).T.astype(np.int32)
                d1 = norm2(pts[0] - pts[1])
                d2 = norm2(pts[1] - pts[2])
                d3 = norm2(pts[2] - pts[3])
                d4 = norm2(pts[3] - pts[0])
                if min([d1, d2, d3, d4]) < 2:
                    continue
                polygons.append(TextInstance(pts, 'c', text_label))

        return polygons
    def export_data(self, data_folder):
        data = []

        # Training data from BKAI
        BKAI_img_folder = os.path.join(data_folder, "training_img/")
        BKAI_gt_folder = os.path.join(data_folder, "training_gt/")

        for gt_file in os.listdir(BKAI_gt_folder):
            image_file = gt_file[3:-4] + ".jpg" 
            item = {}
            item["image_path"] = os.path.join(BKAI_img_folder, image_file)
            item["gt_path"] = os.path.join(BKAI_gt_folder, gt_file)
            data.append(item)

        # Training data from VINAI
        VINAI_gt_folder = os.path.join(data_folder, "vietnamese/labels/")
        list_images = glob(os.path.join(data_folder, "vietnamese") + "/*/*.jpg")

        for image_file in list_images:
            image_idx = int(image_file.split("/")[-1][2:-4])
            gt_file = "".join(["gt_", str(image_idx), ".txt"]) 
            item = {}
            item["image_path"] = image_file
            item["gt_path"] = os.path.join(VINAI_gt_folder, gt_file)
            data.append(item)

        return data

    def __getitem__(self, index):
        if self.is_training:
            image_path = self.train_list[index]["image_path"]
            image = pil_load_img(image_path)

            try:
                # Read annotation
                gt_path = self.train_list[index]["gt_path"]
                polygons = self.parse_txt(gt_path)
            except:
                polygons = None
        else:
             return self.test_list[index]

        return self.get_training_data(image, polygons, image_path.split("/")[-1], image_path)

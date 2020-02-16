import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import avg_iou
from kmeans import kmeans

# 根文件夹
ANNOTATIONS_PATH = "../VOCdevkit/VOC2007/Annotations/"
# 聚类的数目
CLUSTERS = 9
# 模型中图像的输入尺寸
SIZE = 416

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) / width
			ymin = int(obj.findtext("bndbox/ymin")) / height
			xmax = int(obj.findtext("bndbox/xmax")) / width
			ymax = int(obj.findtext("bndbox/ymax")) / height

			xmin = np.float64(xmin)
			ymin = np.float64(ymin)
			xmax = np.float64(xmax)
			ymax = np.float64(ymax)
			if xmax == xmin or ymax == ymin:
				print(xml_file)
			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

if __name__ == "__main__":
	data = load_dataset(ANNOTATIONS_PATH)
	out = kmeans(data, k=CLUSTERS)

	# out = out.astype('int').tolist()
	# out = sorted(out, key=lambda x: x[0] * x[1])

	# print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
	# print("Boxes:\n {}".format(out))
	#
	# ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
	# print("Ratios:\n {}".format(sorted(ratios)))
	print(out)
	print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
	print("Boxes:\n {} \n {}".format(out[:, 0] * SIZE, out[:, 1] * SIZE))

	ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
	print("Ratios:\n {}".format(sorted(ratios)))
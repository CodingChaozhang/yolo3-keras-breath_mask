from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image
import tqdm

yolo = YOLO()

test_file = "test_imglist.txt"
img_file = "./img"

with open(test_file) as f:
    lines = f.readlines()
    n_test = len(lines)
    print("num of test images:",n_test)
    output_csv = open("sub.csv","a")

    for line in lines:
        img_path = line.strip()
        img_name = line.strip().split("/")[1].split(".")[0]
        # print(img_name)
        # print(img_path)
        # print(img_path)
        image = Image.open(img_path)
        predict_img_info = yolo.detect_images(image)
        predict_img_info = str(predict_img_info).strip("[").strip("]").replace("'","")
        #print(predict_img_info)
        text = img_name + ".jpg" + "," + str(predict_img_info) + "\n"
        output_csv.write(text)
    output_csv.close()
yolo.close_session()
print("all done")
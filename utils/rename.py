import os

label_path = "./label"

for label_name in os.listdir(label_path):
	new_name = label_name.replace("new_","")
	os.rename(os.path.join(label_path,label_name),os.path.join(label_path,new_name))
	

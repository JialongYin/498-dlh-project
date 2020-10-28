import os
import time
from bs4 import BeautifulSoup
import joblib
import random
import metapy
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from skimage.transform import resize
import pandas as pd
import numpy as np

def preprocess():
	random.seed(30)
	"""Extract x ray from dcm files and report from findings section of txt files"""
	directory = 'files'
	dataset = []
	word_bag = defaultdict(int)
	IMG_PX_SIZE = 128
	label_csv = pd.read_csv("./mimic-cxr-2.0.0-negbio.csv")
	for pxx in os.listdir(directory):
		pxx_dir = os.path.join(directory, pxx)
		for pat_id in os.listdir(pxx_dir):
			pat_id_dir = os.path.join(pxx_dir, pat_id)
			subject_id = int(pat_id[1:])
			"Only keep existing (report, x rays) mappings"
			file_set = set()
			dir_set = set()
			for f_or_d in os.listdir(pat_id_dir):
				f_or_d_name = os.path.join(pat_id_dir, f_or_d)
				if os.path.isdir(f_or_d_name):
					dir_set.add(f_or_d_name)
				else:
					file_set.add(f_or_d_name[:-4])
			std_dir_set = file_set.intersection(dir_set)
			for std_dir in std_dir_set:
				"""Extract REPORT"""
				report_dir = std_dir+'.txt'
				study_id = int(std_dir[-8:])
				f = open(report_dir, 'r')
				report_raw = f.read()
				"Extract findings part as report"
				start = report_raw.find("FINDINGS:") + len("FINDINGS:")
				end = report_raw.find("IMPRESSION:")
				findings = report_raw[start:end].lower()
				"parse report findings section using metapy"
				doc = metapy.index.Document()
				doc.content(findings)
				tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
				tok.set_content(doc.content())
				report = [token for token in tok]
				for token in report:
					 word_bag[token] += 1
				f.close()
				"""Extract X rays"""
				x_rays = []
				for std_img in os.listdir(std_dir):
					std_img_dir = os.path.join(std_dir, std_img)
					img = pydicom.dcmread(std_img_dir).pixel_array
					img = img / 4095
					resized_img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
					# plt.imshow(resized_img, cmap=plt.cm.bone)
					# plt.show()
					x_rays.append(resized_img)
				"""Extract cls"""
				cls = np.array(label_csv[label_csv['subject_id']==subject_id][label_csv['study_id']==study_id].iloc[0].tolist()[2:])
				cls = np.where(cls == 1, 1, 0)
				print(cls)
				dataset.append((report, x_rays, cls))
	dataset_len = len(dataset)
	random.shuffle(dataset)
	train = dataset[:int(0.7*dataset_len)]
	val = dataset[int(0.7*dataset_len):int(0.8*dataset_len)]
	test = dataset[int(0.8*dataset_len):]
	dataset_dir = "MIMIC_CXR_dataset/"
	os.makedirs(dataset_dir, exist_ok=True)
	joblib.dump((dataset,word_bag), dataset_dir+"dataset.pkl")
	joblib.dump((train,word_bag), dataset_dir+"train.pkl")
	joblib.dump((val,word_bag), dataset_dir+"val.pkl")
	joblib.dump((test,word_bag), dataset_dir+"test.pkl")

if __name__ == '__main__':
	tic = time.time()
	preprocess()
	print('[{:.2f}] Finish preprocessing'.format(time.time() - tic))

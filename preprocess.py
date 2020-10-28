import os
import time
import random
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd
import numpy as np

def preprocess():
	# data(csv) files root
	dataroot = './'
	# dataroot = '/shared/rsaas/jialong2/physionet.org_depre/files/mimic-cxr-jpg/2.0.0/'
	# load csv files: split, metadata, chexpert
	df_split = pd.read_csv(dataroot+"mimic-cxr-2.0.0-split.csv")
	df_metadata = pd.read_csv(dataroot+"mimic-cxr-2.0.0-metadata.csv")
	df_metadata = df_metadata[['dicom_id', 'subject_id', 'study_id', 'ViewPosition']]
	df_chexpert = pd.read_csv(dataroot+"mimic-cxr-2.0.0-chexpert.csv")
	# df_split:377110 df_metadata:377110 df_chexpert:227827
	print("df_split:{} df_metadata:{} df_chexpert:{}".format(len(df_split), len(df_metadata), len(df_chexpert)))

	# merge csv files
	df_spl_meta = pd.merge(df_split, df_metadata, on=["dicom_id", "subject_id", "study_id"], how="left")
	df_merge = pd.merge(df_spl_meta, df_chexpert, on=["subject_id", "study_id"], how="left")

	# filter csv files for ViewPosition = PA
	df_merge = df_merge.loc[df_merge['ViewPosition'] == 'PA']
	del df_merge['ViewPosition']
	# filter csv files for < 15000000 (optional)
	df_merge = df_merge.loc[df_merge['subject_id'] < 15000000]
	# df_merge: 47882
	print("df_merge:{}".format(len(df_merge)))


	# add reports column
	# report file root
	rptroot = dataroot+'mimic-cxr-reports/files'
	for i, row in df_merge.iterrows():
		rpt_path = os.path.join(rptroot, 'p'+str(row['subject_id'])[:2], 'p'+str(row['subject_id']), 's'+str(row['study_id'])+'.txt')
		f = open(rpt_path, 'r')
		report_raw = f.read()
		idx_fdg = report_raw.find("FINDINGS:")
		idx_impr = report_raw.find("IMPRESSION:")
		if idx_fdg == -1:
			start = idx_impr + len("IMPRESSION:")
			findings = '' if idx_impr == -1 else report_raw[start:]
		else:
			start = idx_fdg + len("FINDINGS:")
			findings = report_raw[start:] if idx_impr == -1 else report_raw[start:idx_impr]
		df_merge.at[i,'Text'] = findings
		if i % 1000 == 0:
			print('Already process {} reports!'.format(i))
	# filter csv files for non-empty text
	# Before: df_train:47019, df_test:459, df_validate:404
	# After only findings: df_train:9526, df_test:124, df_validate:101 !
	# After only findings or impression: df_train:45634, df_test:429, df_validate:383
	df_merge = df_merge.loc[df_merge['Text'] != '']
	print(df_merge.head(10))

	# splt train, test, valid
	# merged csv save root
	saveroot = 'MIMIC_CXR_dataset/'
	grouped = df_merge.groupby(df_merge.split)
	for split in ["train", "test", "validate"]:
		df = grouped.get_group(split)
		del df['split']
		df.to_csv(saveroot+split+'.csv', index=False)
		print("df_{}:{}".format(split, len(df)))


if __name__ == '__main__':
	tic = time.time()
	preprocess()
	print('[{:.2f}] Finish preprocessing'.format(time.time() - tic))

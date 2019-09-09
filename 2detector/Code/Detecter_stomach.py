import os
import random
import datetime
import sys
import yaml

import openslide
from DetecterModule import Detecter
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    configure_path = "./configure.yaml"
    kConFile_ = yaml.load(open(configure_path))
    output_dir  = kConFile_['DetecterParameters']["RuningConf"]['output_dir']
    DataDirList = kConFile_['DetecterParameters']["RuningConf"]['data_dir_list']
    # test_txt_file = kConFile_['DetecterParameters']["RuningConf"]['test_txt_file']
    # train_txt_file = kConFile_['DetecterParameters']["RuningConf"]['train_txt_file']
    slide_path_list = []
    detecter = Detecter(configure_path)
    detecter.initial_pipelineB_to_server()
    if not os.path.exists(DataDirList):
        print "\n[ERROR] Cannot find the image to be tested, Please check the data_dir_list in configure file."
        exit(0)
    imagelist = []
    filelist = os.listdir(DataDirList)
    for names in filelist:
        if names.endswith(".svs"):
            imagelist.append(names)

    slide_path_list = []
    for sld_fname in imagelist:
        slidePath = "%s/%s" % (DataDirList, sld_fname)
        print "\n==> Appending the Images to be detected. Name. ", sld_fname
        slide_path_list.append(slidePath)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    detecter.detect_batchly(slide_path_list, output_dir)
    detecter._reset_server()
    # print "Finished"
        
        
        

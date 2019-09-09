import os
from FCNDetecterModule_Paralelled import FCNDetecter
import yaml
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROI_num = 100
def extract(filelist, DataDirList, FCN_NET_FILE, FCN_TRAINED_MODEL_FILE, gpu_list):

    outputDir = "./Outputs/ROIs/features"
    cacheDir = "./Outputs/ROIs/"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    detecter = FCNDetecter(gpu_list)
    detecter.SetOutputDir(outputDir)
    detecter.SetCacheDir(cacheDir)
    detecter.SetFCNNetDeployFile(FCN_NET_FILE)
    if not os.path.exists(FCN_TRAINED_MODEL_FILE):
        print("!ERROR, no weight in", FCN_TRAINED_MODEL_FILE)
        exit(-1)
    detecter.SetFCNTrainedModelFile(FCN_TRAINED_MODEL_FILE)
    detecter.InitializeFCNDetecter()

    for sld_fname in filelist:
        dataName = sld_fname.split(".")[0]
        if os.path.exists("%s/npyDir/%s.npy" % (outputDir, dataName)):
            print "WARNING: %s feature has already existed!" % dataName
            continue
        slidePath = "%s/%s.svs" % (DataDirList, dataName)
        if not os.path.exists(slidePath):
            print "!ERROR: cannot find initial image svs data %s" % slidePath
            exit(-1)

        print("==> Start Extracting Features: %s" % (dataName))
        detecter.FCNDetect(slidePath)
    detecter.TerminateFCNDetecter()


if __name__ == '__main__':

    configure_path = "./configure.yaml"
    kConFile_ = yaml.load(open(configure_path))
    output_dir = kConFile_['ExtracterParameters']["RuningConf"]['output_dir']
    DataDirList = kConFile_['ExtracterParameters']["RuningConf"]['data_dir_list']
    gpu_list = kConFile_['DetecterParameters']['gpu_list']
    FCN_NET_FILE = kConFile_['ExtracterParameters']['net_path']
    FCN_TRAINED_MODEL_FILE = kConFile_['DetecterParameters']['weight_path']

    imagelist = []
    filelist = os.listdir(DataDirList)
    for names in filelist:
        if names.endswith(".svs"):
            imagelist.append(names)

    extract(imagelist, DataDirList, FCN_NET_FILE, FCN_TRAINED_MODEL_FILE, gpu_list)



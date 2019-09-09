import sys
import os
import cv2
import numpy as np
import openslide
import time

from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from tqdm import tqdm
from keras.layers import concatenate
import tensorflow as tf
sys.path.append("./utils/")
from myutils import GetPatch, GetROIList_by_OTSU_detection, ToMultiGPU

CPU_Kernel = 8

r_mean = 185
g_mean = 50
b_mean = 185

class PipelineA(multiprocessing.Process):
    def __init__(self, queue_PreA, queue_A2B_list, token_A2B, ParaList ):
        '''

        :param queue_PreA:  batch data
        :param queue_A2B_list:
        :param token_A2B:
        :param ParaList:
        '''
        multiprocessing.Process.__init__(self)
        self.queue_PreA = queue_PreA
        self.queue_A2B_list = queue_A2B_list
        self.token_A2B = token_A2B
        self._gpuGroupSize = ParaList[1]
        self._batchPara = ParaList[2]
        self._patch_size= ParaList[3]
        self._denseStride = ParaList[9]
        self._GetPatch_Palleled = ParaList[10]
        self._pipeNum_A, self._pipeNum_B, self._pipeNum_C = ParaList[15]

    def run(self):
        while self.queue_PreA.empty() == False:
            DataParaList, slideInfo_list, qid = self.queue_PreA.get(1)
            pool = ThreadPool(CPU_Kernel)
            slideTile_list = pool.map(self._GetPatch_Palleled, DataParaList)
            pool.close()
            pool.join()
            slideTile_sw = np.array(slideTile_list)
            self.queue_A2B_list[qid].put([slideTile_sw, slideInfo_list])
            self.token_A2B.put(qid)


class PipelineB(multiprocessing.Process):
    def __init__(self, queue_A2B_list, queue_B2C, token_A2B, signal_B2C, gpuGroups, pipeID, ParaListB ):
        multiprocessing.Process.__init__(self)
        self.queue_A2B_list = queue_A2B_list
        self.queue_B2C = queue_B2C
        self.token_A2B = token_A2B
        self.signal_B2C = signal_B2C

        self._netPath = ParaListB[0]
        self._weightPath = ParaListB[1]
        self._FCN_ROI_patch_stride = ParaListB[3]
        self._denseStride = ParaListB[4]
        self._denseRatio = ParaListB[5]
        self._pipeNum_A, self._pipeNum_B, self._pipeNum_C = ParaListB[6]
        self._batchSize = ParaListB[7]
        self._pipeID = pipeID
        self._gpuGroups = gpuGroups
        self._netModel = "NULL"

    def run(self):
        if self._netModel == "NULL":
            time.sleep(self._pipeID*5)

            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuGroups).strip('[').strip(']').replace(", ",",")
            NetFile = self._netPath.split("/")[-1].split(".py")[0]
            DeployDir = self._netPath.split(NetFile)[0]
            sys.path.append(DeployDir)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            self._netModel = __import__(NetFile).BuildModel(input_shape=(299, 299, 3), classes=3, isextract=True)

            if os.path.exists(self._weightPath):
                self._netModel.load_weights(self._weightPath, by_name=True)
        while True:
            qid = self.token_A2B.get(1)
            slideTile_sw, slideInfo_list = self.queue_A2B_list[qid].get(1)
            batchSum = slideTile_sw.shape[0]/self._batchSize+1 if slideTile_sw.shape[0]%self._batchSize else slideTile_sw.shape[0]/self._batchSize
            Probs_list = []
            for i_t in xrange(batchSum):
                i_start = i_t * self._batchSize
                i_end   = (i_t+1) * self._batchSize
                i_end = slideTile_sw.shape[0] if i_end > slideTile_sw.shape[0] else i_end
                Probs = self._netModel.predict(slideTile_sw[i_start:i_end,:,:,:])
                Probs_list.append(Probs)
            self.queue_B2C.put([Probs_list,slideInfo_list])

class PipelineC(multiprocessing.Process):
    def __init__(self, ev_EndAll, queue_PreA, queue_A2B_list, queue_B2C, token_A2B, signal_B2C, ParaList ):
        multiprocessing.Process.__init__(self)
        self._ev_EndAll = ev_EndAll
        self.queue_PreA = queue_PreA
        self.queue_A2B_list = queue_A2B_list
        self.queue_B2C = queue_B2C
        self.signal_B2C = signal_B2C
        self.token_A2B = token_A2B

        self._offsetROI_List = ParaList[0]
        self._gpuGroupSize = ParaList[1]
        self._batchPara = ParaList[2]
        self._patch_size= ParaList[3]
        self._Prob_Shape  = ParaList[4]
        self._FCN_threshold = ParaList[5]
        self._outputDir = ParaList[6]
        self._gpuList = ParaList[7]
        self._netPath = ParaList[8]
        self._denseStride = ParaList[9]
        self._weightPath = ParaList[12]
        self._FCN_ROI_patch_stride = ParaList[13]
        self._denseRatio = ParaList[14]
        self._pipeNum_A, self._pipeNum_B, self._pipeNum_C = ParaList[15]
        self._dataName = ParaList[16]
        self._ClassName = ParaList[17]
        self._ClassFreqBand = ParaList[18]
        self._ClassColor = ParaList[19]

    def run(self):
        npyDir =  "%s/npyDir" %self._outputDir
        if os.path.exists(npyDir) == False:
            os.system("mkdir %s" %npyDir)
        npyPath = "%s/%s.npy" %(npyDir, self._dataName)

        self._i_oR = 0
        Features_image = []
        flag_End_C = False
        FLAG = True
        while flag_End_C == False:
            Features_list, slideInfo_list = self.queue_B2C.get(1)
            Features = Features_list[0]
            for i_lst in xrange(1,len(Features_list)):
                Features = np.concatenate((Features, Features_list[i_lst]), axis=0)
            if FLAG:
                Features_image = Features
                FLAG = False
            else:
                Features_image = np.concatenate((Features_image, Features), axis=0)
            for i_pb in xrange(Features.shape[0]):
                self._i_oR = self._i_oR + 1
                queue_A2B_size = 0
                for i_t in xrange(len(self.queue_A2B_list)):
                    queue_A2B_size += self.queue_A2B_list[i_t].qsize()
                if self._i_oR == len(self._offsetROI_List):
                    flag_End_C = True

        Features_image = np.squeeze(Features_image)
        np.save(npyPath, Features_image)
        self._ev_EndAll.set()


class FCNDetecter:
    def __init__(self, gpu_list):
        '''
        Constructor
        '''
        self._gpuList = gpu_list

        self._outputDir = "."
        self._cacheDir = "."

        self._ClassName     = {0:"Normal", 1:"Atypia", 2:"Cancer"}
        self._ClassFreqBand = {0: [0.0, 0.3], 1: [0.35, 0.65], 2: [0.7, 1]}   #Squeeze the channels into one channel
        self._ClassColor    = {0:[0,0,0], 1:[250,0,0], 2:[250,0,250]} #BGR


        self._avg_list = [[135,85,164],[155,100,135],[165,125,151],[187,155,180],[188,130,160],[145,88,122],[110,75,115],[67,28,120],[100,47,140],[100, 64, 110],[108,95,172],[70,50,150],[188,108,190]]
        self._level_downsamples = [1, 4, 16, 64]
        self._outputDir = self._cacheDir = ""

        self._FCN_threshold = 0.5    #Threshold of Positive
        self._FCN_win_size = 299
        self._FCN_workingLevel = 0
        self._FCN_OV_level = 2
        self.gain = 2
        self._FCN_stride = 32

        self._denseRatio = 1  #2 or 4 or 8
        self._denseStride = self._FCN_stride/self._denseRatio  #299

        ### Parelleled definition
        self._batchPara = 20
        self._gpuGroupSize = 1
        self._packA2BSize = 2
        if len(self._gpuList)>=6:
            self._pipeNum_A = 3
        else:
            self._pipeNum_A = 1
        self._pipeNum_B = len(self._gpuList)/self._gpuGroupSize
        self._pipeNum_C = 1
        self._queue_A2B_LEN = 4
        self._queue_PreA = multiprocessing.Queue(-1)
        self._queue_A2B_list = []
        for i_t in xrange(self._queue_A2B_LEN):
            self._queue_A2B = multiprocessing.Queue(1)
            self._queue_A2B_list.append(self._queue_A2B)
        self._queue_B2C = multiprocessing.Queue(8)
        self._token_A2B = multiprocessing.Queue(-1)
        self._signal_B2C = multiprocessing.Queue(-1)


    def SetOutputDir(self, outputDir):
        '''Set the Output Dir
        '''
        self._outputDir = outputDir
        if os.path.exists(self._outputDir) == False:
            os.system("mkdir " + self._outputDir)

    def SetCacheDir(self, cacheDir):
        '''Set the Cache Dir
           It will speed up the re-calculation
        '''
        self._cacheDir = cacheDir
        if os.path.exists(self._cacheDir) == False:
            os.system("mkdir " + self._cacheDir)

    def SetFCNThreshold(self, FCNThreshold):
        '''Set the positive Threshold of FCN Result .
        '''
        self._FCN_threshold = FCNThreshold

    def SetFCNNetDeployFile(self, FCNnetDeployFile):
        '''Set the path of FCN deploy file (.prototxt)
        '''
        self._netPath = FCNnetDeployFile

        NetFile = self._netPath.split("/")[-1].split(".py")[0]
        DeployDir = self._netPath.split(NetFile)[0]
        sys.path.append(DeployDir)
        self._FCN_ROI_patch_size = int(__import__(NetFile).roi_size)     # 448
        self._FCN_prob_patch_size = (self._FCN_ROI_patch_size-self._FCN_win_size)/self._FCN_stride+1   # 1
        self._FCN_ROI_patch_stride  = self._FCN_stride * self._FCN_prob_patch_size   # 448*1

    def SetFCNTrainedModelFile(self, FCNtrainedModelFile):
        '''Set the path of trained FCN model file (.caffemodel)
        '''
        self._weightPath = FCNtrainedModelFile

    def InitializeFCNDetecter(self):
        '''Initialize the FCN Detecter
           ***IMPORTANT***
           This function must be evoked before using FCNDetect()
        '''
        batchSize = self._gpuGroupSize*self._batchPara
        self._processedB = []
        ParaListB = []
        ParaListB.append(self._netPath)
        ParaListB.append(self._weightPath)
        ParaListB.append("N/A")
        ParaListB.append(self._FCN_ROI_patch_stride)
        ParaListB.append(self._denseStride)
        ParaListB.append(self._denseRatio)
        ParaListB.append([self._pipeNum_A, self._pipeNum_B, self._pipeNum_C])
        ParaListB.append(batchSize)

        i_tmp = 0
        id = 0
        pipeID = 0
        gpuGroup = []
        while id < len(self._gpuList):
            gpuGroup.append(self._gpuList[id])
            i_tmp += 1
            if i_tmp == self._gpuGroupSize:
                self._processedB.append(PipelineB( self._queue_A2B_list, self._queue_B2C, self._token_A2B, self._signal_B2C, gpuGroup, pipeID, ParaListB))
                pipeID += 1
                gpuGroup = []
                i_tmp = 0
            id += 1
        for i in range(len(self._processedB)):
            self._processedB[i].start()

    def TerminateFCNDetecter(self):
        '''Initialize the FCN Detecter
           ***IMPORTANT***
           This function must be evoked before using FCNDetect()
        '''
        for i in range(len(self._processedB)):
            self._processedB[i].terminate()
        for i in range(len(self._processedB)):
            self._processedB[i].join()
        del self._processedB[:]
        tf.keras.backend.clear_session()

    def _IsTissue(self, tile):
        '''To judge whether the patch is Tissue
            if it is tissue return True, elif it is background return False
        '''
        #Calculate the number of Tissue pixel...
        avg_list = self._avg_list
        r,g,b,a = cv2.split(tile)
        t_thres = 10
        t_list = []
        t_tissue = np.zeros(b.shape)
        for i in xrange(len(avg_list)):
            t1 = np.zeros(b.shape)
            t2 = np.zeros(b.shape)
            t3 = np.zeros(g.shape)
            t4 = np.zeros(g.shape)
            t5 = np.zeros(r.shape)
            t6 = np.zeros(r.shape)
            t1[r>avg_list[i][0]-t_thres] = 1
            t2[r<avg_list[i][0]+t_thres] = 1
            t3[g>avg_list[i][1]-t_thres] = 1
            t4[g<avg_list[i][1]+t_thres] = 1
            t5[b>avg_list[i][2]-t_thres] = 1
            t6[b<avg_list[i][2]+t_thres] = 1
            t7 = t1+t2+t3+t4+t5+t6
            t_tmp = np.zeros(r.shape)
            t_tmp[t7 > 5.5] = 1
            t_list.append(t_tmp)
        for i in xrange(len(t_list)):
            t_tissue = t_tissue + t_list[i]
        t_tissue[t_tissue>=1] = 1
        #Judge White Image
        tr = np.zeros(r.shape)
        tg = np.zeros(g.shape)
        tb = np.zeros(b.shape)
        tr[r>245] = 1
        tg[g>245] = 1
        tb[b>245] = 1
        t_overall = tr+tg+tb
        t_white = np.zeros(r.shape)
        t_white[t_overall > 2.5] = 1
        thres = b.shape[0]*b.shape[1]*(15.0/16.0)
        '''
        if r.mean() > 240 and b.mean()>240 and g.mean()>240:
            white_flag = True
        else:
            white_flag = False
        '''
        if(t_tissue.sum()>5): #and white_flag == False ):
            return True
        else:
            return False

    def _GetPatch_Palleled(self, DataParaList):
        start_w = DataParaList[0]
        start_h = DataParaList[1]
        windowShape  = DataParaList[2]
        workingLevel = DataParaList[3]
        slideTile = GetPatch(self._slide, start_w, start_h, windowShape, workingLevel)
        slideTile = slideTile[::2,::2,:]

        ##initial
        slideTile = slideTile.astype('float32')
        r, g, b, a = cv2.split(slideTile)
        # return cv2.merge([r-r_mean,g-g_mean,b-b_mean])
        return cv2.merge([r, g, b])/127.5 - 1.0

    def FCNDetect(self, slidePath):
        '''Detect the patches by FCNVGG net to find out the candidates
        '''
        #Clear all the queues
        while self._queue_PreA.empty() == False:
            self._queue_PreA.get(1)
        for i_t in xrange(len(self._queue_A2B_list)):
            while self._queue_A2B_list[i_t].empty() == False:
                self._queue_A2B_list[i_t].get(1)
        while self._queue_B2C.empty() == False:
            self._queue_B2C.get(1)
        while self._token_A2B.empty() == False:
            self._token_A2B.get(1)
        while self._signal_B2C.empty() == False:
            self._signal_B2C.get(1)

        self._slide = slide = openslide.open_slide(slidePath)
        max_level = slide.level_count - 1
        if(self._FCN_workingLevel>max_level or self._FCN_workingLevel<0):
            print "the level to fetch data is out of the range of TIFF image"
            return 0

        slideFileName = slidePath.split('/')[-1]
        postfix = slideFileName.split('.')[-1]
        dataName = slideFileName.split('.%s' %postfix)[0]

        npyPath =  "%s/%s_feature.npy" %(self._outputDir, dataName)
        if os.path.exists(npyPath) == True:
            print "feature of %s has been extracted before, Skip this Prediction!" % npyPath
            return False

        zero_level_size = slide.level_dimensions[0]

        ROI_list = []
        ROITxtDir = "%s/coordinates" %(self._cacheDir)
        ROITxtPath = "%s/%s.txt" %(ROITxtDir, dataName)
        if os.path.exists(ROITxtPath):
            #read coordinates from .txt file
            file1 = open( ROITxtPath,'r')
            t_ROI_lines = file1.readlines()
            for i in xrange(len(t_ROI_lines)):
                line = t_ROI_lines[i]
                elems = line.rstrip().split('\t')
                # left up point
                wcoor_t = int(elems[0])
                hcoor_t = int(elems[1])
                w_size  = int(elems[2])
                h_size  = int(elems[3])
                ROI_list.append([wcoor_t, hcoor_t, w_size, h_size])

        zero_width, zero_height = zero_level_size
        Prob_Shape = (((zero_height-self._FCN_ROI_patch_size*self.gain)/int(self._FCN_ROI_patch_stride*self.gain)+2)*self._FCN_prob_patch_size * self._denseRatio, ((zero_width-self._FCN_ROI_patch_size*self.gain)/int(self._FCN_ROI_patch_stride*self.gain)+2)*self._FCN_prob_patch_size * self._denseRatio )
        offsetROI_List = []
        patch_size = self._FCN_ROI_patch_size * self.gain
        for i_roi in xrange(len(ROI_list)):
            wcoor = ROI_list[i_roi][0]
            hcoor = ROI_list[i_roi][1]
            w_size  = ROI_list[i_roi][2]
            h_size  = ROI_list[i_roi][3]

            for w_offset in xrange(0, self._denseRatio):
                for h_offset in xrange(0, self._denseRatio):
                    offsetROI_List.append([wcoor, hcoor, w_size, h_size, w_offset, h_offset])

        batchSize = self._gpuGroupSize*self._batchPara
        i_ba = 0
        slideInfo_list = []
        DataParaList = []
        i_oR = 0
        qid = 0
        while i_oR < len(offsetROI_List):
            if i_ba < batchSize * self._packA2BSize:
                wcoor = offsetROI_List[i_oR][0]
                hcoor = offsetROI_List[i_oR][1]
                w_size = offsetROI_List[i_oR][2]
                h_size = offsetROI_List[i_oR][3]
                w_offset = offsetROI_List[i_oR][4]
                h_offset = offsetROI_List[i_oR][5]
                wcoor_off = wcoor+ w_offset * self._denseStride
                hcoor_off = hcoor+ h_offset * self._denseStride
                DataParaList.append( [wcoor_off, hcoor_off, (w_size,h_size), 0] )
                slideInfo_list.append([((wcoor)/int(self._FCN_ROI_patch_stride*self.gain)*self._FCN_prob_patch_size),((hcoor)/int(self._FCN_ROI_patch_stride*self.gain)*self._FCN_prob_patch_size),self._FCN_prob_patch_size,self._FCN_prob_patch_size,w_offset,h_offset])
                i_ba = i_ba + 1
                i_oR = i_oR + 1
            if i_ba == batchSize * self._packA2BSize or i_oR == len(offsetROI_List):
                i_ba = 0
                self._queue_PreA.put([DataParaList, slideInfo_list, qid])
                qid += 1
                if qid >= self._queue_A2B_LEN:
                    qid = 0
                slideInfo_list = []
                DataParaList = []

        ev_EndAll = multiprocessing.Event()
        ev_EndAll.clear()
        ParaList = []
        ParaList.append(offsetROI_List)
        ParaList.append(self._gpuGroupSize)
        ParaList.append(self._batchPara)
        ParaList.append(patch_size)
        ParaList.append(Prob_Shape)
        ParaList.append(self._FCN_threshold)
        ParaList.append(self._outputDir)
        ParaList.append(self._gpuList)
        ParaList.append(self._netPath)
        ParaList.append(self._denseStride)
        ParaList.append(self._GetPatch_Palleled)
        ParaList.append("N/A")
        ParaList.append(self._weightPath)
        ParaList.append(self._FCN_ROI_patch_stride)
        ParaList.append(self._denseRatio)
        ParaList.append([self._pipeNum_A, self._pipeNum_B, self._pipeNum_C])
        ParaList.append(dataName)
        ParaList.append(self._ClassName)
        ParaList.append(self._ClassFreqBand)
        ParaList.append(self._ClassColor)

        processedA = []
        processedC = []
        for i_num in xrange(self._pipeNum_A):
            processedA.append(PipelineA( self._queue_PreA, self._queue_A2B_list, self._token_A2B, ParaList))
        processedC.append(PipelineC( ev_EndAll, self._queue_PreA, self._queue_A2B_list, self._queue_B2C, self._token_A2B, self._signal_B2C, ParaList))

        processed = processedA + processedC
        #start processes
        for i in range(len(processed)):
            processed[i].start()

        ev_EndAll.wait()
        if ev_EndAll.is_set():
            for i in range(len(processed)):
                processed[i].terminate()
        #join processes
        for i in range(len(processed)):
            processed[i].join()

    

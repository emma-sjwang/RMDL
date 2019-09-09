import sys
import os
import cv2
import numpy as np
import openslide
import yaml
import time
import random
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('classic')
from scipy.ndimage.filters import gaussian_filter
sys.path.append("./utils/")
from myutils import GetPatch, GetROIList_by_OTSU_detection, ToMultiGPU, _get_ROIs

gpu_list_str = '0'
gpu_list = []
if gpu_list_str.split("-")[0] == 'all':
    gpu_num = int(gpu_list_str.split("-")[1])
    for i_gpu in xrange(gpu_num):
        gpu_list.append(i_gpu)
else:
    for i_gpu in gpu_list_str.split(","):
        gpu_list.append(int(i_gpu))
threshold_in = 0.7
threshold_out = 0.0
FCNWinSize = 299
ROIPatchStride = 32
openingSize = 2
level = 2
mask_level = 2 * level

CPU_Kernel = 4

def _overlay_prob(prob_img, pixelarray):
    pixelarray = np.copy(pixelarray[:,:,0:3])
    for x in range(prob_img.shape[0]):
        for y in range(prob_img.shape[1]):
            if prob_img[x, y, 0] == 128 and prob_img[x, y, 1] == 0 and prob_img[x, y, 2] == 0:
                continue
            else:
                # ignore background
                if pixelarray[x, y, 0] > 200 and pixelarray[x, y, 1] > 200 and pixelarray[x, y, 2] > 200:
                    continue
                else:
                    pixelarray[x, y, :] = prob_img[x, y, :]
    return pixelarray


def _extend_prob(prob_map, pixelarray, index=2):
    image = np.zeros([pixelarray.shape[0], pixelarray.shape[1], 3])
    for x in range(prob_map.shape[0]):
        for y in range(prob_map.shape[1]):
            overlay_x = int((x * ROIPatchStride * 2 ) / pow(4, level))
            overlay_y = int((y * ROIPatchStride * 2) / pow(4, level))
            patch_size = 2
            # if prob_map[x, y, index] > 0.3:
            image[overlay_x-patch_size:overlay_x+patch_size, overlay_y-patch_size:overlay_y+patch_size, :] += prob_map[x,y, index]
    return image

def overlay(slide, prob, output_path, slide_name):
    pixelarray = np.array(slide.read_region((0, 0), 2, slide.level_dimensions[2]))
    label_list = ['_normal', '_dysplasia', '_cancer']
    for i in range(prob.shape[2]):
        ext_prob = _extend_prob(prob, pixelarray, i)
        im_color = cv2.applyColorMap(np.array(ext_prob[:,:,i] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_prob = _overlay_prob(im_color, pixelarray)
        cv2.imwrite(os.path.join(output_path, slide_name + label_list[i]+ '_overlay.png'), overlay_prob)

def prob2heatmap(prob, output_path, slide_name, i):
    expand_img = gaussian_filter(prob, 1)
    plt.imshow(expand_img)
    plt.clim(0, 1)
    plt.colorbar()
    plt.axis('off')
    if i == 0 :
        heatMap_name = '%s/%s_exp_normal.png' % (output_path, slide_name)
    elif i == 1 :
        heatMap_name = '%s/%s_exp_atypia.png' % (output_path, slide_name)
    else:
        heatMap_name = '%s/%s_exp_cancer.png' % (output_path, slide_name)
    plt.savefig(heatMap_name, transparent=True)
    plt.clf()

class DetecterParameterBase:
    def __init__(self, configure_path):
        self._configure_path_ = configure_path
        self._kConFile_ = yaml.load(open(configure_path))
        ''' Common Trainer Parameters'''
        self._kThreadNum_ = self._kConFile_['CommonParameters']['thread_num']
        self._kClassName_ = self._kConFile_['CommonParameters']['ClassInfo']['name']
        self._kClassLabel_ = self._kConFile_['CommonParameters']['ClassInfo']['label']
        self._kClassFrequencyBand_ = self._kConFile_['CommonParameters']['ClassInfo']['frequency_band']
        self._kRMean_ = self._kConFile_['CommonParameters']['ColorMean']['r_mean']
        self._kGMean_ = self._kConFile_['CommonParameters']['ColorMean']['g_mean']
        self._kBMean_ = self._kConFile_['CommonParameters']['ColorMean']['b_mean']
        self._kWorkingSpacing_ = self._kConFile_['CommonParameters']['LevelInfo']['spacing_of_working_level']
        self._kOverviewSpacing_ = self._kConFile_['CommonParameters']['LevelInfo']['spacing_of_overview_level']
        self._net_path_ = self._kConFile_['DetecterParameters']['net_path']
        self._weight_path_ = self._kConFile_['DetecterParameters']['weight_path']
        self._gpu_list_ = self._kConFile_['DetecterParameters']['gpu_list']
        #self._gpu_list_ = gpu_list
        self._gpus_per_group_ = self._kConFile_['DetecterParameters']['gpus_per_group']
        self._batch_size_ = self._kConFile_['DetecterParameters']['batch_size']
        self._batch_size_multipler_ = self._kConFile_['DetecterParameters']['batch_size_multipler']
        self._prob_threshold_ = self._kConFile_['DetecterParameters']['prob_threshold']
        self._queue_A2B_list_LEN_ = self._kConFile_['DetecterParameters']['queue_A2B_list_LEN']
        self._queue_B2C_LEN_ = self._kConFile_['DetecterParameters']['queue_B2C_LEN']
        self._queue_result_LEN_ = self._kConFile_['DetecterParameters']['queue_result_LEN']
        self._block_reinitial_time_ = self._kConFile_['DetecterParameters']['block_reinitial_time']
        
        ''' Compute the pipeline number '''
        self._pipelineA_Num_ = self._kConFile_['DetecterParameters']['pipelineA_number']
        self._pipelineB_Num_ = len(self._gpu_list_)/self._gpus_per_group_
        self._pipelineC_Num_ = 1
            
        
        
class PipelineA(multiprocessing.Process, DetecterParameterBase):
    def __init__(self, configure_path, parameter_list ):
        multiprocessing.Process.__init__(self)
        DetecterParameterBase.__init__(self, configure_path)
        
        '''Extract parameters from parameter_list'''
        self._queue_preA_ = parameter_list[0]
        self._token_preA_ = parameter_list[1]
        self._queue_A2B_list_ = parameter_list[2]
        self._token_A2B_ = parameter_list[3]
        self._token_result_ = parameter_list[4]
        self._slide_ = parameter_list[5]

    def _get_patch_paralleled(self, data_para_list):
        ''' Get patches paralleled by ThreadPool
        '''
        start_w_L0 = data_para_list[0]
        start_h_L0 = data_para_list[1]
        roi_ext_shape  = data_para_list[2]
        slide_working_level = data_para_list[3]
        level0_shape = [roi_ext_shape[0]*2, roi_ext_shape[1]*2]
        slide_tile = GetPatch(self._slide_, start_w_L0, start_h_L0, level0_shape, slide_working_level)
        slide_tile_downsampling = slide_tile[::2,::2,:]
        slide_tile_downsampling = slide_tile_downsampling.astype('float32')
        r,g,b,a = cv2.split(slide_tile_downsampling)
        return cv2.merge([r, g, b])/127.5 - 1.0


    def run(self):
        while self._queue_preA_.empty() == False:
            try:
                self._token_preA_.get(1, self._block_reinitial_time_)
                data_para_list, slide_info_list, qid = self._queue_preA_.get(1, self._block_reinitial_time_)
            except:
                print "#   #   # Blocked in pipelineA..Re-initialize the program.."
                print "#   #   # self._token_preA_.qsize=%d, self._queue_preA_.qsize=%d" %(self._token_preA_.qsize(), self._queue_preA_.qsize())
                self._token_result_.put("BlockError")
                return
            
            pool = ThreadPool(self._kThreadNum_)
            slide_tile_list = pool.map(self._get_patch_paralleled, data_para_list)
            pool.close()
            pool.join()

            self._queue_A2B_list_[qid].put([slide_tile_list, slide_info_list]) # put the data in qid-th queue
            self._token_A2B_.put(qid) # put the qid token into _token_A2B_

class PipelineB(multiprocessing.Process, DetecterParameterBase): 
    def __init__(self, configure_path, parameter_list, pipeline_ID, gpu_list_in_group):
        multiprocessing.Process.__init__(self)
        DetecterParameterBase.__init__(self, configure_path)
        self._pipeline_ID_ = pipeline_ID
        self._gpu_list_in_group_ = gpu_list_in_group
        '''Extract parameters from parameter_list'''
        self._queue_A2B_list_ = parameter_list[0]
        self._token_A2B_ = parameter_list[1]  
        self._queue_B2C_ = parameter_list[2]
        self._token_B2C_ = parameter_list[3]
        self._token_result_ = parameter_list[4]
        self._batch_size_ = parameter_list[5]
        self._class_num_ = parameter_list[-1]
        '''Model Parameters'''
        self._net_ = "NULL"


    def run(self):
        if self._net_ == "NULL":
            time.sleep(self._pipeline_ID_*5) # Avoid the initialization conflict.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_list_in_group_).strip('[').strip(']').replace(", ",",")
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.47
            session = tf.Session(config=config)

            net_file = self._net_path_.split("/")[-1].split(".py")[0]
            deploy_dir = self._net_path_.split(net_file)[0]
            sys.path.append(deploy_dir)
            self._net_s = __import__(net_file).BuildModel(weights='imagenet', classes=self._class_num_)
            self._net_ = ToMultiGPU(self._net_s, self._gpu_list_in_group_)
            if os.path.exists(self._weight_path_):
                print "\n\n==> Loading weights from ", self._weight_path_
                self._net_s.load_weights(self._weight_path_, by_name=True)
                print "==> Finish Loading weights. Begin predict image\n"
            else:
                print "!ERROR: not found weights file", self._weight_path_
                exit(-1)
        while True:
            try:
                qid = self._token_A2B_.get(1)
                slide_tile_arr, slide_info_list = self._queue_A2B_list_[qid].get(1)
                batch_number = len(slide_tile_arr)/self._batch_size_+1 if len(slide_tile_arr)%self._batch_size_ else len(slide_tile_arr)/self._batch_size_
                prob_tiles_list = []
                for i_ba in xrange(batch_number):
                    i_start = i_ba * self._batch_size_
                    i_end   = (i_ba+1) * self._batch_size_
                    i_end = len(slide_tile_arr) if i_end > len(slide_tile_arr) else i_end

                    prob_tiles = self._net_.predict(np.array(slide_tile_arr[i_start:i_end]))
                    prob_tiles_list.append(prob_tiles)
                self._queue_B2C_.put([prob_tiles_list, slide_info_list])
                self._token_B2C_.put(self._pipeline_ID_)
            except:
                print "!ERROR: in PipelineB"
                self._token_result_.put("BlockError")
            
class PipelineC(multiprocessing.Process, DetecterParameterBase): 
    def __init__(self, configure_path, parameter_list):
        multiprocessing.Process.__init__(self)
        DetecterParameterBase.__init__(self, configure_path)
        '''Extract parameters from parameter_list'''
        self._queue_preA_ = parameter_list[0]
        self._token_preA_ = parameter_list[1]
        self._queue_A2B_list_ = parameter_list[2]
        self._token_A2B_ = parameter_list[3]  
        self._queue_B2C_ = parameter_list[4]
        self._token_B2C_ = parameter_list[5]
        self._queue_result_ = parameter_list[6]
        self._token_result_ = parameter_list[7]
        self._batch_size_ = parameter_list[8]
        self._slide_level_downsamples_ = parameter_list[9]
        self._slide_working_level_ = parameter_list[10]
        self._WSI_prob_shape_ = parameter_list[11]
        self._inner_size_ = parameter_list[12]
        self._inner_stride_ = parameter_list[13]
        self._roi_size_ = parameter_list[14]
        self._roi_stride_ = parameter_list[15]
        self._dense_coef_   = parameter_list[16]
        self._dense_stride_ = parameter_list[17]
        self._offset_roi_list_LEN_ = parameter_list[18]
        self._roi_prob_size_ = parameter_list[19]
        self.iner_num = 0
        
    def run(self):
        work_DSP = self._slide_level_downsamples_[self._slide_working_level_]
        WSI_prob_map = np.zeros((self._WSI_prob_shape_[0]+50,
                                 self._WSI_prob_shape_[1]+50, 3), dtype='float32')
        # print "WSI_prob_map.shape:", WSI_prob_map.shape
        flag_Stop_C = False
        i_progress = 0
        pbar = tqdm(total=self._offset_roi_list_LEN_)
        while flag_Stop_C == False:
            try:
                prob_tiles_list, slide_info_list = self._queue_B2C_.get(1)
            except:
                print "#   #   # Blocked in pipelineC..Re-initialize the program.."
                self._token_result_.put("BlockError")
                return

            # Concatenate the prob tile list in one np array.
            prob_tiles_arr = prob_tiles_list[0]
            for i_lst in xrange(1, len(prob_tiles_list)):
                prob_tiles_arr = np.concatenate((prob_tiles_arr, prob_tiles_list[i_lst]), axis=0)
            # Stitching the prob tiles
            for i_pb in xrange(prob_tiles_arr.shape[0]):
                i_progress = i_progress + 1
                wcoor_lt_L0 = (slide_info_list[i_pb][0])
                hcoor_lt_L0 = (slide_info_list[i_pb][1])

                w_offset = slide_info_list[i_pb][2]
                h_offset = slide_info_list[i_pb][3]

                w_loc1 = int(((wcoor_lt_L0 + self._roi_size_) / 2) / (self._dense_stride_ * work_DSP) - (((self._inner_size_ - 299) / self._roi_stride_ + 1) * self._roi_stride_ * work_DSP / (self._dense_stride_ * work_DSP))/2)
                h_loc1 = int(((hcoor_lt_L0 + self._roi_size_)/ 2) / (self._dense_stride_ * work_DSP)- (((self._inner_size_ - 299) / self._roi_stride_ + 1) * self._roi_stride_ * work_DSP / (self._dense_stride_ * work_DSP))/2)
                w_loc2 = int(w_loc1 + ((((self._inner_size_ - 299) / self._roi_stride_ + 1) * self._roi_stride_ * work_DSP / (self._dense_stride_ * work_DSP))))
                h_loc2 = int(h_loc1 + ((((self._inner_size_ - 299) / self._roi_stride_ + 1) * self._roi_stride_ * work_DSP / (self._dense_stride_ * work_DSP))))
                WSI_prob_map[h_loc1:h_loc2, w_loc1:w_loc2][h_offset::self._dense_coef_, w_offset::self._dense_coef_, :] = prob_tiles_arr[i_pb, :, :, :]
                queue_A2B_qsize = 0
                for i_lst in xrange(len(self._queue_A2B_list_)):
                    queue_A2B_qsize += self._queue_A2B_list_[i_lst].qsize()

                if i_progress == self._offset_roi_list_LEN_:
                    flag_Stop_C = True
                self.iner_num += 1
                pbar.update(1)
                pbar.set_description("==> Predicting patches %d" % i_progress)
        self._queue_result_.put(WSI_prob_map)
        self._token_result_.put("Done!")


class Detecter(object, DetecterParameterBase):
    ''' Detect objects from WSI
    '''
    def __init__(self, configure_path):
        '''
        Constructor
        '''
        DetecterParameterBase.__init__(self, configure_path)
        
        ''''''
        self._processorB_ = "NULL"
        ''' Compute the parameters like roi_size and roi_stride from net deployment file'''
        net_file = self._net_path_.split("/")[-1].split(".py")[0]
        deploy_dir = self._net_path_.split(net_file)[0]
        sys.path.append(deploy_dir)
        self._roi_size_ = int(__import__(net_file).roi_size)
        self._roi_ext_size_ = int(__import__(net_file).roi_ext_size)
        self._inner_size_ = int(__import__(net_file).inner_size)
        self._inner_stride_ = int(__import__(net_file).inner_stride)
        self._dense_coef_ = int(__import__(net_file).dense_coef)
        self._dense_stride_ = int(__import__(net_file).dense_stride)
        self._roi_prob_size_ = (self._roi_ext_size_-self._inner_size_)/self._dense_stride_+1
        # self._roi_stride_  = self._dense_stride_ * self._roi_prob_size_
        self._roi_stride_  = self._inner_size_ - 299 + self._inner_stride_
        ''' Initialize the Queues'''

        self._queue_preA_ = "NULL"
        self._token_preA_ = "NULL"
        self._queue_A2B_list_ = "NULL"
        self._token_A2B_ = "NULL"
        self._queue_B2C_ = "NULL"
        self._token_B2C_ = "NULL"
        self._queue_result_ = "NULL"
        self._token_result_ = "NULL"

    def _delete_queues(self):
        del self._token_preA_
        del self._queue_preA_
        del self._token_A2B_ 
        del self._queue_B2C_ 
        del self._token_B2C_ 
        del self._queue_result_ 
        del self._token_result_ 
        if self._queue_A2B_list_ == list:
            for q in self._queue_A2B_list_:
                del q
        else:
           del self._queue_A2B_list_
        self._queue_preA_ = "NULL"
        self._token_preA_ = "NULL"
        self._queue_A2B_list_ = "NULL"
        self._token_A2B_ = "NULL"
        self._queue_B2C_ = "NULL"
        self._token_B2C_ = "NULL"
        self._queue_result_ = "NULL"
        self._token_result_ = "NULL"

    def _initial_queues(self):
        ''' Initialize the Queues'''
        self._delete_queues()
        self._queue_preA_ = multiprocessing.Queue(-1)
        self._token_preA_ = multiprocessing.Queue(-1)
        self._queue_A2B_list_ = [multiprocessing.Queue(1) for i_lst in xrange(self._queue_A2B_list_LEN_)]
        self._token_A2B_ = multiprocessing.Queue(-1)
        self._queue_B2C_ = multiprocessing.Queue(self._queue_B2C_LEN_)
        self._token_B2C_ = multiprocessing.Queue(-1)
        self._queue_result_ = multiprocessing.Queue(self._queue_result_LEN_)
        self._token_result_ = multiprocessing.Queue(-1)

    def _reset_server(self):
        ''' Reset the server if bugs appear'''
        # print "#   #   # Reseting..."
        self.terminate_pipelineB_from_server()
        tf.keras.backend.clear_session()
        # self.initial_pipelineB_to_server()
        # print "#   #   # Server reset! "
        
    def initial_pipelineB_to_server(self):
        ''' Initialize and deploy the pipelineB to the server.
        '''
        self._initial_queues()

        if self._processorB_ != "NULL":
            self.terminate_pipelineB_from_server()
        assert self._processorB_ == "NULL"
        self._processorB_ = []
        
        B_parameter_list = []
        B_parameter_list.append(self._queue_A2B_list_)
        B_parameter_list.append(self._token_A2B_)
        B_parameter_list.append(self._queue_B2C_)
        B_parameter_list.append(self._token_B2C_)
        B_parameter_list.append(self._token_result_)
        B_parameter_list.append(self._batch_size_)
        B_parameter_list.append(len(self._kClassName_))

        gpu_group_list = []
        i_tmp = i_lst = pipe_ID = 0
        while i_lst < len(self._gpu_list_):
            gpu_group_list.append(self._gpu_list_[i_lst])
            i_tmp += 1
            if i_tmp == self._gpus_per_group_:
                pipe_ID += 1
                self._processorB_.append(PipelineB(self._configure_path_, B_parameter_list, pipe_ID, gpu_group_list))
                gpu_group_list = []
                i_tmp = 0
            i_lst += 1 
        for i in range(len(self._processorB_)):
            self._processorB_[i].start()
            
    def terminate_pipelineB_from_server(self):
        ''' Terminate the pipelineB from the server.
        '''
        self._delete_queues()
        if self._processorB_ == "NULL":
            return
        for i in range(len(self._processorB_)):
            self._processorB_[i].terminate()
        for i in range(len(self._processorB_)):
            self._processorB_[i].join()
        del self._processorB_[:]
        self._processorB_ = "NULL"
        
    def detect(self, ov_path, slide, debug_mode = False, flag = False):
        ''' Detect the targets from the given slide.
        '''
        if self._processorB_ == "NULL":
            print "Please deploy the pipelineB by initial_pipelineB_to_server() at first time."
            return "NULL", "NULL"

        '''1. Compute working&overview level based on spacing'''

        slide_working_level = 0
        slide_overview_level = 2

        slide_level_downsamples = [int(dsp) for dsp in slide.level_downsamples]
        work_DSP = slide_level_downsamples[slide_working_level]
        overview_DSP = slide_level_downsamples[slide_overview_level]
        
        '''2. Extract ROIs from tissue regions.'''
        roi_list, slide_on_overview_OTSU, threshold_OTSU = GetROIList_by_OTSU_detection(slide, \
                                                                              self._roi_ext_size_, self._roi_stride_,
                                                                              slide_working_level, slide_overview_level, \
                                                                                        # start_w= 0,start_h=0,
                                                                              start_w=-self._inner_size_/2 + (299-self._inner_stride_)/2 ,
                                                                              start_h=-self._inner_size_/2 + (299-self._inner_stride_)/2 ,
                                                                              abandon_threshold=9000,
                                                                              show_progress_flag=False)

        '''3. Compute overview image if debug_mode == True.'''
        if debug_mode == True:
            WSI_ov = GetPatch(slide, 0, 0, slide.level_dimensions[slide_overview_level], slide_overview_level)
            for i_roi in xrange(len(roi_list)):
                wcoor_L0 = roi_list[i_roi][0]
                hcoor_L0 = roi_list[i_roi][1]
                w_roi_size_L0 = roi_list[i_roi][2]
                h_roi_size_L0 = roi_list[i_roi][3]
                wcoor_ov = int(wcoor_L0 / overview_DSP)
                hcoor_ov = int(hcoor_L0 / overview_DSP)
                w_roi_size_ov = int(w_roi_size_L0 / overview_DSP)
                h_roi_size_ov = int(h_roi_size_L0 / overview_DSP)
                cv2.rectangle(WSI_ov, (wcoor_ov, hcoor_ov), (wcoor_ov + w_roi_size_ov, hcoor_ov + h_roi_size_ov),
                              (0, 0, 255), 3)
            cv2.imwrite(ov_path, WSI_ov)
        else:
            WSI_ov = "NULL"

        '''4. Prepare the queue_preA with offset ROI.'''
        # level 0-1
        WSI_width_size_L0, WSI_height_size_L0 = slide.level_dimensions[0]
        WSI_width_size_L0 = WSI_width_size_L0/2 + 1
        WSI_height_size_L0 = WSI_height_size_L0/2 + 1
        # self._roi_size_ = self._roi_size_ * 2
        WSI_prob_width_size = ((WSI_width_size_L0) / int(
            self._dense_stride_ * work_DSP ) ) + 1
        WSI_prob_height_size = ((WSI_height_size_L0) / int(
            self._dense_stride_ * work_DSP ) ) + 1
        WSI_prob_shape = (WSI_prob_height_size, WSI_prob_width_size)
        roi_ext_shape = (self._roi_ext_size_, self._roi_ext_size_)
        offset_roi_list = []
        for i_roi in xrange(len(roi_list)):
            wcoor_L0 = roi_list[i_roi][0]
            hcoor_L0 = roi_list[i_roi][1]
            for w_offset in xrange(0, self._dense_coef_):
                for h_offset in xrange(0, self._dense_coef_):
                    offset_roi_list.append([wcoor_L0, hcoor_L0, w_offset, h_offset])
        offset_roi_list_LEN = len(offset_roi_list)
        slide_info_list = []
        data_para_list = []
        i_oR = i_ba = qid = 0
        while i_oR < len(offset_roi_list):
            if i_ba < self._batch_size_ * self._batch_size_multipler_:
                wcoor_L0 = offset_roi_list[i_oR][0]
                hcoor_L0 = offset_roi_list[i_oR][1]
                w_offset = offset_roi_list[i_oR][2]
                h_offset = offset_roi_list[i_oR][3]
                # offset in level 0
                wcoor_off_L0 = wcoor_L0 + w_offset * self._dense_stride_ * work_DSP
                hcoor_off_L0 = hcoor_L0 + h_offset * self._dense_stride_ * work_DSP
                # left up point after offset
                data_para_list.append([wcoor_off_L0, hcoor_off_L0, roi_ext_shape, slide_working_level])
                # left up point before offset
                slide_info_list.append([wcoor_L0, hcoor_L0, w_offset, h_offset])
                i_ba += 1
                i_oR += 1
            if i_ba == self._batch_size_ * self._batch_size_multipler_ or i_oR == len(offset_roi_list):
                self._queue_preA_.put([data_para_list, slide_info_list, qid])
                self._token_preA_.put("Done")
                i_ba = 0
                qid += 1
                if qid >= self._queue_A2B_list_LEN_:
                    qid = 0
                slide_info_list = []
                data_para_list = []

        A_Parameter_list = []
        A_Parameter_list.append(self._queue_preA_)
        A_Parameter_list.append(self._token_preA_)
        A_Parameter_list.append(self._queue_A2B_list_)
        A_Parameter_list.append(self._token_A2B_)
        A_Parameter_list.append(self._token_result_)
        A_Parameter_list.append(slide)
        C_Parameter_list = []
        C_Parameter_list.append(self._queue_preA_)
        C_Parameter_list.append(self._token_preA_)
        C_Parameter_list.append(self._queue_A2B_list_)
        C_Parameter_list.append(self._token_A2B_)
        C_Parameter_list.append(self._queue_B2C_)
        C_Parameter_list.append(self._token_B2C_)
        C_Parameter_list.append(self._queue_result_)
        C_Parameter_list.append(self._token_result_)
        C_Parameter_list.append(self._batch_size_)
        C_Parameter_list.append(slide_level_downsamples)
        C_Parameter_list.append(slide_working_level)
        C_Parameter_list.append(WSI_prob_shape)
        C_Parameter_list.append(self._inner_size_)
        C_Parameter_list.append(self._inner_stride_)
        C_Parameter_list.append(self._roi_size_)
        C_Parameter_list.append(self._roi_stride_)
        C_Parameter_list.append(self._dense_coef_)
        C_Parameter_list.append(self._dense_stride_)
        C_Parameter_list.append(offset_roi_list_LEN)
        C_Parameter_list.append(self._roi_prob_size_)

        processorA = []
        processorC = []

        for i_num in xrange(self._pipelineA_Num_):
            processorA.append(PipelineA(self._configure_path_, A_Parameter_list))
        processorC.append(PipelineC(self._configure_path_, C_Parameter_list))
        processor_list = processorA + processorC
        # Start processors
        for i in xrange(len(processor_list)):
            processor_list[i].start()

        ''' Get the result and kill the processor A and C'''
        token_info = self._token_result_.get(1)
        if token_info == "BlockError":  # Block Error, re-run the detect()
            for i in xrange(len(processor_list)):
                processor_list[i].terminate()
            for i in xrange(len(processor_list)):
                processor_list[i].join()
            self._reset_server()
            print "#   #   # Start re-detect..."
            WSI_prob_map, WSI_ov = self.detect(ov_path, slide, debug_mode)
            return WSI_prob_map, WSI_ov
        try:
            WSI_prob_map = self._queue_result_.get(1, 300)
        except:
            print "!ERROR : CANNOT get WSI_prob_map form queue"
            for i in xrange(len(processor_list)):
                processor_list[i].terminate()
            for i in xrange(len(processor_list)):
                processor_list[i].join()
            self._reset_server()
            print "#   #   # Start re-detect..."
            WSI_prob_map, WSI_ov = self.detect(ov_path, slide, debug_mode)
            return WSI_prob_map, WSI_ov

        for i in xrange(len(processor_list)):
            processor_list[i].terminate()
        for i in xrange(len(processor_list)):
            processor_list[i].join()
        return WSI_prob_map, WSI_ov


    def detect_batchly(self, slide_path_list, output_dir):
        ROIs_path_coor = "./Outputs/ROIs/coordinates/"
        ROIs_path_image = "./Outputs/ROIs/coor_in_image/"
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        cache_dir = "%s/cache" %output_dir
        if os.path.exists(cache_dir) == False:
            os.makedirs(cache_dir)
        npy_dir = "%s/npyDir" %cache_dir
        if os.path.exists(npy_dir) == False:
            os.makedirs(npy_dir)
        token_dir = "%s/tokenDir" %cache_dir
        if os.path.exists(token_dir) == False:
            os.makedirs(token_dir)
        ov_dir = "%s/" %output_dir
        if os.path.exists(ov_dir) == False:
            os.makedirs(ov_dir)

        ''' Get the Undo slides'''
        fst_prior_list = []
        sec_prior_list = []
        for slide_path in slide_path_list:
            slide_file_name = slide_path.split('/')[-1]
            postfix = slide_file_name.split('.')[-1]
            slide_name = slide_file_name.split('.%s'%postfix )[0]
            token_path = "%s/%s.token" %(token_dir, slide_name)
            npy_path   = "%s/%s.npy" %(npy_dir, slide_name)
            ROI_path   = "%s/%s.npy" %(ROIs_path_coor, slide_name)

            if os.path.exists(npy_path) == False:
                fst_prior_list.append(slide_path)
            elif os.path.exists(ROI_path) == False:
                sec_prior_list.append(slide_path)

        print "==> Total number of images: %d." % len(fst_prior_list)
        while len(fst_prior_list)>0 or len(sec_prior_list)>0:

            if len(fst_prior_list)>0:
                i_rand = random.randint(0, len(fst_prior_list)-1)
                slide_path = fst_prior_list[i_rand]
                slide_file_name = slide_path.split('/')[-1]
                postfix = slide_file_name.split('.')[-1]
                slide_name = slide_file_name.split('.%s'%postfix )[0]
                token_path = "%s/%s.token" %(token_dir, slide_name)
                npy_path   = "%s/%s.npy" %(npy_dir, slide_name)
                ov_path   = "%s/%s.png" %(ov_dir, slide_name)
                with open(token_path, 'w') as file_token:
                    pass
            elif len(sec_prior_list)>0:
                i_rand = random.randint(0, len(sec_prior_list)-1)
                slide_path = sec_prior_list[i_rand]
                slide_file_name = slide_path.split('/')[-1]
                postfix = slide_file_name.split('.')[-1]
                slide_name = slide_file_name.split('.%s'%postfix )[0]
                npy_path   = "%s/%s.npy" %(npy_dir, slide_name)
                ov_path = "%s/%s.png" % (ov_dir, slide_name)
            else:
                break
            print "\n==> Processing %s" %(slide_name)
            slide = openslide.open_slide(slide_path)
            WSI_prob_map = None
            if not os.path.exists(npy_path):
                WSI_prob_map, WSI_ROI_ov = self.detect(ov_path, slide, debug_mode=False)
                print "\n==> Generating visual results in ", ov_dir
                np.save(npy_path, WSI_prob_map)
                overlay(slide, WSI_prob_map, ov_dir, slide_name)
            if WSI_prob_map is None:
                WSI_prob_map = np.load(npy_path)
            _get_ROIs(slide, WSI_prob_map, slide_name, 100, ROIs_path_coor, ROIs_path_image)

            del fst_prior_list[:]
            del sec_prior_list[:]
            fst_prior_list = []
            sec_prior_list = []
            for slide_path in slide_path_list:
                slide_file_name = slide_path.split('/')[-1]
                postfix = slide_file_name.split('.')[-1]
                slide_name = slide_file_name.split('.%s'%postfix )[0]
                token_path = "%s/%s.token" %(token_dir, slide_name)
                npy_path   = "%s/%s.npy" %(npy_dir, slide_name)
                if os.path.exists(token_path) == False:
                    fst_prior_list.append(slide_path)
                elif os.path.exists(npy_path) == False:
                    sec_prior_list.append(slide_path)
        
                    
                    
                    
        
        

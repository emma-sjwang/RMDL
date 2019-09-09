
import numpy as np
import openslide
import cv2
from scipy.ndimage import gaussian_filter
import skimage.morphology as sm
import os
# from tqdm import tqdm

def GetPatch(whole_slide_image, start_w, start_h, window_shape, working_level):
    tile = np.array(whole_slide_image.read_region((start_w, start_h), working_level, window_shape)) 
    return tile

def RectifyLevels(working_level, overview_level, slide, mask="NULL"):
    ''' Adjust the working & overview level of slide & mask, which increase the robust of in matching style of slide & mask,
        return slide_working_level, slide_overview_level, mask_working_level, mask_overview_level
        E.g. self._rectify_levels(working_level, overview_level, slide, mask="NULL")
    '''   
    max_level = slide.level_count - 1
    '''1. Adjust the working & overview levels of slide.'''
    if (working_level < 0):
        print "WARNING: working level %d is out of range of the WSI. It is forcely adjusted to 0." \
              %(working_level)
        slide_working_level = 0
    elif (working_level > max_level):
        print "WARNING: working level %d is out of range of the WSI. It is forcely adjusted to %d." \
              %(working_level, max_level)
        slide_working_level = max_level
    else:
        slide_working_level = working_level
            
    if (overview_level < slide_working_level ):
        print "WARNING: overview level %d is out of range of the WSI.It is forcely adjusted to %d" \
              %(overview_level, slide_working_level)
        slide_overview_level = slide_working_level
    elif (overview_level > max_level):
        print "WARNING: overview level %d is out of range of the WSI. It is forcely adjusted to %d." \
              %(overview_level, max_level)
        slide_overview_level = max_level
    else:
        slide_overview_level = overview_level
        
    if mask == "NULL":
        return slide_working_level, slide_overview_level, "NULL", "NULL"
        
    '''2. Adjust the overview level and working levels of mask, if mask is provided.'''      
    slide_level_downsamples = [int(dsp) for dsp in slide.level_downsamples]
    mask_level_downsamples = [int(dsp) for dsp in mask.level_downsamples]
        
    if slide_level_downsamples[slide_working_level] in mask_level_downsamples:
        mask_working_level = mask_level_downsamples.index(slide_level_downsamples[slide_working_level])
        mask_working_level = 0
    else:
        print "Error occur because the working level of slide and mask is unmatched!"
        return -1
    if slide_level_downsamples[slide_overview_level] in mask_level_downsamples:
        mask_overview_level = mask_level_downsamples.index(slide_level_downsamples[slide_overview_level])
    else:
        print "rectify_levels(): Error occur because the working level of slide and mask is unmatched!"
        raise Exception("rectify_levels(): Error occur because the working level of slide and mask is unmatched!")
        return "NULL", "NULL", "NULL", "NULL"
    del slide_level_downsamples[:]
    del slide_level_downsamples[:]
    return slide_working_level, slide_overview_level, mask_working_level, mask_overview_level

def FindLevel(slide, spacing, toleration_error = 0.3):
    spacing_L0 = float(slide.properties.get("openslide.mpp-x"))
    level_downsamples = [int(dsp) for dsp in slide.level_downsamples]
    level_spacing = [float(spacing_L0*dsp) for dsp in level_downsamples]
    print "level_spacing: ", level_spacing
    level = "NULL"
    for i_lv in xrange(len(level_spacing)):
        if spacing<level_spacing[i_lv]*(1.0+toleration_error) and \
           spacing>level_spacing[i_lv]*(1.0-toleration_error):
            level = i_lv
            break
        print "level_spacing[i_lv]: ", level_spacing[i_lv]
    return level


def GetROIList_by_OTSU(slide, ROI_size, ROI_stride, slide_working_level, slide_overview_level, start_w=0, start_h=0,
                       abandon_threshold=1, show_progress_flag=True):
    ''' Extract the tissue regions by OTSU algorithm, and return a ROI list contains:
            w_ov_L0, h_ov_L0, overview_window_size_w_L0, overview_window_size_h_L0
        return ROI_list, slide_on_overview_OTSU, threshold_OTSU
        E.g. get_ROI_list_by_OTSU(slide, ROI_size, ROI_stride, slide_working_level, slide_overview_level ,start_w, start_h, abandon_threshold, show_progress_flag=True)
        The start_w and start_h are the coordinates on the working level
    '''
    # if show_progress_flag == True:
    #     print "    -Begin ROI computing by OTSU..."
    '''1. Prepare the parameters'''
    slide_level_downsamples = [int(dsp) for dsp in slide.level_downsamples]
    slide_on_overview = GetPatch(slide, 0, 0, slide.level_dimensions[slide_overview_level], slide_overview_level)
    slide_on_overview_copy = slide_on_overview.copy()
    slide_size_w_working, slide_size_h_working = slide.level_dimensions[slide_working_level]
    '''1. Rectify the black Background in White.'''
    t1 = np.zeros(slide_on_overview[:, :, 0].shape)
    t2 = np.zeros(slide_on_overview[:, :, 0].shape)
    t3 = np.zeros(slide_on_overview[:, :, 0].shape)
    t1[slide_on_overview[:, :, 0] < 5] = 1
    t2[slide_on_overview[:, :, 1] < 5] = 1
    t3[slide_on_overview[:, :, 2] < 5] = 1
    t4 = t1 + t2 + t3
    slide_on_overview[:, :, 0][t4 > 2.5] = 254
    slide_on_overview[:, :, 1][t4 > 2.5] = 254
    slide_on_overview[:, :, 2][t4 > 2.5] = 254

    '''2. Compute the ROIs by OTSU and store in ROI_list'''
    slide_on_overview_gray = cv2.cvtColor(slide_on_overview, cv2.COLOR_BGR2GRAY)
    threshold_OTSU, slide_on_overview_OTSU = cv2.threshold(slide_on_overview_gray, 0, 255,
                                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    slide_on_overview_OTSU = cv2.medianBlur(slide_on_overview_OTSU, 5)
    slide_on_overview_OTSU[slide_on_overview_OTSU <= threshold_OTSU] = 0
    # slide_on_overview_OTSU [slide_on_overview_OTSU> threshold_OTSU] = 255
    slide_on_overview_OTSU[slide_on_overview_OTSU > threshold_OTSU] = 1

    slide_on_overview_OTSU = sm.erosion(slide_on_overview_OTSU, sm.square(4))
    slide_on_overview_OTSU = sm.dilation(slide_on_overview_OTSU, sm.square(4))

    slide_on_overview_OTSU_copy = np.zeros([slide_on_overview_OTSU.shape[0], slide_on_overview_OTSU.shape[1], 4])
    slide_on_overview_OTSU_copy[:, :, 0] = slide_on_overview_OTSU
    slide_on_overview_OTSU_copy[:, :, 1] = slide_on_overview_OTSU
    slide_on_overview_OTSU_copy[:, :, 2] = slide_on_overview_OTSU
    slide_on_overview_OTSU_copy[:, :, 3] = slide_on_overview_OTSU
    # cv2.imwrite(ov_dir + "_initial.png", slide_on_overview_copy[:, :, 0:3])
    slide_on_overview_copy = slide_on_overview_copy * (1 - slide_on_overview_OTSU_copy)
    cv2.imwrite('test.png', slide_on_overview_copy[:, :, 0:3])
    # print "ov_dir: ", ov_dir
    slide_on_overview_OTSU[slide_on_overview_OTSU > 0.9] = 255


    progress_bar_sum = 0
    progress_bar_tmp = 0
    progress_bar_total = (float(slide_size_w_working) / ROI_stride + 1) \
                         * (float(slide_size_h_working) / ROI_stride + 1)
    progress_bar_interval = progress_bar_total / 20

    ROI_list = []
    stop_flag_1 = False
    w_wk = start_w
    while (w_wk < slide_size_w_working and stop_flag_1 == False):
        stop_flag_2 = False
        h_wk = start_h
        while (h_wk < slide_size_h_working and stop_flag_2 == False):
            if (w_wk + ROI_size / 2 + 1 >= slide_size_w_working):
                stop_flag_1 = True
            if (h_wk + ROI_size / 2 + 1 >= slide_size_h_working):
                stop_flag_2 = True

                # tracing the procedure...
            progress_bar_sum = progress_bar_sum + 1
            progress_bar_tmp = progress_bar_tmp + 1
            if show_progress_flag == True and progress_bar_tmp > progress_bar_interval:
                progress_bar_tmp = 0
                print "       Overview complete: %.2f %%" % (progress_bar_sum * 100.0 / progress_bar_total)

            h_ov = h_wk * slide_level_downsamples[slide_working_level] / slide_level_downsamples[slide_overview_level]
            w_ov = w_wk * slide_level_downsamples[slide_working_level] / slide_level_downsamples[slide_overview_level]
            ROI_size_ov = ROI_size * slide_level_downsamples[slide_working_level] / slide_level_downsamples[
                slide_overview_level]

            h_ov_end = slide_on_overview_OTSU.shape[0] \
                if h_ov + ROI_size_ov > slide_on_overview_OTSU.shape[0] else h_ov + ROI_size_ov
            w_ov_end = slide_on_overview_OTSU.shape[1] \
                if w_ov + ROI_size_ov > slide_on_overview_OTSU.shape[1] else w_ov + ROI_size_ov

            h_ov = 0 if h_ov < 0 else h_ov
            w_ov = 0 if w_ov < 0 else w_ov

            OTSU_tile = slide_on_overview_OTSU[h_ov:h_ov_end, w_ov:w_ov_end]
            # print OTSU_tile.shape[0]
            assert OTSU_tile.shape[0] != 0

            # wanted ROI   too much backgraound will beb discarded
            OTSU_Num = np.zeros(OTSU_tile.shape)
            OTSU_Num[OTSU_tile < threshold_OTSU] = 1

            if (OTSU_tile.min() < threshold_OTSU and OTSU_Num.sum() > OTSU_tile.shape[0]*OTSU_tile.shape[1]/10):
                w_L0 = w_ov * slide_level_downsamples[slide_overview_level]
                h_L0 = h_ov * slide_level_downsamples[slide_overview_level]
                ROI_size_L0 = ROI_size * slide_level_downsamples[slide_working_level]
                ROI_list.append([w_L0, h_L0, ROI_size_L0, ROI_size_L0])
            h_wk = h_wk + ROI_stride
        w_wk = w_wk + ROI_stride
    # if show_progress_flag == True:
        # print "    -Finish ROI computing by OTSU..."
    return ROI_list, slide_on_overview_OTSU, threshold_OTSU

def GetROIList_by_OTSU_detection(slide, ROI_size, ROI_stride, slide_working_level, slide_overview_level, start_w=0, start_h=0,
                       abandon_threshold=1, show_progress_flag=True):
    ''' Extract the tissue regions by OTSU algorithm, and return a ROI list contains:
            w_ov_L0, h_ov_L0, overview_window_size_w_L0, overview_window_size_h_L0
        return ROI_list, slide_on_overview_OTSU, threshold_OTSU
        E.g. get_ROI_list_by_OTSU(slide, ROI_size, ROI_stride, slide_working_level, slide_overview_level ,start_w, start_h, abandon_threshold, show_progress_flag=True)
        The start_w and start_h are the coordinates on the working level
    '''
    ROI_size = 2 * ROI_size
    ROI_stride =  2 * ROI_stride
    if show_progress_flag == True:
        print "    -Begin ROI computing by OTSU..."
    '''1. Prepare the parameters'''
    slide_level_downsamples = [int(dsp) for dsp in slide.level_downsamples]
    slide_on_overview = GetPatch(slide, 0, 0, slide.level_dimensions[slide_overview_level], slide_overview_level)
    slide_on_overview_copy = slide_on_overview.copy()
    slide_size_w_working, slide_size_h_working = slide.level_dimensions[slide_working_level]
    slide_size_w_working = slide_size_w_working
    slide_size_h_working = slide_size_h_working
    '''1. Rectify the black Background in White.'''
    t1 = np.zeros(slide_on_overview[:, :, 0].shape)
    t2 = np.zeros(slide_on_overview[:, :, 0].shape)
    t3 = np.zeros(slide_on_overview[:, :, 0].shape)
    t1[slide_on_overview[:, :, 0] < 5] = 1
    t2[slide_on_overview[:, :, 1] < 5] = 1
    t3[slide_on_overview[:, :, 2] < 5] = 1
    t4 = t1 + t2 + t3
    slide_on_overview[:, :, 0][t4 > 2.5] = 254
    slide_on_overview[:, :, 1][t4 > 2.5] = 254
    slide_on_overview[:, :, 2][t4 > 2.5] = 254

    '''2. Compute the ROIs by OTSU and store in ROI_list'''
    slide_on_overview_gray = cv2.cvtColor(slide_on_overview, cv2.COLOR_BGR2GRAY)
    threshold_OTSU, slide_on_overview_OTSU = cv2.threshold(slide_on_overview_gray, 0, 255,
                                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    slide_on_overview_OTSU = cv2.medianBlur(slide_on_overview_OTSU, 5)
    slide_on_overview_OTSU[slide_on_overview_OTSU <= threshold_OTSU] = 0
    # slide_on_overview_OTSU [slide_on_overview_OTSU> threshold_OTSU] = 255
    slide_on_overview_OTSU[slide_on_overview_OTSU > threshold_OTSU] = 1

    slide_on_overview_OTSU = sm.erosion(slide_on_overview_OTSU, sm.square(1))
    slide_on_overview_OTSU = sm.dilation(slide_on_overview_OTSU, sm.square(1))

    slide_on_overview_OTSU_copy = np.zeros([slide_on_overview_OTSU.shape[0], slide_on_overview_OTSU.shape[1], 4])
    slide_on_overview_OTSU_copy[:, :, 0] = slide_on_overview_OTSU
    slide_on_overview_OTSU_copy[:, :, 1] = slide_on_overview_OTSU
    slide_on_overview_OTSU_copy[:, :, 2] = slide_on_overview_OTSU
    slide_on_overview_OTSU_copy[:, :, 3] = slide_on_overview_OTSU
    # cv2.imwrite(ov_dir + "_initial.png", slide_on_overview_copy[:, :, 0:3])
    slide_on_overview_copy = slide_on_overview_copy * (1 - slide_on_overview_OTSU_copy)
    # cv2.imwrite('test.png', slide_on_overview_copy[:, :, 0:3])
    # print "ov_dir: ", ov_dir
    slide_on_overview_OTSU[slide_on_overview_OTSU > 0.9] = 255


    progress_bar_sum = 0
    progress_bar_tmp = 0
    progress_bar_total = (float(slide_size_w_working) / ROI_stride + 1) \
                         * (float(slide_size_h_working) / ROI_stride + 1)
    progress_bar_interval = progress_bar_total / 20

    ROI_list = []
    stop_flag_1 = False
    w_wk = start_w
    # print("start_h:", start_h)
    # print("start_w:", start_w)
    # pbar = tqdm(total = progress_bar_total)
    while (w_wk < slide_size_w_working and stop_flag_1 == False):
        stop_flag_2 = False
        h_wk = 0
        # print("h_wk:", h_wk)
        while (h_wk < slide_size_h_working and stop_flag_2 == False):
            if (w_wk + ROI_size  + 1 >= slide_size_w_working):
                stop_flag_1 = True

            if (h_wk + ROI_size  + 1 >= slide_size_h_working):
                stop_flag_2 = True
            progress_bar_sum = progress_bar_sum + 1
            progress_bar_tmp = progress_bar_tmp + 1
            # if show_progress_flag == True and progress_bar_tmp > progress_bar_interval:
            #     progress_bar_tmp = 0
            #     print "       Overview complete: %.2f %%" % (progress_bar_sum * 100.0 / progress_bar_total)

            h_ov = h_wk * slide_level_downsamples[slide_working_level] / slide_level_downsamples[slide_overview_level]
            w_ov = w_wk * slide_level_downsamples[slide_working_level] / slide_level_downsamples[slide_overview_level]
            ROI_size_ov = ROI_size * slide_level_downsamples[slide_working_level] / slide_level_downsamples[
                slide_overview_level]

            h_ov_end = slide_on_overview_OTSU.shape[0] \
                if h_ov + ROI_size_ov > slide_on_overview_OTSU.shape[0] else h_ov + ROI_size_ov
            w_ov_end = slide_on_overview_OTSU.shape[1] \
                if w_ov + ROI_size_ov > slide_on_overview_OTSU.shape[1] else w_ov + ROI_size_ov

            h_ov = 0 if h_ov < 0 else h_ov
            w_ov = 0 if w_ov < 0 else w_ov


            OTSU_tile = slide_on_overview_OTSU[h_ov:h_ov_end, w_ov:w_ov_end]
            assert OTSU_tile.shape[0] != 0

            # wanted ROI   too much backgraound will beb discarded
            OTSU_Num = np.zeros(OTSU_tile.shape)
            OTSU_Num[OTSU_tile < threshold_OTSU] = 1

            if (OTSU_tile.min() < threshold_OTSU and OTSU_Num.sum() > OTSU_tile.shape[0]*OTSU_tile.shape[1]/100):
                w_L0 = w_ov / slide_level_downsamples[slide_working_level] * slide_level_downsamples[slide_overview_level] * slide_level_downsamples[slide_working_level]
                h_L0 = h_ov / slide_level_downsamples[slide_working_level] * slide_level_downsamples[slide_overview_level] * slide_level_downsamples[slide_working_level]

                ROI_size_L0 = ROI_size * slide_level_downsamples[slide_working_level]
                ROI_list.append([w_L0, h_L0, ROI_size_L0, ROI_size_L0])
            h_wk = h_wk + ROI_stride
            # pbar.update(1)
        w_wk = w_wk + ROI_stride
    # if show_progress_flag == True:
    #     print "    -Finish ROI computing by OTSU..."
    return ROI_list, slide_on_overview_OTSU, threshold_OTSU

def ToMultiGPU(model, gpuList = [0]):
    ''' Model conversion to multi-gpu
    '''
    import tensorflow as tf
    from keras.layers import Input, concatenate
    from keras.layers.core import Lambda
    from keras.models import Model
    from keras import backend as K
    
    def _slice_batch(x, n_gpus, part):
        sh = K.shape(x)
        L = sh[0] / n_gpus
        if part == n_gpus - 1:
            return x[part*L:]
        return x[part*L:(part+1)*L]
    
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])
    towers = []
    for g in xrange(len(gpuList)):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(_slice_batch, lambda shape: shape, arguments={'n_gpus':len(gpuList), 'part':g})(x)
            towers.append(model(slice_g))
    with tf.device('/cpu:0'):
        if len(gpuList)==1:
            merged = towers[0]
        else:
            merged = concatenate(towers, axis=0)
    return Model(input=[x], output=merged)

def CompressProbMap(prob_tile_multi_channel, class_frequency_band):
    ''' Use to Compress the Multi-Channel into Single-Channel to consume smaller storage
    '''
    prob_tile_single_channel = np.zeros(prob_tile_multi_channel[:,:,0].shape, dtype = 'float32')
    prob_tile_max = np.zeros(prob_tile_single_channel.shape, dtype = 'float32')
    for i_cla in xrange(0, len(class_frequency_band)):
        prob_tile = prob_tile_multi_channel[:,:,i_cla]
        bandMin = class_frequency_band[i_cla][0]
        bandMax = class_frequency_band[i_cla][1]
        prob_tile_single_channel  = np.where(prob_tile>prob_tile_max, prob_tile*(bandMax-bandMin)+bandMin, prob_tile_single_channel)
        prob_tile_max = np.where(prob_tile>prob_tile_max, prob_tile, prob_tile_max)             
    return prob_tile_single_channel

def RestoreProbMap(prob_tile_single_channel, class_frequency_band):
    ''' Use to Restore the Single-Channel into Multi-Channel for Storage Saving
    '''
    prob_tile_multi_channel = np.zeros((prob_tile_single_channel.shape[0],prob_tile_single_channel.shape[1],len(class_frequency_band)), dtype = 'float32')
    for i_cla in xrange(len(class_frequency_band)):
        prob_tile_single_channel_copy = np.array(prob_tile_single_channel)
        bandMin = class_frequency_band[i_cla][0]
        bandMax = class_frequency_band[i_cla][1]
        prob_tile_single_channel_copy[prob_tile_single_channel_copy<bandMin] = 0
        prob_tile_single_channel_copy[prob_tile_single_channel_copy>bandMax] = 0
        prob_tile_single_channel_copy = (prob_tile_single_channel_copy-bandMin)/(bandMax-bandMin) if bandMax-bandMin != 0 else prob_tile_single_channel_copy
        prob_tile_single_channel_copy[prob_tile_single_channel_copy<0.0] = 0.0
        prob_tile_multi_channel[:,:,i_cla] = prob_tile_single_channel_copy
    return prob_tile_multi_channel


def _transfer_index_to_2_dims(max_index_in_Prob_map, shape):
    len = max_index_in_Prob_map.shape[0]
    x = []
    y = []
    for i in range(len):
        index = max_index_in_Prob_map[i]
        _x = np.floor(index/shape[1])
        _y = index - _x * shape[1]
        x.append(_x)
        y.append(_y)
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)
    return x,y


def _get_ROIs(slide, Probs_map, DataName, ROI_NUM, ROIs_path_coor, ROIs_path_image):
    filename = DataName + ".svs"
    if not os.path.exists(ROIs_path_coor):
        os.makedirs(ROIs_path_coor)
    if not os.path.exists(ROIs_path_image):
        os.makedirs(ROIs_path_image)
    ROIs_save_coor_path = ROIs_path_coor + filename[:-4] + ".txt"
    if os.path.exists(ROIs_save_coor_path):
        return
    print "==> generating ROI list "
    level_downsamples = [1, 4, 16, 64]
    workingLevel = 2

    show_Result = np.array(slide.read_region((0, 0), workingLevel, slide.level_dimensions[workingLevel]))

    r, g, b, a = cv2.split(show_Result)
    show_Result = cv2.merge([b, g, r])

    if Probs_map is None:
        print "!ERROR: Probs map is None!"
        exit(-1)
    shape = Probs_map.shape

    if not os.path.dirname(ROIs_save_coor_path):
        os.makedirs(os.path.dirname(ROIs_save_coor_path))
    ROIs_save_file = open(ROIs_save_coor_path, 'wb')
    num_group = np.zeros(3)
    for slice_index in range(shape[2]-1, -1, -1):
        ROI_list = []
        Prob_map_per_slice = Probs_map[:, :, slice_index]
        Prob_map_per_slice = gaussian_filter(Prob_map_per_slice, 1)

        Prob_map_in_flatten = Prob_map_per_slice.flatten(order='C')
        max_index_in_Prob_map = Prob_map_in_flatten.argsort()[-8000:]
        x, y = _transfer_index_to_2_dims(max_index_in_Prob_map, Prob_map_per_slice.shape[0:2])

        for x_index in range(x.shape[0]-1, -1, -1):

            flag = False
            xx = x[x_index]
            yy = y[x_index]
            for xi, yi in ROI_list:
                if abs(xi - xx) < 4 and abs(yi -yy) < 4:
                    flag = True
                    break
            if flag:
                continue

            if len(ROI_list)>=ROI_NUM:
                break

            w = yy * 32 * 2
            h = xx * 32 * 2
            ROIs_save_file.write('%d\t%d\t%d\t%d\t%d\t%f\n' % (w, h, 299*2, 299*2, slice_index, Probs_map[int(x[x_index]), int(y[x_index]), slice_index]))
            Probs_map[x[x_index] , y[x_index] , slice_index] = 255
            if ((h / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel]) >= show_Result.shape[0]):
                continue
            if ((w / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel]) >= show_Result.shape[1]):
                continue

            if (show_Result[h / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel],
                                        w / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel], 0] > 230 ) and(
                show_Result[h / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel],
                            w / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel], 1] > 230) and(
                show_Result[h / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel],
                            w / level_downsamples[workingLevel] - 150 / level_downsamples[workingLevel], 2] > 230):
                continue

            ROI_list.append([xx, yy])
            if slice_index == 0:
                num_group[0] += 1
                cv2.rectangle(show_Result, (w / level_downsamples[workingLevel] - 299 / level_downsamples[workingLevel],
                                        h / level_downsamples[workingLevel] - 299 / level_downsamples[workingLevel]), (
                          w / level_downsamples[workingLevel] + 299 / level_downsamples[workingLevel],
                          h / level_downsamples[workingLevel] + 299 / level_downsamples[workingLevel]), (255, 0, 0), 2)

            elif slice_index == 1:
                num_group[1] += 1
                cv2.rectangle(show_Result, (w / level_downsamples[workingLevel] - 299 / level_downsamples[workingLevel],
                                            h / level_downsamples[workingLevel] - 299 / level_downsamples[workingLevel]), (
                                  w / level_downsamples[workingLevel] + 299 / level_downsamples[workingLevel],
                                  h / level_downsamples[workingLevel] + 299 / level_downsamples[workingLevel]),
                              (0, 255, 0), 2)
            elif slice_index == 2:
                num_group[2] += 1
                cv2.rectangle(show_Result, (w / level_downsamples[workingLevel] - 299 / level_downsamples[workingLevel],
                                            h / level_downsamples[workingLevel] - 299 / level_downsamples[workingLevel]), (
                                  w / level_downsamples[workingLevel] + 299 / level_downsamples[workingLevel],
                                  h / level_downsamples[workingLevel] + 299 / level_downsamples[workingLevel]),
                              (0, 0, 255), 2)
    if not os.path.exists(ROIs_path_image + "/overview/"):
        os.makedirs(ROIs_path_image + "/overview/")

    cv2.imwrite(ROIs_path_image + "/overview/" + filename[:-4] + "_overview.jpg", show_Result)
    ROIs_save_file.close()



import numpy as np
import matplotlib.pyplot as plt

def show_anomalies(patch_array):
        num_figs = len(patch_array)
        fig = plt.figure(figsize=(num_figs * 30,30))
        plt.tight_layout()
        for i in range(len(patch_array)):
            plt.subplot(num_figs,1, i+1)
            plt.imshow(patch_array[i])
            plt.axis('off')
def make_3_channel(image):
    return np.array([[[s,s,s] for s in r] for r in image],dtype="u1")
def add_color_red_2d(image):
    return np.array([[[0.7,s,s] for s in r] for r in image],dtype="u1")
def add_color_green_2d(image):
    return np.array([[[0.4,s,0.9] for s in r] for r in image],dtype="u1")
def add_color_yellow_2d(image):
    return np.array([[[s,0.3,0.3] for s in r] for r in image],dtype="u1")
def paint_image_anomalies(image_list,true_labels,pred_labels):
    imgs = []
    h_turns = 21
    w_turns = 32
    for img in image_list:
        image = make_3_channel(img)
        top = 0
        left = 0
        h, w = image.shape[:2]
        for adv_h in range(h_turns):
            for adv_w in range(w_turns):
                tag = img_tag[adv_h * 32 :(adv_h +1) * 32 , adv_w * 32 : (adv_w + 1 ) * 32]
                anomaly = np.sum(tag)
                if anomaly:
                    mask = np.array(tag==255)
                    image[adv_h * 32 :(adv_h +1) * 32 , adv_w * 32 : (adv_w + 1 ) * 32,0][mask] = 255
        imgs.append(image)
    return imgs

def connect_imgs(imgs):
    patch = np.squeeze(imgs[0])
    for i in range(1,len(imgs)):
        patch = np.vstack((patch,np.squeeze(imgs[i])))
    return patch

def paint_anomalies(num,patches,scores_pred,tl_bool,statistics=False,show=False):
    patch_image = np.zeros(2064384,dtype=int)
    patch_image = patch_image.reshape(672,1024,3)
    #plt.imshow(patch_image)
    tests = patches[672 * num:672 * (num+1)]
    preds = scores_pred[672 * num:672 * (num+1)]
    tl_bool = tl_bool.astype(bool)
    real = tl_bool[672 * num:672 * (num+1)]
    height = 21
    width = 32
    trues = 0
    fps = 0
    fns = 0
    for i in range(height):
        for j in range(width):
            index = j + (width * i)
            if preds[index] and real[index]:
                # make it green, correct_guess
                add = add_color_green_2d(tests[index] * 255)
                trues+= 1
            elif preds[index]: #false positive
                add = add_color_red_2d(tests[index] * 255)
                fps+=1
            elif real[index]: # False Negative
                add = add_color_yellow_2d(tests[index] * 255)
                fns +=1
            else:
                add = make_3_channel(tests[index] * 255)
            patch_image[i * 32 : (i+1) * 32 , j * 32 : (j+1) * 32] += add 
    if statistics:
        print("true predictions: {}".format(trues))
        print("False Positives: {}".format(fps))
        print("False Negatives: {}".format(fns))
    if show:
        plt.figure(figsize=(15,15))
        plt.imshow(patch_image)
        return
    return patch_image

def compute_predictions(scores,percentile):
    per = np.percentile(scores,percentile)
    predictions = scores >= per
    return predictions
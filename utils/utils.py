import torch
import torch.nn.functional as F
import numpy as np
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt


def collect_batch(batch):
    batch_image = []
    batch_label = []
    for image, label in batch:
        batch_image.append(image.type(torch.FloatTensor))
        batch_label.append(label.type(torch.FloatTensor))
    return torch.stack(batch_image), torch.stack(batch_label)


def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss.sum(), dice.sum()


def loss_func(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    pred = torch.sigmoid(pred)
    dlv, _ = dice_loss(pred, target)
    loss = bce + dlv
    return loss


def metric_batch(pred, target):
    pred = torch.sigmoid(pred)
    _, metric_b = dice_loss(pred, target)
    return metric_b


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    pred = torch.sigmoid(output)
    _, metric_b = dice_loss(pred, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss, metric_b


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_epoch(epoch, model, loss_func, dl, sanity_check=False, opt=None, roc=None):
    runing_loss = 0.0
    runing_metric = 0.0
    len_data = len(dl.dataset)
    with tqdm(desc=f'Epoch {epoch}', postfix=dict, mininterval=0.3) as pbar:
        for xb, yb in dl:
            if torch.cuda.is_available():
                xb, yb = xb.cuda(), yb.cuda()
            output = model(xb)

            # if roc is None:
            #     roc.update(output, yb)
            #     ture_positive_rate, false_positive_rate, recall, precision= roc.get()
            #     with open("log.txt",'a') as f:
            #         for i in range(len(recall)):
            #             if i== 0:
            #                 f.write('Epoch %d: '%epoch)
            #             f.write(
            #                 'TP %.3f  FP %.3f  Recall %.3f Precision %.3f \n'%(ture_positive_rate[i], false_positive_rate[i], recall[i], precision[i])
            #             )

            loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

            runing_loss += loss_b

            if metric_b is not None:
                runing_metric += metric_b
            if sanity_check:
                break

            pbar.set_postfix(
                **{'loss': round(float(runing_loss.cpu().detach().numpy()) / float(len_data), 3),
                   'metric': round(float(runing_metric.cpu().detach().numpy()) / float(len_data), 3),
                   }
            )
            pbar.update(1)

        loss = runing_loss / float(len_data)
        metric = runing_metric / float(len_data)

        if opt is not None:
            pbar.set_postfix(
                **{'train loss': round(float(loss.cpu().detach().numpy()), 3),
                   'train metric': round(float(metric.cpu().detach().numpy()), 3),
                   }
            )
        else:
            pbar.set_postfix(
                **{'val loss': round(float(loss.cpu().detach().numpy()), 3),
                   'val metric': round(float(metric.cpu().detach().numpy()), 3),
                   }
            )
        pbar.update(1)

    return loss, metric


class ROCMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):  # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(
                preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (torch.sigmoid(output) > score_thresh).float()

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())
    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos


class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        preds = preds* 255
        labels = labels * 255
        for iBin in range(self.bins + 1):
            score_thresh = iBin * (255 / self.bins)
            predits = np.array(preds > score_thresh).astype('int64')
            predits = np.reshape(predits, (256, 256))
            labelss = np.array(labels).astype('int64')  # P
            labelss = np.reshape(labelss, (256, 256))

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin] += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [
                x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)

    def get(self, img_num):

        Final_FA = self.FA / ((256 * 256) * img_num)
        Final_PD = self.PD / self.target

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])

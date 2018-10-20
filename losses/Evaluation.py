import numpy as np

# rle encode/edcode
def batch_encode(batch):
    rle = []
    for i in range(len(batch)):
        rle.append(do_length_encode(batch[i]))
    return rle

def do_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    # if len(rle)!=0 and rle[-1]+rle[-2] == x.size:
    #    rle[-2] = rle[-2] -1

    rle = ' '.join([str(r) for r in rle])
    return rle


def batch_decode(batch, H, W, fill_value=255):
    im = []
    for i in range(len(batch)):
        im.append(do_length_decode(batch[i], H, W, fill_value))
    return im

def do_length_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H, W), np.uint8)
    if rle == '' or rle == 'nan': return mask

    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0] - 1
        end = start + r[1]
        mask[start: end] = fill_value
    mask = mask.reshape(W, H).T  # H, W need to swap as transposing.
    return mask


def unpad_im(im, pad=((14, 13), (14, 13))):
    im = im[:, :, pad[0][0]:-pad[0][1], pad[1][0]:-pad[1][1]]
    return im


def do_kaggle_metric(predict, truth, threshold=0.5, smooth=1e-12):
    predict = unpad_im(predict)
    truth = unpad_im(truth)

    N = len(predict)
    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)

    predict = predict > threshold
    truth = truth > 0.5
    intersection = truth & predict
    union = truth | predict
    iou = intersection.sum(1) / (union.sum(1) + smooth)

    # -------------------------------------------
    result = []
    precision = []
    is_empty_truth = (truth.sum(1) == 0)
    is_empty_predict = (predict.sum(1) == 0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        # p = iou >= t

        tp = (~is_empty_truth) & (~is_empty_predict) & (iou > t)
        fp = (~is_empty_truth) & (~is_empty_predict) & (iou <= t)
        fn = (~is_empty_truth) & (is_empty_predict)
        fp_empty = (is_empty_truth) & (~is_empty_predict)
        tn_empty = (is_empty_truth) & (is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append(np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
        precision.append(p)

    result = np.array(result).transpose(1, 2, 0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold


def dice_accuracy(prob, truth, threshold=0.5, is_average=True, smooth=1e-12):
    prob = unpad_im(prob)
    truth = unpad_im(truth)

    batch_size = prob.size(0)
    p = prob.detach().contiguous().view(batch_size, -1)
    t = truth.detach().contiguous().view(batch_size, -1)

    p = p > threshold
    t = t > 0.5
    intersection = p & t
    union = p | t
    dice = (intersection.float().sum(1) + smooth) / (union.float().sum(1) + smooth)

    if is_average:
        dice = dice.sum() / batch_size

    return dice


def accuracy(prob, truth, threshold=0.5, is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size, -1)
    t = truth.detach().view(batch_size, -1)

    p = p > threshold
    t = t > 0.5
    correct = (p == t).float()
    accuracy = correct.sum(1) / p.size(1)

    if is_average:
        accuracy = accuracy.sum() / batch_size

    return accuracy


def do_mAP(pred, truth, is_average=False, threshold=0.5):
    pred = pred > threshold
    batch_size = truth.shape[0]
    metric = []
    for batch in range(batch_size):
        p, t = pred[batch] > 0, truth[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    if is_average:
        return np.mean(metric)
    else:
        return metric

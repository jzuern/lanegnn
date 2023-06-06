from random import randint
import numpy as np
import cv2


class ColorCode():

    def get_coding_1(self):
        coding = {}
        coding[-1] = [100,100,100]
        coding[0] = [0, 0, 0]
        coding[1] = [200, 100, 50]
        coding[2] = [0, 0, 255]
        coding[3] = [255, 220, 200]
        coding[4] = [0, 255, 0]
        coding[5] = [0, 255, 255]
        coding[6] = [255, 255, 0]
        coding[7] = [20, 180, 150]
        coding[8] = [200, 50, 255]
        coding[9] = [80, 10, 100]
        coding[10] = [20, 150, 220]
        coding[11] = [230, 120, 10]
        coding[12] = [255, 0, 0]

        return coding

    def color_code_labels(self, net_out, argmax=True):
        if argmax:
            labels, indices = net_out.max(1)
            labels_cv = indices.cpu().numpy().squeeze()
        else:
            labels_cv = net_out.cpu().numpy().squeeze()

        h = labels_cv.shape[0]
        w = labels_cv.shape[1]

        color_coded = np.zeros((h, w, 3), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                color_coded[y, x, :] = self.color_coding[labels_cv[y, x]]

        return color_coded / 255.

    def __init__(self, max_classes):
        super(ColorCode, self).__init__()
        self.color_coding = self.get_coding_1()


def visImage3Chan(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)
    return cv


def visImage1Chan(data, name):
    cv = data.cpu().data.numpy().squeeze()
    cv2.normalize(cv, cv, 0, 255, cv2.NORM_MINMAX)
    cv = cv.astype(np.uint8)
    cv2.imshow(name, cv)
    return cv

def visDepth(data, name):
    disp_cv = data.cpu().data.numpy().squeeze()
    cv2.normalize(disp_cv, disp_cv, 0, 255, cv2.NORM_MINMAX)
    disp_cv_color = cv2.applyColorMap(disp_cv.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imshow(name, disp_cv_color)
    return disp_cv_color

def drawCriticsLabels(image, critic_vals, size_dots=10):
    num_critics = len(critic_vals)
    total_radius = num_critics * size_dots
    disp_cv_color = cv2.circle(image,
                               (image.shape[1] - (total_radius + 1), image.shape[0] - (total_radius + 1)),
                               total_radius + 1, (255, 255, 255), -1)
    size_circle = total_radius / num_critics
    for i, c in enumerate(critic_vals):
        disp_cv_color = cv2.circle(disp_cv_color, (
        (image.shape[1] - (total_radius + 1)), int(image.shape[0] - (size_circle + int(i * 2 * size_circle)))),
                                   int(size_circle),
                                   (0, 255, 0) if c else (0, 0, 255), -1)

    return image


def visSegDisc(data, name, disc_class, vis=True):
    disp_cv = data.cpu().data.numpy().squeeze()
    cv2.normalize(disp_cv, disp_cv, 0, 255, cv2.NORM_MINMAX)
    disp_cv_color = cv2.applyColorMap(disp_cv.astype(np.uint8), cv2.COLORMAP_JET)

    drawCriticsLabels(disp_cv_color, disc_class)

    if vis:
        cv2.imshow(name, disp_cv_color)
    return disp_cv_color
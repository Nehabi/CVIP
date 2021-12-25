###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates.
#                                            True if the intrinsic parameters are invariable.
# It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners


def calibrate(imgname):
    img = imread(imgname)
    gray = cvtColor(img, COLOR_BGR2GRAY)
    ret, imgCo = findChessboardCorners(gray, (4, 9), None)

    # if pattern detected
    if ret is True:
        criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
        imgPlane = cornerSubPix(gray, imgCo, (11, 11), (-1, -1), criteria)
        imgPlane = imgPlane.reshape(-1, 2)

        # world plane coordinates right to left, top to bottom
        worldPlane = \
            [[40, 0, 40], [40, 0, 30], [40, 0, 20], [40, 0, 10],
             [30, 0, 40], [30, 0, 30], [30, 0, 20], [30, 0, 10],
             [20, 0, 40], [20, 0, 30], [20, 0, 20], [20, 0, 10],
             [10, 0, 40], [10, 0, 30], [10, 0, 20], [10, 0, 10],
             [0, 0, 40], [0, 0, 30], [0, 0, 20], [0, 0, 10],
             [0, 10, 40], [0, 10, 30], [0, 10, 20], [0, 10, 10],
             [0, 20, 40], [0, 20, 30], [0, 20, 20], [0, 20, 10],
             [0, 30, 40], [0, 30, 30], [0, 30, 20], [0, 30, 10],
             [0, 40, 40], [0, 40, 30], [0, 40, 20], [0, 40, 10]]

        # Forming the equation as per Preliminary 2
        eqForm = []
        for i in range(len(imgPlane)):
            val1 = -1 * worldPlane[i][0] * imgPlane[i][0]
            val2 = -1 * worldPlane[i][1] * imgPlane[i][0]
            val3 = -1 * worldPlane[i][2] * imgPlane[i][0]
            val4 = -1 * imgPlane[i][0]
            val5 = -1 * worldPlane[i][0] * imgPlane[i][1]
            val6 = -1 * worldPlane[i][1] * imgPlane[i][1]
            val7 = -1 * worldPlane[i][2] * imgPlane[i][1]
            val8 = -1 * imgPlane[i][1]
            arr = [worldPlane[i][0], worldPlane[i][1], worldPlane[i][2], 1, 0, 0, 0, 0, val1, val2, val3, val4]
            arr1 = [0, 0, 0, 0, worldPlane[i][0], worldPlane[i][1], worldPlane[i][2], 1, val5, val6, val7, val8]
            eqForm.append(arr)
            eqForm.append(arr1)
        # SVD operation to get the VT
        U, S, VT = np.linalg.svd(eqForm)

        m31 = VT[11][8]
        m32 = VT[11][9]
        m33 = VT[11][10]
        # lambda calculation
        lamda = 1 / np.sqrt(m31 * m31 + m32 * m32 + m33 * m33)

        m11 = VT[11][0]
        m12 = VT[11][1]
        m13 = VT[11][2]

        m21 = VT[11][4]
        m22 = VT[11][5]
        m23 = VT[11][6]

        m1 = np.array([m11, m12, m13])
        m1 = np.multiply(m1, lamda)
        m2 = np.array([m21, m22, m23])
        m2 = np.multiply(m2, lamda)
        m3 = np.array([m31, m32, m33])
        m3 = np.multiply(m3, lamda)

        # ox, oy calculation
        ox = np.dot(m1.T, m3)
        oy = np.dot(m2.T, m3)

        # fx calculation
        a = np.dot(m1.T, m1)
        b = ox * ox
        fx = np.sqrt(a - b)

        # fy calculation
        a = np.dot(m2.T, m2)
        b = oy * oy
        fy = np.sqrt(a - b)

        # intrinsic_params
        intrinsic_params = ([fx, fy, ox, oy])

    else:
        # else corner reading failed
        print("Corner reading failed")
        intrinsic_params = ([0, 0, 0, 0])

    is_constant = True
    return intrinsic_params, is_constant


if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)

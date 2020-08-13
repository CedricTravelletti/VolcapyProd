""" Defines the indices of the different paths travelled on the volcano during
the data collection. Also defines the coast.

"""
import numpy as np


# Indices of the data points that are along the coast.
coast_data_inds = np.array(list(range(0, 47)) + list(range(73, 88))
        + list(range(189, 193)) + list(range(284, 316))
        + list(range(336, 340)) + list(range(349, 367))
        + list(range(507, 510)) + list(range(536, 541)))

path1_inds = np.array(list(range(47, 62)))
path2_inds = np.array(list(range(88, 110)))
path3_inds = np.array(list(range(144, 156)))

path4_inds = np.array(list(range(157, 169)))
path5_inds = np.array(list(range(170, 189)))
path6_inds = np.array(list(range(194, 218)))

path7_inds = np.array(list(range(219, 257)))
path8_inds = np.array(list(range(258, 283)))
path9_inds = np.array(list(range(316, 335)))

path10_inds = np.array(list(range(340, 348)))
path11_inds = np.array(list(range(367, 387)))
path12_inds = np.array(list(range(388, 398)))

path13_inds = np.array(list(range(401, 409)))
path14_inds = np.array(list(range(411, 421)))
path15_inds = np.array(list(range(425, 439)))

path16_inds = np.array(list(range(441, 451)))
path17_inds = np.array(list(range(452, 473)))
path18_inds = np.array(list(range(474, 490)))

path19_inds = np.array(list(range(492, 506)))
path20_inds = np.array(list(range(510, 516)))
path21_inds = np.array(list(range(517, 524)))

path22_inds = np.array(list(range(525, 535)))

paths = [path1_inds, path2_inds, path3_inds, path4_inds, path5_inds, path6_inds,
        path7_inds, path8_inds, path9_inds, path10_inds, path11_inds, path12_inds,
        path13_inds, path14_inds, path15_inds, path16_inds, path17_inds, path18_inds,
        path19_inds, path20_inds, path21_inds, path22_inds]


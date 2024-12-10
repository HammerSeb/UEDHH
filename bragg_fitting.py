### This tool is adapted from Laurenz Kremeyer, Siwick group, McGill University, Montreal

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from skued import bragg_peaks_persistence
from datetime import datetime

from uedhhlib import *




# from collections.abc import Iterable


start = datetime.now()

CYCLES = [i+1 for i in range(3)]
MASK = [True, True]
dset = Dataset(".", cycles=CYCLES)
PADDING = 60
dset.all_imgs = np.array(dset.all_imgs)

peaks, _, _, persistencies = bragg_peaks_persistence(dset.pump_off, min_dist=20, prominence=4000)
BRAGG_LOCATIONS = [p[::-1] for p in peaks if MASK[p[1], p[0]]]

# MINUTES = [ts.total_seconds() / 60 for ts in dset.timedeltas]
FIGSIZE = (16, 9)

fit_data = np.empty((len(BRAGG_LOCATIONS), 6, len(dset.all_imgs)))

with PdfPages("bragg_analysis.pdf") as pdf:

    f, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(
        dset.pump_off,
        origin="lower",
        vmin=0,
        vmax=np.percentile(dset.pump_off, 99.2),
        cmap="inferno",
    )
    for i, bragg_location in enumerate(BRAGG_LOCATIONS):
        ax.add_patch(
            patches.Rectangle(
                (bragg_location[1] - PADDING, bragg_location[0] - PADDING),
                PADDING * 2,
                PADDING * 2,
                linewidth=1,
                edgecolor="w",
                facecolor="none",
            )
        )
        ax.text(
            bragg_location[1],
            bragg_location[0] + PADDING * 1.5,
            f'{i}: {bragg_location[::-1]}',
            horizontalalignment="center",
            verticalalignment="center",
            color="w",
        )
    ax.axis("off")
    f.tight_layout()
    pdf.savefig(f)
    
#     for i, bragg_location in enumerate(BRAGG_LOCATIONS):
#         bragg_realtime = dset.all_imgs[
#             :,
#             bragg_location[0] - PADDING : bragg_location[0] + PADDING,
#             bragg_location[1] - PADDING : bragg_location[1] + PADDING,
#         ]
#         bragg_data = dset.data[
#             :,
#             bragg_location[0] - PADDING : bragg_location[0] + PADDING,
#             bragg_location[1] - PADDING : bragg_location[1] + PADDING,
#         ]

#         parameter_names = [
#             "amplitude",
#             "center_x",
#             "center_y",
#             "sigma_x",
#             "sigma_y",
#             "fraction",
#         ]
#         initial_guess = (2000, 25, 25, 1.75, 1.75, 0.5)
#         bounds = (
#             [0, 0, 0, 0, 0, 0],
#             [1e5, bragg_realtime[0].shape[0], bragg_realtime[0].shape[1], 5, 5, 1],
#         )
#         f, axs = plt.subplots(2, 6, figsize=FIGSIZE)
#         for ax in axs.flatten():
#             ax.grid()
#         f.suptitle(f"results for bragg peak {i} at {bragg_location[::-1]}")
        
#         try:
#             x, y = np.array(range(bragg_realtime[0].shape[0])), np.array(
#                 range(bragg_realtime[0].shape[1])
#             )
#             xx, yy = np.meshgrid(x, y)
#             popts, _ = fit_mp(pvoigt_2d, (xx, yy), bragg_realtime, initial_guess, bounds, max_workers=8)

#             fit_data[i] = popts.T


#             for ax, parameter, name in zip(axs[0], popts.T, parameter_names):
#                 ax.scatter(
#                     MINUTES,
#                     parameter,
#                     25,
#                     color=colors_from_arr(dset.real_time_delays),
#                     zorder=10,
#                 )
#                 ax.set_title(name)
#                 ax.set_xlabel("lab time [min]")
#         except RuntimeError:
#             pass

#         try:
#             x, y = np.array(range(bragg_data[0].shape[0])), np.array(
#                 range(bragg_data[0].shape[1])
#             )
#             xx, yy = np.meshgrid(x, y)
#             popts, _ = fit_mp(pvoigt_2d, (xx, yy), bragg_data, initial_guess, bounds, max_workers=8)

#             for ax, parameter, name in zip(axs[1], popts.T, parameter_names):
#                 ax.scatter(
#                     dset.delays,
#                     parameter,
#                     25,
#                     color=colors_from_arr(dset.delays),
#                     zorder=10,
#                 )
#                 ax.set_title(name)
#                 ax.set_xlabel("delay [ps]")
#         except:
#             pass

#         f.tight_layout()
#         pdf.savefig(f)

# # np.save('TiSe2_run_0034_bragg_fit_data.npy', fit_data)
print(f'{__file__} done; took {round((datetime.now()-start).total_seconds())}s to fit {len(BRAGG_LOCATIONS)} bragg peaks')

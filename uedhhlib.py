from typing import Union, Literal
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from lmfit.lineshapes import pvoigt
from matplotlib.cm import get_cmap
import matplotlib.colors as clr
from os import PathLike, listdir, cpu_count
from os.path import join, isfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from re import findall
from scipy.constants import speed_of_light
from tqdm import tqdm
from datetime import datetime
import h5py



class StaticDataSet:
    def __init__(self, path: PathLike, background: Union[PathLike, None] = None):
        """
        generate static dataset from folder. perform background correction if background file is given

        Parameters
        ----------
        path : PathLike
            _description_
        background : Union[PathLike, None], optional
            _description_, by default None
        """

        self.path = path
        self.background_file = background

        filelist = [
            entry for entry in listdir(self.path) if isfile(join(self.path, entry))
        ]

        _raw_imgs = []
        self.imgs = []

        for file in filelist:
            if background:
                if file.endswith(".tif") and file != background:
                    _raw_imgs.append(self._load_image(join(self.path, file)))
                elif file == background:
                    self.background = self._load_image(join(self.path, file))

            else:
                if file.endswith(".tif"):
                    self.imgs.append(self._load_image(join(self.path, file)))

        if background:
            for img in _raw_imgs:
                self.imgs.append(img - self.background)

        self.mean = np.mean(np.array(self.imgs), axis=0)

    def save_mean(self, path: PathLike):
        """
        save dataset

        Parameters
        ----------
        path : PathLike
            _description_
        """
        np.save(path, self.mean)

    def _load_image(self, filepath: PathLike) -> np.ndarray:
        return np.array(Image.open(filepath), dtype=np.float32)


class Dataset:
    def __init__(
        self,
        basedir: PathLike,
        mask: np.ndarray = None,
        correct_laser: bool = True,
        all_imgs: bool = False,
        progress: bool = True,
        cycles: Union[int, tuple] = None,
        ignore: list = None,
    ):
        """
        Loads full UED dataset taken in the SchwEpp group at the MPSD for further analysis.
        save() method saves h5 file which can be opened in Iris.
        Parameters
        ----------
        basedir : PathLike
            base directory cotaining the "Cycle X" directories
        mask : np.ndarray, optional
            constant mask applied to all images of the delay scans. If 'None' all points of the images are used, by default None
        correct_laser : bool, optional
            wether laser background is corrected for or not, by default True
        all_imgs : bool, optional
            wether all raw images are kept or just the final data set is kept. Using this makes the data set LARGE. USE with care!, by default False
        progress : bool, optional
            turn on progress notification during data loading, by default True
        cycles : Union[int,tuple], optional
            For now, give this parameter an iterable containing the cycle number you want to load, e.g: (1,2,4,7,10,11,12) to load cycles 1,2,4.... you get it
        ignore : list
             option are:
              - list of tuples of the form (cycle_number, stage_position as string, frame number). To ignore frames three of cycle 5 at stage position 105.4 mm use (5, 105.4, 3).
              Note: Make sure that stage position is given with correct decimal separator "."

              Should be fixed now!
              Note: Ingoring files can lead to errors if all frames of a single delay step are sorted out. I am working on a fix, handle with care for now.
        """

        self.basedir = basedir
        self.mask = mask
        self.correct_laser = correct_laser
        self.progress = progress
        self.cycles = cycles
        self.ignore = ignore

        # decide if all images are kept or not
        if all_imgs:
            self.all_imgs = []
            self.all_imgs_flag = True
        else:
            self.all_imgs_flag = False

        self.real_time_intensities = []
        self.loaded_files = []
        self.timestamps = []

        # load pump off files
        if self.progress:
            print("loading pump offs")
        self._load_pump_offs()

        # load laser only files
        if self.progress:
            print("loading pump only")
        self._load_pump_only()

        # make list of ignored files
        if self.ignore:
            if self.progress:
                print("compile list of ignored files")
            if isinstance(self.ignore, list):
                self._make_ignored_files_list()
        else:
            self.ignored_files = []

        # infere standard mask from pump off shape
        if not self.mask:
            self.mask = np.ones(self.pump_off.shape)

        # get delay time steps, smallest delay time is arbitrarily set to 0 ps
        if self.progress:
            print("accessing delay times")
        self.delaytime_from_stage_position = self._get_delay_times_mapping()
        self.delay_times = [
            self.delaytime_from_stage_position(position)
            for position in self.stage_positions
        ]

        # this array stores how many times a delay time step has no entry per cycle because all frames of a cycle at this state position had to be ignored due to arcing
        self._empties = np.zeros(len(self.delay_times))

        # This is were all the data from each cycle is loaded and averaged
        self.data = np.zeros(
            (len(self.delay_times), self.pump_off.shape[0], self.pump_off.shape[1])
        )
        if self.progress:
            print("loading cycles")
            for cycle in tqdm(cycles):
                self.data += np.array(self._load_cycle(cycle))

            self.data /= len(self.cycles) - self._empties[:, np.newaxis, np.newaxis]

        else:
            for cycle in cycles:
                self.data += np.array(self._load_cycle(cycle))

            self.data /= len(self.cycles) - self._empties[:, np.newaxis, np.newaxis]

        # here we sort the data so that small delay times are at low index values just for convenience
        self.delay_times = self.delay_times[::-1]
        self.stage_positions = self.stage_positions[::-1]
        self.data = self.data[::-1]

        # here we sort the image intensities, loaded files and images according the lab time they were recorded
        if self.all_imgs_flag:
            (
                self.timestamps,
                self.all_imgs,
                self.real_time_intensities,
                self.loaded_files,
            ) = zip(
                *sorted(
                    zip(
                        self.timestamps,
                        self.all_imgs,
                        self.real_time_intensities,
                        self.loaded_files,
                    )
                )
            )
        else:
            self.timestamps, self.real_time_intensities, self.loaded_files = zip(
                *sorted(
                    zip(self.timestamps, self.real_time_intensities, self.loaded_files)
                )
            )

        self.timestamps = [timestamp.isoformat() for timestamp in self.timestamps]

        self.mean = np.mean(self.data, axis=0)

    def save(self, filename: PathLike):
        """
        saves the dataset as an h5 file which can be read by Iris

        Parameters
        ----------
        filename : PathLike
            filepath
        """
        with h5py.File(filename, "w") as f:
            f.create_dataset("time_points", data=self.delay_times)
            f.create_dataset("valid_mask", data=self.mask)
            proc_group = f.create_group("processed")
            proc_group.create_dataset("equilibrium", data=self.pump_off)
            proc_group.create_dataset("intensity", data=np.moveaxis(self.data, 0, -1))
            realtime_group = f.create_group("real_time")
            realtime_group.create_dataset("intensity", data=self.real_time_intensities)
            realtime_group.create_dataset("loaded_files", data=self.loaded_files)
            realtime_group.create_dataset("timestamps", data=self.timestamps)
            if self.all_imgs_flag:
                realtime_group.create_dataset("all_imgs", data=self.all_imgs)

    def _load_cycle(self, cycle: int) -> list:
        """
        loads the data of cycle one.

        Parameters
        ----------
        cycle : int
            cycle number of cycle to be loaded

        Returns
        -------
        list
            list of arrays containing the diffraction data of each stage position averaged ober all recorded frames
        """
        _cycle_path = join(self.basedir, f"Cycle {cycle}")
        _filelist = listdir(_cycle_path)
        cycle_data = []
        for _idx, position in enumerate(self.stage_positions):
            _position_files = []
            _name = f"z_ProbeOnPumpOn_{str(position).replace(".",",")} mm_Frm"
            if isinstance(self.ignore, list):
                for file in _filelist:
                    if (
                        _name in file
                        and file.endswith(".npy")
                        and join(_cycle_path, file) not in self.ignored_files
                    ):
                        _position_files.append(file)
            else:
                for file in _filelist:
                    if (
                        _name in file
                        and file.endswith(".npy")
                        and join(_cycle_path, file)
                    ):  # the "and join(...)" condition at the end is unnecessary and should be deleted in future versions
                        _position_files.append(file)

            # check for empty image and fill with zeros in cas
            if not _position_files:
                self._empties[_idx] += 1
                _position_data = np.zeros((1, *self.pump_off.shape))

            else:
                # load images
                _position_data = []
                for file in _position_files:
                    _img = np.load(join(_cycle_path, file))
                    self.real_time_intensities.append(_img.sum())
                    self.loaded_files.append(join(_cycle_path, file))

                    # extract epoch timestamp from server-path of loaded file
                    _serverpath_file = file.split(".")[0] + ".txt"
                    with open(join(_cycle_path, _serverpath_file), "r") as f:
                        self.timestamps.append(
                            datetime.fromtimestamp(
                                int(findall(r"(?<=\\A1\\)\d+", f.readlines()[1])[0])
                            )
                        )

                    if self.all_imgs_flag:
                        self.all_imgs.append(_img)

                    if self.correct_laser:
                        _position_data.append((_img - self.pump_only) * self.mask)
                    else:
                        _position_data.append(_img * self.mask)

            cycle_data.append(np.mean(_position_data, axis=0))
        return cycle_data

    def _get_delay_times_mapping(self):
        """
        get the delay time steps from Cycle 1

        Returns
        -------
        function
            this function maps a stage position to a relative time with respect to the lowest delay time
        """
        self.stage_positions = []
        for file in sorted(listdir(join(self.basedir, "Cycle 1"))):
            if "ProbeOnPumpOn" in file and "Frm1" in file and file.endswith(".npy"):
                self.stage_positions.append(
                    float(findall(r"\d+\,\d*", file)[0].replace(",", "."))
                )

        def delaytime_from_stageposition(position):
            speed_of_light_mm_per_ps = speed_of_light * 1e3 / 1e12
            pos_zero = max(self.stage_positions)
            return 2 * (pos_zero - position) / speed_of_light_mm_per_ps

        return delaytime_from_stageposition

    def _make_ignored_files_list(self):
        """
        makes a list of files to be ignored during loading
        """
        self.ignored_files = [0]
        for ign in self.ignore:
            self.ignored_files.append(
                join(
                    join(self.basedir, f"Cycle {int(ign[0])}"),
                    f"z_ProbeOnPumpOn_{str(ign[1]).replace(".", ",")} mm_Frm{int(ign[2])}.npy",
                )
            )

    def _load_pump_offs(self):
        """
        loads the pump off data of all cycles and makes a mean pump off image from all of them
        """
        self.pump_offs = []
        for cycle in self.cycles:
            _cycle_path = join(self.basedir, f"Cycle {int(cycle)}")
            _pump_off_list = []
            for file in listdir(_cycle_path):
                if "ProbeOnPumpOff" in file and file.endswith(".npy"):
                    _pump_off_list.append(join(_cycle_path, file))

            for pumpoff in sorted(_pump_off_list):
                self.pump_offs.append(np.load(pumpoff))

        self.pump_off = np.mean(np.array(self.pump_offs), axis=0)

    def _load_pump_only(self):
        """
        loads the pump only data of all cycles and makes a mean pump off image from all of them
        """
        self.pump_onlys = []
        for cycle in self.cycles:
            _cycle_path = join(self.basedir, f"Cycle {int(cycle)}")
            _pump_only_list = []
            for file in listdir(_cycle_path):
                if "ProbeOffPumpOn" in file and file.endswith(".npy"):
                    _pump_only_list.append(join(_cycle_path, file))

            for pumponly in sorted(_pump_only_list):
                self.pump_onlys.append(np.load(pumponly))

        self.pump_only = np.mean(np.array(self.pump_onlys), axis=0)



#### THIS IS ALL FOR bragg_fitting.py

# def color_enumerate(iterable, start=0, cmap=CMAP):
#     """
#     same functionality as enumerate, but additionally yields sequential colors from
#     a given cmap
#     """

#     n = start
#     try:
#         length = len(iterable)
#     except TypeError:
#         length = len(list(iterable))
#     for item in iterable:
#         yield n, cmap(n/(length-1)), item
#         n += 1


def colors_from_arr(a, cmap=None, start=0, end=1):
    if cmap is None:
        # https://davidjohnstone.net/lch-lab-colour-gradient-picker#fcf0ba,3774a0
        # orange -> purple
        # clist = ['#ff854d', '#ff834d', '#ff814e', '#ff7f4e', '#ff7d4f', '#ff7b4f', '#ff7950', '#ff7751', '#ff7551', '#ff7352', '#ff7152', '#fe6f53', '#fe6d54', '#fe6a55', '#fe6855', '#fd6656', '#fd6457', '#fd6258', '#fc6059', '#fc5e5a', '#fc5c5a', '#fb595b', '#fb575c', '#fa555d', '#fa535e', '#f9515f', '#f84f60', '#f84c61', '#f74a62', '#f64863', '#f64664', '#f54466', '#f44167', '#f33f68', '#f23d69', '#f23a6a', '#f1386b', '#f0366c', '#ef346d', '#ee316f', '#ec2f70', '#eb2c71', '#ea2a72', '#e92873', '#e82575', '#e62376', '#e52077', '#e41d78', '#e21b7a', '#e1187b', '#df157c', '#de127e', '#dc0f7f', '#db0c80', '#d90881', '#d70583', '#d60284', '#d40085', '#d20087', '#d00088', '#ce0089', '#cc008b', '#ca008c', '#c8008d', '#c6008e', '#c40090', '#c20091', '#bf0092', '#bd0094', '#bb0095', '#b80096', '#b60097', '#b30099', '#b0009a', '#ae009b', '#ab009c', '#a8009e', '#a5019f', '#a203a0', '#9f05a1', '#9c08a2', '#990aa3', '#960da5', '#930fa6', '#8f11a7', '#8c14a8', '#8815a9', '#8517aa', '#8119ab', '#7d1bac', '#791cad', '#751eae', '#701faf', '#6c21b0', '#6722b1', '#6223b2', '#5d25b2', '#5826b3', '#5227b4']
        # rose -> blue
        # clist = ['#ff7575', '#ff7477', '#ff7479', '#fe737a', '#fe727c', '#fe727e', '#fe7180', '#fd7182', '#fd7084', '#fc7086', '#fc6f87', '#fb6f89', '#fa6e8b', '#fa6e8d', '#f96e8f', '#f86e91', '#f76d92', '#f76d94', '#f66d96', '#f56d98', '#f46d9a', '#f36d9b', '#f26d9d', '#f16d9f', '#ef6da1', '#ee6da2', '#ed6da4', '#ec6da6', '#ea6da7', '#e96da9', '#e76dab', '#e66eac', '#e46eae', '#e36eb0', '#e16eb1', '#e06fb3', '#de6fb4', '#dc6fb6', '#da70b7', '#d970b9', '#d771ba', '#d571bc', '#d371bd', '#d172be', '#cf72c0', '#cd73c1', '#cb73c2', '#c974c4', '#c674c5', '#c475c6', '#c275c7', '#c076c8', '#bd76c9', '#bb77ca', '#b977cb', '#b678cc', '#b478cd', '#b179ce', '#af79cf', '#ac7ad0', '#a97bd1', '#a77bd2', '#a47cd2', '#a17cd3', '#9e7dd4', '#9c7dd4', '#997ed5', '#967ed6', '#937fd6', '#907fd7', '#8d80d7', '#8a80d7', '#8780d8', '#8481d8', '#8081d8', '#7d82d9', '#7a82d9', '#7783d9', '#7383d9', '#7083d9', '#6c84d9', '#6984d9', '#6585d9', '#6185d9', '#5e85d9', '#5a86d9', '#5686d9', '#5286d8', '#4d87d8', '#4987d8', '#4487d7', '#4087d7', '#3b88d6', '#3588d6', '#2f88d5', '#2988d5', '#2189d4', '#1789d4', '#0989d3']
        # beige -> blue
        clist = [
            "#fcf0ba",
            "#f9efb9",
            "#f6eeb8",
            "#f3edb7",
            "#f0ecb6",
            "#edebb5",
            "#eaeab4",
            "#e7eab3",
            "#e4e9b2",
            "#e1e8b1",
            "#dee7b0",
            "#dbe6b0",
            "#d8e5af",
            "#d5e4ae",
            "#d1e3ae",
            "#cee2ad",
            "#cbe1ad",
            "#c8e0ac",
            "#c5dfac",
            "#c2deab",
            "#bfddab",
            "#bcdcab",
            "#b9dbaa",
            "#b6daaa",
            "#b3d9aa",
            "#b0d8aa",
            "#add6a9",
            "#aad5a9",
            "#a7d4a9",
            "#a4d3a9",
            "#a1d2a9",
            "#9ed1a9",
            "#9bd0a9",
            "#98cfa9",
            "#95cea9",
            "#92cca9",
            "#8fcba9",
            "#8ccaa9",
            "#89c9a9",
            "#86c8a9",
            "#83c7a9",
            "#80c5aa",
            "#7dc4aa",
            "#7ac3aa",
            "#78c2aa",
            "#75c0aa",
            "#72bfaa",
            "#6fbeab",
            "#6cbdab",
            "#6abbab",
            "#67baab",
            "#64b9ab",
            "#61b8ac",
            "#5fb6ac",
            "#5cb5ac",
            "#5ab4ac",
            "#57b2ac",
            "#54b1ac",
            "#52b0ad",
            "#4faead",
            "#4dadad",
            "#4aabad",
            "#48aaad",
            "#46a9ad",
            "#44a7ad",
            "#41a6ad",
            "#3fa5ad",
            "#3da3ad",
            "#3ba2ad",
            "#39a0ad",
            "#379fad",
            "#369dad",
            "#349cad",
            "#329aad",
            "#3199ad",
            "#3098ad",
            "#2f96ad",
            "#2e95ac",
            "#2d93ac",
            "#2c92ac",
            "#2c90ac",
            "#2b8fab",
            "#2b8dab",
            "#2b8caa",
            "#2b8aaa",
            "#2b88aa",
            "#2b87a9",
            "#2c85a9",
            "#2d84a8",
            "#2d82a7",
            "#2e81a7",
            "#2f7fa6",
            "#307ea5",
            "#317ca4",
            "#327aa4",
            "#3379a3",
            "#3477a2",
            "#3676a1",
            "#3774a0",
        ]
        cmap = clr.LinearSegmentedColormap.from_list("my map", clist)
        start = 0.2
        end = 1
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    a = np.array(a)
    a = a.astype("float")
    a += np.abs(a.min())
    a -= a.min()
    a /= a.max()
    color_scale = start + a * (end - start)
    return cmap(start + a * (end - start))

def pvoigt_2d(data_tuple, amplitude, center_x, center_y, sigma_x, sigma_y, fraction):
    x, y = data_tuple
    return (
        amplitude
        * pvoigt(x, 1, center_x, sigma_x, fraction)
        * pvoigt(y, 1, center_y, sigma_y, fraction)
    ).ravel()


def pvoigt_2d_height(amplitude, center_x, center_y, sigma_x, sigma_y, fraction):
    return pvoigt_2d(
        (center_x, center_y), amplitude, center_x, center_y, sigma_x, sigma_y, fraction
    )

def fit_worker(*args, **kwargs):
    args = list(args)
    args[2] = args[2].ravel()
    return curve_fit(*args, **kwargs)

def fit_mp(
    fit_func,
    coords,
    iterable_data,
    initial_guess=None,
    bounds=None,
    update_guess=True,
    max_workers=None,
    progress=False,
):
    if update_guess:
        initial_guess, _ = fit_worker(
            fit_func, coords, iterable_data[0], p0=initial_guess, bounds=bounds
        )
    if max_workers is None:
        max_workers = cpu_count() - 2
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fit_worker, fit_func, coords, data, p0=initial_guess, bounds=bounds
            )
            for data in iterable_data
        ]
        if progress:
            for _ in tqdm(
                as_completed(futures),
                total=len(iterable_data),
                desc=f"fitting {fit_func.__name__}",
            ):
                pass
        executor.shutdown(wait=True)
        popts = np.array([f.result()[0] for f in futures])
        pcovs = np.array([f.result()[1] for f in futures])
    return popts, pcovs
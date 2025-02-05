from typing import Union, Literal, Tuple
import numpy as np
from PIL import Image
from os import PathLike, listdir
from os.path import join, isfile
from re import findall
from scipy.constants import speed_of_light
from tqdm import tqdm
from datetime import datetime
from lmfit.model import Model, ModelResult
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
                                int(findall(r"(?<=\\A2\\)\d+", f.readlines()[1])[0])
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
        self.stage_positions = sorted(self.stage_positions)

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


def pvoigt_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    m: float,
    amp: float,
    x0: float,
    y0: float,
    a: float,
    b: float,
    c: float,
    bg_sx: float,
    bg_sy: float,
    bg_offset: float,
) -> np.ndarray:
    """2D pseudo-voigt profile with linear background with a general quadratic form Q(x,y) = a*(x-x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0**2). This enables fitting of 2d-line profiles that are rotated with respect to the general xy-coordinate system.  (see: https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation)

    Parameters
    ----------
    xy : (np.ndarray, np.ndarray)
        x- and y-coordinate as tuple of arrays (x,y)
    m : float
        mixing parameter of lorentz and gaussian line shape (0 <= m <= 1)
    amp : float
        amplitude
    x0 : float
        center x-value
    y0 : float
        center y-value
    a : float
        a parameter of quadratic form Q(x,y)
    b : float
        b parameter of quadratic form Q(x,y)
    c : float
        c parameter of quadratic form Q(x,y)
    bg_sx : float
        background, slope in x-direction
    bg_sy : float
        background, slope in y-direction
    bg_offset : float
        background, constant offset

    Returns
    -------
    np.ndarray

    """
    x, y = xy
    quadratic_form = a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2
    lorentz = 1 / (1 + 4 * quadratic_form)
    gaussian = np.exp(-4 * np.log(2) * quadratic_form)
    background = bg_sx * x + bg_sy * y + bg_offset
    return amp * (m * lorentz + (1 - m) * gaussian) + background


def fit_pvoigt_2d(data: np.ndarray, initial_guess=None) -> ModelResult:
    """fits a 2d pseudo-voigt profile to a Bragg peak

    Parameters
    ----------
    data : np.ndarray
        2d data containing one Bragg peak

    Returns
    -------
    ModelResult
        lmfit ModelResult of the fitting process
    """

    nx, ny = data.shape

    # Create coordinate arrays for x and y
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)  # Note: X and Y have the same shape as data

    # 1D arrays for lmfit:
    x_flat = X.ravel()
    y_flat = Y.ravel()
    data_flat = data.ravel()

    pvoigt_model = Model(pvoigt_2d, independent_vars=["xy"])

    if not initial_guess:
        pvoigt_params = pvoigt_model.make_params(
            m=0.5,
            amp=data.max(),
            x0=nx / 2,
            y0=ny / 2,
            a=1 / (2 * (nx / 4) ** 2),
            b=0,
            c=1 / (2 * (ny / 4) ** 2),
            bg_sx=(data_flat[-1] - data_flat[0]) / nx,
            bg_sy=(data_flat[-1] - data_flat[0]) / ny,
            bg_offset=data.min(),
        )
    else:
        pvoigt_params = pvoigt_model.make_params()

        for name, val in zip(pvoigt_model.param_names, initial_guess):
            pvoigt_params[name].set(value=val)

    pvoigt_params["m"].set(min=0, max=1)
    pvoigt_params["amp"].set(min=0, max=2 * data.max())
    pvoigt_params["a"].set(min=1 / (2 * (nx) ** 2)),
    pvoigt_params["b"].set(min=0),
    pvoigt_params["c"].set(min=1 / (2 * (ny) ** 2)),
    pvoigt_params["bg_offset"].set(min=0),

    return pvoigt_model.fit(
        data_flat, pvoigt_params, xy=[x_flat, y_flat], nan_policy="propagate"
    )

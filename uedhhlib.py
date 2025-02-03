from typing import Union, Literal, Tuple
import numpy as np
from PIL import Image
from os import PathLike, listdir
from os.path import join, isfile
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


def pvoigt_2d(x: np.ndarray,
            y: np.ndarray,
            m: float,
            amp: float,
            center: Tuple,
            q_form_parameters: Tuple,
            bg_parameters: Tuple):
    """2D pseudo-voigt profile with linear background with a general quadratic form Q(x,y) = a*(x-x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0**2). This enables fitting of 2d-line profiles that are rotated with respect to the general xy-coordinate system.  (see: https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation)

    Parameters
    ----------
    x : np.ndarray
        x-coordinate 
    y : np.ndarray
        y-coordinate
    m : float
        mixing parameter of lorentz and gaussian line shape (0 <= m <= 1)
    amp : float
        amplitude 
    center : Tuple
        (x0, y0) center of the profile
    q_form_parameters : Tuple
        parameters of the quadratic form (a, b, c)
    bg_parameters : Tuple
        parameters for linear background (m_x, m_y, offset), where m_x and m_y are the slopes in x and y direction and offset is a constant offset. 
    """
    quadratic_form = q_form_parameters[0] * (x - center[0])**2 + q_form_parameters[1] * (x - center[0]) * (y - center[1]) + q_form_parameters[2] * (y - center[1])**2
    lorentz = 1 / ( 1 + 4 * quadratic_form)
    gaussian = np.exp(-4 * np.log(2) * quadratic_form)
    background = bg_parameters[0] * x + bg_parameters[1] * y + bg_parameters[2] 
    return amp * (m * lorentz + (1 - m) *  gaussian) + background
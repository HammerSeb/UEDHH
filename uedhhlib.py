from typing import Union
import numpy as np
from PIL import Image
from os import PathLike, listdir
from os.path import join, isfile
from re import findall
from scipy.constants import speed_of_light
from tqdm import tqdm
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

        filelist = [entry for entry in listdir(self.path) if isfile(join(self.path,entry))]

        _raw_imgs = []
        self.imgs = []

        for file in filelist:
            if background:
                if file.endswith(".tif") and file != background:
                    _raw_imgs.append(self._load_image(join(self.path,file)))
                elif file == background:
                    self.background = self._load_image(join(self.path,file))

            else:
                if file.endswith(".tif"):
                    self.imgs.append(self._load_image(join(self.path, file)))

        if background:
            for img in _raw_imgs:
                self.imgs.append(img-self.background)

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
    def __init__(self, 
                 basedir: PathLike,
                 mask: np.ndarray = None,
                 correct_laser:bool = True,
                 all_imgs: bool = True,
                 progress: bool = True,
                 cycles: Union[int,tuple] = None,
                 ignore: list = None):
        """
        _summary_

        Parameters
        ----------
        basedir : PathLike
            _description_
        all_imgs : bool, optional
            _description_, by default True
        progress : bool, optional
            _description_, by default True
        cycles : Union[int,tuple], optional
            _description_, by default None
        ignore : list, optional
            list of tuples of the form (cycle_number, stage_position as string, (frame1, frame2,...)). To ignore three frames of cycle 5 at stage position 105.4 mm use (5, "105,4", (1,2,3)).
        """
        
        self.basedir = basedir

        self.mask = mask

        self.correct_laser = correct_laser

        if all_imgs:
            self.all_imgs = []
            self.all_imgs_flag = True
        else:
            self.all_imgs_flag = False

        self.progress = progress
        self.cycles = cycles
        self.ignore = ignore

        self.real_time_intensities = []
        self.loaded_files = []

        if self.progress:
            print("loading pump offs")
        self._load_pump_offs()

        if not self.mask:
            self.mask = np.ones(self.pump_off.shape)

        if self.progress:
            print("loading pump only")
        self._load_pump_only()

        if self.ignore:
            if self.progress:
                print("compile list of ignored files")
            self._make_ignored_files_list()
        else:
            self.ignored_files = []

        if self.progress:
            print("accessing delay times")
        self.delaytime_from_stage_position = self._get_delay_times_mapping()
        self.delay_times = [self.delaytime_from_stage_position(position) for position in self.stage_positions]

        self._empties = np.zeros(len(self.delay_times))

        self.data = np.zeros((len(self.delay_times), self.pump_off.shape[0], self.pump_off.shape[1]))
        if self.progress:
            print("loading cycles")
            for cycle in tqdm(cycles):
                self.data += np.array(self._load_cycle(cycle))
            
            self.data /= len(self.cycles)

        else:
            for cycle in cycles:
                self.data += np.array(self._load_cycle(cycle))
            
            self.data /= len(self.cycles)


        self.delay_times = self.delay_times[::-1]
        self.stage_positions = self.stage_positions[::-1]
        self.data = self.data[::-1]

    # def save(self, filename):
    #     with h5py.File(filename, "w") as f:
    #         f.create_dataset("time_points", data=self.delay_times)
    #         f.create_dataset("valid_mask", data=self.mask)
    #         proc_group = f.create_group("processed")
    #         proc_group.create_dataset("equilibrium", data=self.pump_off)
    #         proc_group.create_dataset("intensity", data=np.moveaxis(self.data, 0, -1))
    #         realtime_group = f.create_group("real_time")
    #         realtime_group.create_dataset(
    #             "minutes", data=[td.total_seconds() / 60 for td in self.timedeltas]
    #         )
    #         realtime_group.create_dataset("intensity", data=self.real_time_intensities)
    #         realtime_group.create_dataset("time_points", data=self.real_time_delays)

    def _load_cycle(self, cycle):
        _cycle_path = join(self.basedir, f"Cycle {cycle}")
        _filelist = listdir(_cycle_path)
        cycle_data = []
        for _idx, position in enumerate(self.stage_positions):
            _position_files = []
            _name = f"z_ProbeOnPumpOn_{str(position).replace(".",",")} mm_Frm"
            for file in _filelist:
                if _name in file and file.endswith(".npy") and join(_cycle_path,file) not in self.ignored_files:
                    _position_files.append(file)
            
            if not _position_files:
                self._empties[_idx] +=1
            
            _position_data = []
            for file in _position_files:
                _img = np.load(join(_cycle_path,file))
                self.real_time_intensities.append(_img.sum())
                self.loaded_files.append(join(_cycle_path,file))

                if self.all_imgs_flag:
                    self.all_imgs.append(_img)
                
                if self.correct_laser:
                    _position_data.append((_img - self.pump_only)*self.mask)
                else:
                    _position_data.append(_img*self.mask)
            
            cycle_data.append(np.mean(_position_data, axis=0))
        return cycle_data
                
    def _get_delay_times_mapping(self):
        self.stage_positions = []
        for file in sorted(listdir(join(self.basedir, "Cycle 1"))):
            if "ProbeOnPumpOn" in file and "Frm1" in file and file.endswith(".npy"):
                self.stage_positions.append(float(findall(r"\d+\,\d*", file)[0].replace(",",".")))
        
        def delaytime_from_stageposition(position):
            speed_of_light_mm_per_ps = speed_of_light *1e3 /1e12
            pos_zero = max(self.stage_positions)
            return 2*(pos_zero-position)/speed_of_light_mm_per_ps

        return delaytime_from_stageposition

    def _make_ignored_files_list(self):
        self.ignored_files = [0]
        for ign in self.ignore:
            for frame in ign[2]:
                self.ignored_files.append(
                    join(
                    join(self.basedir, f"Cycle {int(ign[0])}"), f"z_ProbeOnPumpOn_{ign[1]} mm_Frm{int(frame)}.npy"
                    )
                )

    def _load_pump_offs(self):
        self.pump_offs = []
        for cycle in self.cycles:
            _cycle_path = join(self.basedir,f"Cycle {int(cycle)}")
            _pump_off_list = []
            for file in listdir(_cycle_path):
                if "ProbeOnPumpOff" in file and file.endswith(".npy"):
                    _pump_off_list.append(join(_cycle_path, file))

            for pumpoff in sorted(_pump_off_list):
                self.pump_offs.append(np.load(pumpoff))

        self.pump_off = np.mean(np.array(self.pump_offs), axis=0)
            

    def _load_pump_only(self):
        self.pump_onlys = []
        for cycle in self.cycles:
            _cycle_path = join(self.basedir,f"Cycle {int(cycle)}")
            _pump_only_list = []
            for file in listdir(_cycle_path):
                if "ProbeOffPumpOn" in file and file.endswith(".npy"):
                    _pump_only_list.append(join(_cycle_path, file))

            for pumponly in sorted(_pump_only_list):
                self.pump_onlys.append(np.load(pumponly))

        self.pump_only = np.mean(np.array(self.pump_onlys), axis=0)
# uedhh

contains uedhhlib which is a collection of tools to analyze electron diffraction data recorded at MPSD - November 24

## Requirments

required packages are

- ```numpy```
- ```matplotlib```
- ```pillow```
- ```tqdm```
- ```h5py```

## Data Structure

### Delay time ordering

The log file of each cycle contains the absolute times corresponding to the stage positions. The delay times are in reverse order compared to the absolute times. Hence, larger stage position corresponds to longer absolute time means earlier delay times
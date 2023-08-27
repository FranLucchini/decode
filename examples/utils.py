import os
import rioxarray
import mxnet as mx
import xarray as xr

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def file_to_nparray(file_path, bands=['B2','B3','B4','B8']):
    # Open .nc or .tif file with XArray
    ds = xr.open_dataset(file_path)
    # Transform variables from selected bands to numpy
    np = ds[bands].to_array().to_numpy()
    # Reshape array
    imgs_np = [np[:, t , :, :] for t in range(np.shape[1])]
    return imgs_np

# input_path = '~/datasets/AI4Boundaries2/sentinel2/images/AT/AT_10032_S2_10m_256.nc'
# outs = file_to_nparray(input_path)
# len(outs), outs[0].shape

def label_to_nparray(label_path, img_count):
    # Open .nc or .tif file with XArray, filter bands
    ds = rioxarray.open_rasterio(label_path)[1:]
    label_np = np.array([[ds]]*img_count)
    labels_np = label_np.tolist()
    return labels_np
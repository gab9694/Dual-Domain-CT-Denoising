import numpy as np
import argparse
import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from joblib import Memory
from helper import save_to_tiff_stack_with_metadata, load_tiff_stack_with_metadata
from rebinning_functions import rebin_curved_to_flat_detector, rebin_helical_to_fan_beam_trajectory, \
    _rebin_curved_to_flat_detector_core, rebin_curved_to_flat_detector_multiprocessing
from read_data import read_dicom


def run(parser, basic_noise, dose=None):
    args = parser.parse_args()
    print('Processing scan {}.'.format(args.scan_id))

    # Load projections and read out geometry data from the DICOM header.
    raw_projections, parser = read_dicom(parser)
    args = parser.parse_args()
    dose_level = 200
    
    # we can insert noise here to simulate low-dose data
    # method refer from paper.''Noise simulation for low-dose computed tomography''
    if dose:
        dose_level = dose
        raw_projections = raw_projections + basic_noise * np.sqrt( ((200 - dose_level) / dose_level) * np.exp(raw_projections) )
    
    if args.save_all:
        save_path = Path(args.path_out) / Path(f'{args.scan_id}_curved_helix_projections_{dose_level}mAs.tif')
        save_to_tiff_stack_with_metadata(raw_projections,
                                         save_path,
                                         metadata=vars(args))
        
    # Rebin helical projections from curved detector to flat detector.
    # Step can be skipped if the reconstruction supports curved detectors.
    if args.no_multiprocessing:
        proj_flat_detector = rebin_curved_to_flat_detector(args, raw_projections)
    else:
        location = 'tmp/cache_dir'  # Todo: '/home/fabian/Desktop/tmp/cache_dir'
        memory = Memory(location, verbose=0)

        cached_rebin_curved_to_flat_detector_core = memory.cache(_rebin_curved_to_flat_detector_core)
        data = (args, raw_projections)
        proj_flat_detector = np.array(Parallel(n_jobs=8)(
            delayed(rebin_curved_to_flat_detector_multiprocessing)(data, col)
            for col in tqdm.tqdm(range(data[1].shape[0]), 'Rebin curved to flat detector')))

    if args.save_all:
        save_path = Path(args.path_out) / Path(f'{args.scan_id}_flat_helix_projections_{dose_level}mAs.tif')
        save_to_tiff_stack_with_metadata(proj_flat_detector,
                                         save_path,
                                         metadata=vars(args))

    # Rebinning of projections acquired on a helical trajectory to full-scan (2pi) fan beam projections.
    proj_fan_geometry = rebin_helical_to_fan_beam_trajectory(args, proj_flat_detector)

    save_path = Path(args.path_out) / Path(f'{args.scan_id}_flat_fan_projections_{dose_level}mAs.tif')
    save_to_tiff_stack_with_metadata(proj_fan_geometry,
                                     save_path,
                                     metadata=vars(args))

    print('Finished. Results saved at {}.'.format(save_path.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 1. select segment in parser.add_argument
    # parser.add_argument('--path_dicom', type=str, required=True, help='Local path of helical projection data.')
    parser.add_argument('--path_dicom', type=str, default='to/path', help='Local path of helical projection data.')
    parser.add_argument('--path_out', type=str, default='out', help='Output path of rebinned data.')
    parser.add_argument('--scan_id', type=str, default='L506', help='Custom scan ID.')
    parser.add_argument('--idx_proj_start', type=int, default=16500, help='First index of helical projections that are processed.')
    parser.add_argument('--idx_proj_stop', type=int, default=27500, help='Last index of helical projections that are processed.')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='Save all intermediate results.')
    parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', action='store_true', help='Switch off multiprocessing using joblib.')
    
    # 2. make random noise and insert into projection data
    basic_noise = np.random.normal(0, 1, (2304, 736, 266))

    # 3. run the rebinning function
    run(parser, basic_noise)


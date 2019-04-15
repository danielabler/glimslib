import os
import shlex
import subprocess
import shutil

import glimslib.utils.file_utils as fu

def ants_apply_transforms(input_img, reference_img, output_file, transforms, dim=3):
    print("  - Starting ANTS Apply Transforms:")
    print("    - INPUT IMG      : %s"%input_img)
    print("    - REFERENCE IMG  : %s" % reference_img)
    print("    - OUTPUT         : %s" % output_file)
    ants_cmd = build_ants_apply_transforms_command(input_img, reference_img, output_file, transforms, dim)
    print("ANTS command: %s"%ants_cmd)
    fu.ensure_dir_exists(os.path.dirname(output_file))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args, env=os.environ.copy())
    process.wait()
    #return process.returncode
    print("ANTS apply transforms terminated with return code: '%s'"%process.returncode)


def build_ants_apply_transforms_command(input_img, reference_img, output_file, transforms, dim=3):
    ants_params_dict = {'verbose': 0,
                        'dimensionality': dim,
                        'input-image-type': 'scalar',
                        'input' : input_img,
                        'reference-image': reference_img,
                        'output': output_file,
                        'interpolation': 'Linear'}
    ants_params_str = ' --'.join([' '.join([key, str(value)]) for key, value in ants_params_dict.items()])
    for transform in transforms:
        ants_params_str = ants_params_str + ' --transform %s'%transform
    ants_cmd = 'antsApplyTransforms --'+ants_params_str
    return ants_cmd


def build_ants_registration_command(fixed_img, moving_img, output_prefix, registration_type='Rigid', image_ext='mha',
                                    fixed_mask=None, moving_mask=None, verbose=0, dim=3):
    if registration_type=='Syn':
        reg_params = [0.1, 3, 0]
        metric = 'CC'+str([fixed_img, moving_img, 1, 4])
        convergence = '[100x70x50x20,1e-8,10]'
    else:
        reg_params = [0.1]
        metric = 'MI'+str([fixed_img, moving_img, 1, 32, 'Regular', 0.25])
        convergence = '[1000x500x250x100,1e-6,10]'
    ants_params_dict = { 'verbose'        : verbose,
                         'dimensionality' : dim,
                         'output'         : [ output_prefix, output_prefix+'.'+image_ext ],
                         'interpolation'  : 'Linear',
                         'winsorize-image-intensities': [ 0.005 , 0.995 ],
                         'use-histogram-matching':       1,
                         'initial-moving-transform':     [fixed_img, moving_img, 1],
                         'transform'      : registration_type+str(reg_params),
                         'metric'         :  metric,
                         'convergence'    :  convergence,
                         'shrink-factors' : '8x4x2x1',
                         'smoothing-sigmas': '3x2x1x0vox'}
    if fixed_mask and moving_mask:
        ants_params_dict['x']='['+fixed_mask+','+moving_mask+']'
    elif fixed_mask:
        ants_params_dict['x'] = fixed_mask
    elif moving_mask:
        ants_params_dict['x'] = '[,'+moving_mask+']'
    ants_params_str = ' --'.join([' '.join([key, str(value)]) for key, value in ants_params_dict.items()])
    ants_cmd = 'antsRegistration --'+ants_params_str
    return ants_cmd


def register_ants(fixed_img, moving_img, output_prefix, path_to_transform=None, registration_type='Rigid',
                  image_ext='mha', fixed_mask=None, moving_mask=None, verbose=0, dim=3):
    print("  - Starting ANTS registration:")
    print("    - FIXED IMG : %s"%fixed_img)
    print("    - MOVING IMG: %s" % moving_img)
    print("    - OUTPUT    : %s" % output_prefix)
    ants_cmd = build_ants_registration_command(fixed_img, moving_img, output_prefix, registration_type, image_ext,
                                               fixed_mask, moving_mask, verbose, dim=dim)
    print("ANTS command: %s"%ants_cmd)
    fu.ensure_dir_exists(os.path.dirname(output_prefix))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args)
    process.wait()
    if process.returncode==0 and path_to_transform!=None:
        #-- rename trafo file
        if registration_type=='Rigid' or registration_type=='Affine':
            path_to_transform_ants = output_prefix+'0GenericAffine.mat'
            shutil.move(path_to_transform_ants, path_to_transform)
        if registration_type=='Syn':
            path_to_transform_ants = output_prefix + '1Warp.nii.gz'
            shutil.move(path_to_transform_ants, path_to_transform)
    print("Registration terminated with return code: '%s'"%process.returncode)

    return process.returncode


def register_ants_synquick(fixed_img, moving_img, output_prefix, registration='s', fixed_mask=None, dim=3):
    """
    registration:
        - r -> rigid
        - a -> rigid, affine
        - s -> rigid, affine, syn
    """
    ants_params_dict = {'d' : dim,
                        'f': fixed_img,
                        'm': moving_img,
                        't': registration,
                        'o': output_prefix,
                        'n': 4,
                        'j': 1,
                        'z': 0 }
    if fixed_mask:
        ants_params_dict['x'] = fixed_mask
    ants_params_str = ' -'.join([' '.join([key, str(value)]) for key, value in ants_params_dict.items()])
    ants_cmd = "%s -"%'antsRegistrationSyNQuick.sh' + ants_params_str
    print("ANTS command SYNquick: %s" % ants_cmd)
    fu.ensure_dir_exists(os.path.dirname(output_prefix))
    args = shlex.split(ants_cmd)
    process = subprocess.Popen(args, env=os.environ.copy())
    process.wait()
    print("ANTS terminated with return code: '%s'" % process.returncode)

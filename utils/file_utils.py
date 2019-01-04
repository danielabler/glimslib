import os



def get_file_extension(path_to_file):
    """
    Returns file extension by splitting at '.' location, or None if no '.' in file name.
    :param path_to_file: path to file to be tested
    :return: string or None
    """
    # need to test_cases for existence of '.'
    # return None of component has no file extension
    file_name = os.path.split(path_to_file)[-1]
    file_name_split = file_name.split(".")
    if file_name_split[-1] == file_name:
        # there is no '.' in file_name
        return None
    else:
        return file_name_split[-1]


def ensure_dir_exists(path):
    """
    Checks whether path exists and creates main directory if not.
    :param path: path to be tested
    """
    if get_file_extension(path) == None:
        # assume that 'path' is directory, add trailing '/'
        path = path + '/'
    if os.path.exists(os.path.dirname(path)):
        return True
    else:
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
        return False

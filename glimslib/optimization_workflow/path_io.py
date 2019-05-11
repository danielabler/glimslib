import json
import os

import grabbit as gb

import glimslib.optimization_workflow.config as config
import glimslib.utils.file_utils as fu


class PathIO:

    def __init__(self, data_root, path_to_bids_config=None):
        if path_to_bids_config:
            self.path_to_bids_config = path_to_bids_config
        else:
            self.path_to_bids_config = config.path_to_bids_config
        # -- read file to extract path patterns
        with open(self.path_to_bids_config) as json_data:
            self.bids_config = json.load(json_data)
        # -- directory to which all other dirs are relative
        self.data_root = data_root
        # -- initialize bids layout for reading
        fu.ensure_dir_exists(data_root)
        self.init_bids_layout()
        # -- attach path generator instance
        # self.path_generator = PathGenerator(self.data_root)

    def init_bids_layout(self):
        self.bids_layout = gb.Layout([(self.data_root, self.path_to_bids_config)])
        # -- the attribute 'bids_layout.path_patterns' should be populated, but is not ...
        #   we do this manually:
        self.bids_layout.path_patterns = self.bids_config.get('default_path_patterns')

    def create_path(self, path_pattern_list=None, abs_path=True, create=True, with_ext=True, **kwargs):
        if path_pattern_list:
            path = self.bids_layout.build_path(kwargs, path_pattern_list)
        else:
            path = self.bids_layout.build_path(kwargs)
        if abs_path:
            path = os.path.join(self.data_root, path)
        if create:
            fu.ensure_dir_exists(os.path.dirname(path))
        if not with_ext:
            path = '.'.join(path.split('.')[:-1])
        return path

    def create_image_path(self, processing, datasource,
                          domain='full', frame='reference', datatype='image', content='T1', extension='mha',
                          abs_path=True, create=True, **kwargs):
        path = self.create_path(processing=processing, datasource=datasource, domain=domain,
                                frame=frame, datatype=datatype, content=content, extension=extension,
                                abs_path=abs_path, create=create, **kwargs)
        return path

    def create_fenics_path(self, processing, datasource,
                           domain='full', frame='reference', datatype='fenics', content='mesh', extension='h5',
                           abs_path=True, create=True, **kwargs):
        path = self.create_path(processing=processing, datasource=datasource, domain=domain,
                                frame=frame, datatype=datatype, content=content, extension=extension,
                                abs_path=abs_path, create=create, **kwargs)
        return path

    def create_trafo_path(self, processing, datasource='registration',
                          domain=None, frame='ref2def', datatype='trafo', content='regaffine', extension='mat',
                          abs_path=True, create=True, **kwargs):
        path = self.create_path(processing=processing, datasource=datasource, domain=domain,
                                frame=frame, datatype=datatype, content=content, extension=extension,
                                abs_path=abs_path, create=create, **kwargs)
        return path

    def create_params_path(self, processing, datasource='simulation',
                           domain=None, frame=None, datatype='parameterset', content=None, extension='pkl',
                           abs_path=True, create=True, **kwargs):
        path = self.create_path(processing=processing, datasource=datasource, domain=domain,
                                frame=frame, datatype=datatype, content=content, extension=extension,
                                abs_path=abs_path, create=create, **kwargs)
        return path

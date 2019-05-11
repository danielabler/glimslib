from glimslib.optimization_workflow.path_io import PathIO
import glimslib.optimization_workflow.config as config

project_root = config.output_dir
data = PathIO(project_root)

data.create_path(processing='DomainPreparation', datasource='atlas2',
                 datatype='image', domain='full', frame='reference',
                 extension='mha', abs_path=False)


data.create_image_path(processing='DomainPreparation', datasource='patient')

data.create_fenics_path(processing='DomainPreparation', datasource='patient')

data.create_trafo_path(processing='DomainPreparation', with_ext=False)

data.create_trafo_path(processing='DomainPreparation', with_ext=False)


data.create_params_path(processing=None)


# file_name = data.create_image_path(subject='1', session='1232-23-23', modality='T1w', extension='mha')
# print(file_name)
#
# file_name = data.create_registered_image_path(subject='1', session='1232-23-23', modality='T1w', extension='mha', reg_type='affine')
# print(file_name)


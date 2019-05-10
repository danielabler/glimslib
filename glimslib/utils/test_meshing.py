import os
import glimslib.utils.meshing as meshing
import glimslib.config as config

tissues_dict = {'gray_matter': {'domain_id': 2, 'cell_size': 2},
                'global'     : {"cell_radius_edge_ratio": 2.1,
                                "cell_size": 5,
                                "facet_angle": 30.0,
                                "facet_distance": 2,
                                "facet_size": 2}
                }

path_to_image_in=os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
path_to_mesh_out = os.path.join(config.output_dir, "mesh.vtu")

path_to_xml_file = os.path.join(config.output_dir, "seed.xml")

meshing.create_mesh_xml(path_to_image_in=path_to_image_in,
                path_to_mesh_out=path_to_mesh_out,
                tissues_dict=tissues_dict,
                path_to_xml_file=path_to_xml_file)

meshing.mesh_image(path_to_meshtool_bin=config.path_to_meshtool_bin,
           path_to_meshtool_xsd=config.path_to_meshtool_xsd,
           path_to_config_file=path_to_xml_file)





import os
import glimslib.config as config
import subprocess



def mesh_image(path_to_meshtool_bin, path_to_meshtool_xsd, path_to_config_file):
    cmd_string = "%s -x %s -c %s -m 'image'" % (path_to_meshtool_bin, path_to_meshtool_xsd, path_to_config_file)
    print("Executing: '%s'"%cmd_string)
    # -- Run command
    return_code = subprocess.call(cmd_string, shell=True)
    print("Return code:", return_code)
    return return_code


def create_mesh_xml(path_to_image_in, path_to_mesh_out, tissues_dict, path_to_xml_file):
    xml_string_list = []
    xml_string_list.append("<?xml version='1.0' encoding='UTF-8'?>")
    xml_string_list.append("<MeshTool_Config xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='xxx'>")
    xml_string_list.append("<Operation_Image2Mesh path_to_input_file='%s' path_to_output_file='%s'>"%(path_to_image_in, path_to_mesh_out))
    if 'global' in tissues_dict:
        global_dict = tissues_dict['global']
        xml_string_list.append("<MeshCriteria_global cell_radius_edge_ratio='%f' cell_size='%f' facet_angle='%f' facet_distance='%f' facet_size='%f'/>"%(
                                global_dict['cell_radius_edge_ratio'], global_dict['cell_size'], global_dict['facet_angle'],
                                global_dict['facet_distance'], global_dict['facet_size']   ))
    else:
        xml_string_list.append(
            "<MeshCriteria_global cell_radius_edge_ratio='2.1' cell_size='4' facet_angle='30' facet_distance='2' facet_size='2'/>")
        print("No global meshing parameters specified, using defaults.")

    for tissue, tissue_dict in tissues_dict.items():
        if not tissue == 'global':
            xml_string_list.append("<MeshCriteria_SubDomain cell_size='%f' dimension='%i' domain_id='%i'/>"%(
                                    tissue_dict['cell_size'], 3, tissue_dict['domain_id']))
    xml_string_list.append("</Operation_Image2Mesh>")
    xml_string_list.append("</MeshTool_Config>")
    f = open(path_to_xml_file, 'w')
    s1 = '\n'.join(xml_string_list)
    f.write(s1)
    f.close()

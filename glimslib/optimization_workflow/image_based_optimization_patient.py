import os

import SimpleITK as sitk

from glimslib import config

config.USE_ADJOINT = True

from glimslib.optimization_workflow.image_based_optimization import ImageBasedOptimizationBase
import glimslib.utils.image_registration_utils as reg
from glimslib.visualisation import plotting as plott


class ImageBasedOptimizationPatient(ImageBasedOptimizationBase):

    def __init__(self, base_dir,
                 path_to_labels_atlas=None, path_to_image_atlas=None,
                 path_to_labels_patient=None, path_to_image_patient=None,
                 image_z_slice=None, plot=False):
        super().__init__(base_dir=base_dir,
                         path_to_labels_atlas=path_to_labels_atlas,
                         path_to_image_atlas=path_to_image_atlas,
                         image_z_slice=image_z_slice, plot=plot)
        if path_to_image_patient and path_to_labels_patient:
            self.path_to_image_patient_orig = path_to_image_patient
            self.path_to_labels_patient_orig = path_to_labels_patient
        else:
            self.logger.error("Patient images missing")
            self.logger.error(path_to_image_patient)
            self.logger.error(path_to_labels_patient)
        self._save_state()

    # Functions for Domain Preparation
    def _create_patient_specific_reference_from_atlas(self, path_atlas_img, path_patient_img, path_atlas_img_reg,
                                                      path_trafo, path_atlas_labels=None, path_atlas_labels_reg=None):
        """
        Affine registration to match original (3d) atlas to original (3d) patient image:
        """
        self.logger.info("path_atlas_img:        %s" % path_atlas_img)
        self.logger.info("path_patient_img:      %s" % path_patient_img)
        self.logger.info("path_atlas_img_reg:    %s" % path_atlas_img_reg)
        self.logger.info("path_trafo:            %s" % path_trafo)
        self.logger.info("path_atlas_labels:     %s" % path_atlas_labels)
        self.logger.info("path_atlas_labels_reg: %s" % path_atlas_labels_reg)

        if os.path.exists(path_atlas_img):
            reg.register_ants(fixed_img=path_patient_img,
                              moving_img=path_atlas_img,
                              output_prefix='.'.join(path_atlas_img_reg.split('.')[:-1]),
                              path_to_transform=path_trafo,
                              registration_type='Affine',
                              image_ext=path_atlas_img_reg.split('.')[-1],
                              fixed_mask=None, moving_mask=None, verbose=1, dim=3)

        if os.path.exists(path_atlas_labels) and os.path.exists(path_trafo):
            reg.ants_apply_transforms(input_img=path_atlas_labels,
                                      reference_img=path_atlas_img_reg,
                                      output_file=path_atlas_labels_reg,
                                      transforms=[path_trafo],
                                      dim=3, interpolation='GenericLabel')

    def prepare_domain(self, plot=True):

        self.path_to_atlas_img_patient_specific = self.data.create_image_path(
            processing=self.steps_sub_path_map['domain_prep'],
            datasource='registration',
            domain='full', frame='reference',
            datatype='image', content='T1',
            extension='mha')
        self.path_to_atlas_labels_patient_specific = self.data.create_image_path(
            processing=self.steps_sub_path_map['domain_prep'],
            datasource='registration',
            domain='full', frame='reference',
            datatype='image', content='labels',
            extension='mha')

        self.path_to_reg_affine = self.data.create_trafo_path(processing=self.steps_sub_path_map['domain_prep'])

        # this is done in 3d
        self._create_patient_specific_reference_from_atlas(
            path_atlas_img=self.path_to_image_atlas_orig,
            path_patient_img=self.path_to_image_patient_orig,
            path_atlas_img_reg=self.path_to_atlas_img_patient_specific,
            path_trafo=self.path_to_reg_affine,
            path_atlas_labels=self.path_to_labels_atlas_orig,
            path_atlas_labels_reg=self.path_to_atlas_labels_patient_specific)
        self.path_to_domain_image_3d = self.path_to_atlas_img_patient_specific
        self.path_to_domain_labels_3d = self.path_to_atlas_labels_patient_specific

        self.logger.info("path_to_domain_image_3d:   %s" % self.path_to_domain_image_3d)
        self.logger.info("path_to_domain_labels_3d:  %s" % self.path_to_domain_labels_3d)
        self.mesh_domain(plot=plot)

    def create_target_fields(self, plot=True, T1_label=5, T2_label=6):

        if not plot:
            plot = self.plot

        if self.dim == 2:
            # extract 2D image from original patient image
            self.path_to_image_patient_orig_2d = self.data.create_image_path(
                processing=self.steps_sub_path_map['target_fields'], datasource='domain',
                domain='reduced', frame='deformed', datatype='image', content='T1', extension='mha', dim=2)
            self.path_to_labels_patient_orig_2d = self.data.create_image_path(
                processing=self.steps_sub_path_map['target_fields'], datasource='domain',
                domain='reduced', frame='deformed', datatype='image', content='labels', extension='mha', dim=2)

            self._extract_2d_domain(path_img_3d=self.path_to_image_patient_orig,
                                    path_labels_3d=self.path_to_labels_patient_orig,
                                    path_img_2d=self.path_to_image_patient_orig_2d,
                                    path_labels_2d=self.path_to_labels_patient_orig_2d,
                                    plot_path=self.data.create_path(
                                        processing=self.steps_sub_path_map['target_fields']),
                                    plot=plot)

            path_to_deformed_img = self.path_to_image_patient_orig_2d
            path_to_deformed_labels = self.path_to_labels_patient_orig_2d

        else:
            path_to_deformed_img = self.path_to_image_patient_orig
            path_to_deformed_labels = self.path_to_labels_patient_orig

        # Deformation
        # -- Compute deformation field between patient image and patient specific atlas reference
        self.path_to_warp_field = os.path.join(self.path_target_fields,
                                               'registered_image_deformed_to_reference_warp.nii.gz')

        # deformation field is reconstructed only in the chosen dimension, i.e. in 3d if self.dim==3d and in 2d if self.dim==2d
        self._reconstruct_deformation_field(path_to_reference_image=self.path_to_domain_image_main,
                                            path_to_deformed_image=path_to_deformed_img,
                                            path_to_warp_field=self.path_to_warp_field,
                                            path_to_reduced_domain=self.path_to_domain_meshfct_main, plot=plot)

        # Apply deformation field to segmentation (and other) information, to obtain target fields in (undeformed) reference configuration
        self.path_to_img_ref_frame = self.data.create_image_path(processing=self.steps_sub_path_map['target_fields'],
                                                                 datasource='patient',
                                                                 datatype='image',
                                                                 content='T1', domain='reduced', frame='reference')
        self.path_to_labels_ref_frame = self.data.create_image_path(processing=self.steps_sub_path_map['target_fields'],
                                                                    datasource='patient',
                                                                    datatype='image',
                                                                    content='labels', domain='reduced',
                                                                    frame='reference')
        self.path_to_img_ref_frame_fct = self.data.create_fenics_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='patient',
            datatype='image',
            content='T1', domain='reduced', frame='reference')
        self.path_to_labels_ref_frame_fct = self.data.create_fenics_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='patient',
            datatype='image',
            content='labels', domain='reduced',
            frame='reference')
        self.path_to_meshfct_ref_frame = self.data.create_fenics_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='patient',
            datatype='image',
            content='mesh', domain='reduced',
            frame='reference')

        self.path_to_image_ref_frame = self.data.create_image_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='patient',
            datatype='image',
            content='T1', domain='reduced', frame='reference')

        self.path_to_labels_ref_frame = self.data.create_image_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='patient',
            datatype='image',
            content='labels', domain='reduced', frame='reference')

        self._warp_deformed_to_reference(path_to_deformed_img=path_to_deformed_img,
                                         path_to_img_ref_frame=self.path_to_image_ref_frame,
                                         path_to_deformed_labels=path_to_deformed_labels,
                                         path_to_labels_ref_frame=self.path_to_labels_ref_frame,
                                         path_to_warp_field=self.path_to_warp_field,
                                         path_to_img_ref_frame_fct=self.path_to_img_ref_frame_fct,
                                         path_to_labels_ref_frame_fct=self.path_to_labels_ref_frame_fct,
                                         path_to_meshfct_ref_frame=self.path_to_meshfct_ref_frame
                                         )

        # Concentration
        self.path_conc_from_seg = self.data.create_fenics_path(processing=self.steps_sub_path_map['target_fields'],
                                                               datasource='patient', frame='deformed', content='conc')

        # concentration fields already in undeformed reference
        self.create_conc_fields_from_segmentation(path_to_label_function=self.path_to_labels_ref_frame_fct,
                                                  path_to_conc_field_out=self.path_conc_from_seg,
                                                  plot=plot, T1_label=T1_label, T2_label=T2_label)

        self.create_thresholded_conc_fields(path_conc_field=self.path_conc_from_seg,
                                            path_to_reduced_domain=self.path_to_domain_meshfct_main, plot=plot)
        self._save_state()

    def plot_atlas_patient_from_image(self):
        # from patient
        patient_image = sitk.ReadImage(self.path_to_image_patient_orig_2d)
        patient_labels = sitk.Cast(sitk.ReadImage(self.path_to_labels_patient_orig_2d), sitk.sitkUInt8)
        # from patient in reference frame
        patient_image_ref = sitk.ReadImage(self.path_to_image_ref_frame)
        patient_labels_ref = sitk.Cast(sitk.ReadImage(self.path_to_labels_ref_frame), sitk.sitkUInt8)
        # from domain with target fields
        atlas_image = sitk.ReadImage(self.path_to_domain_image_main)
        atlas_labels = sitk.ReadImage(self.path_to_domain_labels_main)

        plott.show_img_seg_f(image=patient_image, segmentation=patient_labels,
                             path=os.path.join(self.steps_path_map['plots'], 'patient_image_tumor_labels.png'))
        plott.show_img_seg_f(image=patient_image_ref, segmentation=patient_labels_ref,
                             path=os.path.join(self.steps_path_map['plots'],
                                               'patient_image_tumor_labels_ref_frame.png'))
        plott.show_img_seg_f(image=atlas_image, segmentation=patient_labels_ref,
                             path=os.path.join(self.steps_path_map['plots'],
                                               'domain_with_tumor_labels_in_ref_frame.png'))
        plott.show_img_seg_f(image=patient_image, segmentation=atlas_labels,
                             path=os.path.join(self.steps_path_map['plots'], 'patient_image_atlas_tissue_labels.png'))

    def compare_original_optimized(self, plot=None):
        pass

    def post_process(self):
        sim_list = ['optimized']
        threshold_list = [self.conc_threshold_levels['T2'], self.conc_threshold_levels['T2']]
        super().post_process(sim_list=sim_list, threshold_list=threshold_list)


    def plot_all(self):
        super().plot_all()
        self.plot_domain_with_seed(seed_from='com')
        self.plot_atlas_patient_from_image()

#!/usr/bin/env bash
export PYTHONPATH=/opt/project:$PYTHONPATH

BASE_PATH=/opt/project
BASE_PATH_CODE=$BASE_PATH/test_cases/test_image_based_optimisation/scripts_by_step
BASE_PATH_OUT=$BASE_PATH/output/test_cases/image_based_optimisation

TIMESTEPS=50

#OUTPATH=$BASE_PATH_OUT/2017.2.0/single-core/steps-$TIMESTEPS_smaller-tolerance/
#OUTPATH=$BASE_PATH_OUT/2017.2.0/single-core/steps-$TIMESTEPS/
OUTPATH=$BASE_PATH_OUT/2018.1.0/single-core/steps-$TIMESTEPS/
export SIMOUTPATH=$OUTPATH
export SIMTIME=$TIMESTEPS

echo "==== 00_extract_data_from_image.py"
python3 $BASE_PATH_CODE/00_extract_data_from_image.py

echo "==== 01_forward_simulation.py"
python3 $BASE_PATH_CODE/01_forward_simulation.py

echo "==== 01b_forward_simulation_save_vtu.py"
python3 $BASE_PATH_CODE/01b_forward_simulation_save_vtu.py

echo "==== 02_create_deformed_image.py"
python3 $BASE_PATH_CODE/02_create_deformed_image.py

echo "==== 03_estimate_deformation_from_image.py"
python3 $BASE_PATH_CODE/03_estimate_deformation_from_image.py

echo "==== 04_estimate_forward_sim_parameters_from_image.py"
python3 $BASE_PATH_CODE/04a_estimate_forward_sim_parameters_from_image_threshold.py
python3 $BASE_PATH_CODE/04b_estimate_forward_sim_parameters_from_simulation.py
python3 $BASE_PATH_CODE/04c_estimate_forward_sim_parameters_from_simulation_reduced.py

echo "==== 05_forward_sim_optimized_parameters_from_image.py"
python3 $BASE_PATH_CODE/05a_forward_sim_optimized_parameters_from_image.py
python3 $BASE_PATH_CODE/05b_forward_sim_optimized_parameters_from_simulation.py
python3 $BASE_PATH_CODE/05c_forward_sim_optimized_parameters_from_simulation_reduced.py

echo "==== 06b_forward_sim_optimized_parameters_from_simulation_plot.py"
python3 $BASE_PATH_CODE/06a_forward_sim_optimized_parameters_image_plot.py
python3 $BASE_PATH_CODE/06b_forward_sim_optimized_parameters_simulation_plot.py
python3 $BASE_PATH_CODE/06c_forward_sim_optimized_parameters_simulation_reduced_plot.py

echo "==== 07_comparison_target_opt_plot.py"
python3 $BASE_PATH_CODE/07_comparison_target_opt_plot.py


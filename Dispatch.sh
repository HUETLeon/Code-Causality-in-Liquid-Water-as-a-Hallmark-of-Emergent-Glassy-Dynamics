#!/bin/bash

WORKDIR=$(pwd)

# Displacements

cat 02_22.4_pickles_dipoles.tar.zst_00 02_22.4_pickles_dipoles.tar.zst_01 02_22.4_pickles_dipoles.tar.zst_02 > 02_Anaysis_and_IG_estimation/22.4_water300_178K_2000B_10mcs_dipoles/pickles_dipoles.tar.zst
mv 02_22.4_pickles_imb_E10.targz  02_Anaysis_and_IG_estimation/22.4_water300_178K_2000B_10mcs_dipoles/pickles_imb_E10.targz

mv 02_22.6_pickles_d.tar.zst 02_Anaysis_and_IG_estimation/22.6_water300_178K_2000B_10mcs_distances/pickles_d.tar.zst
mv 02_22.6_pickles_imb.targz 02_Anaysis_and_IG_estimation/22.6_water300_178K_2000B_10mcs_distances/pickles_imb.targz

mv 02_22.7_pickles_imb.targz 02_Anaysis_and_IG_estimation/22.7_water300_178K_2000B_10mcs_distancesxdipoles/pickles_imb.targz

cat 03_02_pickles_dipoles_2.tar.zst_00 03_02_pickles_dipoles_2.tar.zst_01 03_02_pickles_dipoles_2.tar.zst_02 > 03_50_Sampling_and_analysis_TIP4P_300K_1B/02_50times_300K_1atm/pickles_dipoles_2.tar.zst
mv  03_02_pickles_imb_E50_without_out.targz 03_50_Sampling_and_analysis_TIP4P_300K_1B/02_50times_300K_1atm/pickles_imb_E50_without_out.targz

mv 03_04_pickles_imb.targz 03_50_Sampling_and_analysis_TIP4P_300K_1B/04_GPUMD_300K_1B/pickles_imb.targz

mv 03_06_pickles_d.tar.zst 03_50_Sampling_and_analysis_TIP4P_300K_1B/06_SDIG_50times_300K_1atm/pickles_d.tar.zst
mv 03_06_pickles_imb.targz 03_50_Sampling_and_analysis_TIP4P_300K_1B/06_SDIG_50times_300K_1atm/pickles_imb.targz

mv 03_09_pickles_imb.targz 03_50_Sampling_and_analysis_TIP4P_300K_1B/09_Distance_Dipole_cross_IG/pickles_imb.targz

# Inflates

cd 02_Anaysis_and_IG_estimation/22.4_water300_178K_2000B_10mcs_dipoles/
zstd -d pickles_dipoles.tar.zst 
tar -xf pickles_dipoles.tar
tar -xzf pickles_imb_E10.targz
cd $WORKDIR

cd 02_Anaysis_and_IG_estimation/22.6_water300_178K_2000B_10mcs_distances/
zstd -d pickles_d.tar.zst
tar -xf pickles_d.tar
tar -xzf pickles_imb.targz
cd $WORKDIR

cd 02_Anaysis_and_IG_estimation/22.7_water300_178K_2000B_10mcs_distancesxdipoles/
tar -xzf pickles_imb.targz
cd $WORKDIR

cd 03_50_Sampling_and_analysis_TIP4P_300K_1B/02_50times_300K_1atm/
zstd -d pickles_dipoles_2.tar.zst
tar -xf pickles_dipoles_2.tar
tar -xzf pickles_imb_E50_without_out.targz
cd $WORKDIR

cd 03_50_Sampling_and_analysis_TIP4P_300K_1B/04_GPUMD_300K_1B/
tar -xzf pickles_imb.targz
cd $WORKDIR

cd 03_50_Sampling_and_analysis_TIP4P_300K_1B/06_SDIG_50times_300K_1atm/
zstd -d pickles_d.tar.zst
tar -xf pickles_d.tar
tar -xzf pickles_imb.targz
cd $WORKDIR

cd 03_50_Sampling_and_analysis_TIP4P_300K_1B/09_Distance_Dipole_cross_IG/
tar -xzf pickles_imb.targz
cd $WORKDIR

echo "Done"
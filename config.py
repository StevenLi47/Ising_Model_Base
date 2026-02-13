import utils
import numpy as np

avg_Jij = utils.get_matrix('avg_Jij_no_outliers_norm')

regions = np.shape(avg_Jij)[0]
FC_1p, FC_2p, FC_3p = utils.get_matrix('avg_TS_1p'), utils.get_matrix('avg_TS_2p'), utils.get_matrix('avg_TS_3p')
avg_FCp = utils.average_matrices(FC_1p, FC_2p, FC_3p)
FC_1, FC_2, FC_3 = utils.get_matrix('avg_TS_1'), utils.get_matrix('avg_TS_2'), utils.get_matrix('avg_TS_3')
avg_FC = utils.average_matrices(FC_1, FC_2, FC_3)
FC_1pb, FC_2pb, FC_3pb = utils.get_matrix('avg_TS_1pb'), utils.get_matrix('avg_TS_2pb'), utils.get_matrix('avg_TS_3pb')
avg_FCpb = utils.average_matrices(FC_1pb, FC_2pb, FC_3pb)

ind_avg_Jij = np.mean(avg_Jij, 0)
norm_ind_avg_Jij = utils.normalize_array(ind_avg_Jij)
ind_avg_FC = np.mean(avg_FC, 0)

sort_ind_avg_FC = ind_avg_FC.copy()
sort_ind_avg_FC.sort()
sort_ind_avg_Jij = ind_avg_Jij.copy()
sort_ind_avg_Jij.sort()

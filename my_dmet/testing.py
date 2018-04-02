#import sys
#sys.path.insert (1, "/panfs/roc/groups/6/gagliard/phamx494/pyscf-1.3/pyscf/")
#import localintegrals, dmet, qcdmet_paths
#from pyscf import gto, scf, symm, future
#from pyscf import mcscf
#import numpy as np
#import HeH2_struct

import mrh.util
import numpy as np
import mrh.my_dmet.pyscf_mp2, mrh.my_dmet.pyscf_rhf

def schmidt_decompose_1RDM (the_1RDM, loc2frag, num_zero_atol=1.0e-8):
	fn_head = "schmidt_decompose_1RDM ::"
	norbs_tot = mrh.util.la.assert_matrix_square (the_1RDM)
	norbs_frag = loc2frag.shape[1]
	assert (norbs_tot >= norbs_frag and loc2frag.shape[0] == norbs_tot)
	assert (mrh.util.basis.is_basis_orthonormal (loc2frag))
	norbs_env = norbs_tot - norbs_frag
	nelec_tot = np.trace (the_1RDM)

	# We need to SVD the fragment-environment block of the 1RDM
	# The bath states are from the right-singular vectors corresponding to nonzero singular value
	loc2env = mrh.util.basis.get_complementary_states (loc2frag)
	loc2bath = mrh.util.basis.get_overlapping_states (loc2env, loc2frag, across_operator=the_1RDM)[0]
	norbs_bath = loc2bath.shape[1]
	loc2imp = np.append (loc2frag, loc2bath, axis=1)
	assert (mrh.util.basis.is_basis_orthonormal (loc2imp))

	loc2emb = mrh.util.basis.get_complete_basis (loc2imp)
	assert (mrh.util.basis.is_basis_orthonormal_and_complete (loc2emb))

	# Calculate the number of electrons in the would-be impurity model
	nelec_imp = mrh.util.basis.compute_nelec_in_subspace (the_1RDM, loc2imp)

	return loc2emb, norbs_bath, nelec_imp


def deform_embedding_basis (imp_case_str, loc2rfrag, loc2fragbanned, loc2envbanned):
	fn_head = "deforming_coeffs ({0}) ::".format (imp_case_str)
	norbs_tot = loc2rfrag.shape[0]
	norbs_frag = loc2rfrag.shape[1]
	norbs_env = norbs_tot - norbs_frag
	assert (len (loc2rfrag.shape) == 2)
	loc2fragallowed = np.eye (norbs_tot, dtype=float) if not np.any (loc2fragbanned) else mrh.util.basis.get_complementary_states (loc2fragbanned)
	loc2envallowed = np.eye (norbs_tot, dtype=float) if not np.any (loc2envbanned) else mrh.util.basis.get_complementary_states (loc2envbanned)

	# Expand the fragment states in the fragallowed basis
	loc2dfrag = mrh.util.basis.get_overlapping_states (loc2fragallowed, loc2rfrag, nlvecs=norbs_frag)[0]
	assert (loc2dfrag.shape[1] == norbs_frag)
	loc2env = mrh.util.basis.get_complementary_states (loc2dfrag)

	# Expand the environment states in the envallowed basis, then expand ~those~ in the original environment basis
	# (I.E., P_E * P_allowed * |environment>)
	loc2renv = mrh.util.basis.get_overlapping_states (loc2envallowed, loc2env)[0]
	loc2denv = mrh.util.basis.get_overlapping_states (loc2env, loc2renv)[0]
	loc2emb = np.append (loc2dfrag, loc2denv, axis=1)
	norbs_emb = loc2emb.shape[1]

	# Complete the basis if necessary
	loc2def = mrh.util.basis.get_complete_basis (loc2emb)
	assert (mrh.util.basis.is_basis_orthonormal_and_complete (loc2def))

	return loc2def, norbs_emb

def calc_mp2_ecorr_correction_to_dmetcasci_using_idempotent_1RDM (imp_case, DMET_object, idempotent_1RDM, correlated_1RDM, loc2def, norbs_frag, norbs_emb, num_zero_atol=1.0e-8):
	norbs_tot = mrh.util.la.assert_matrix_square (idempotent_1RDM)
	mrh.util.la.assert_matrix_square (correlated_1RDM, matdim=norbs_tot)
	mrh.util.la.assert_matrix_square (loc2def, matdim=norbs_tot)
	assert (norbs_tot >= norbs_frag)
	assert (norbs_tot >= norbs_emb)
	norbs_froz = norbs_tot - norbs_emb
	if imp_case == "physical frag, overlapping bath":
		imp_case = "raw"

	# Do the Schmidt decomposition. The fragment basis functions are the first norbs_frag deformed basis functions by construction.
	def2frag = np.eye (norbs_emb, dtype=idempotent_1RDM.dtype)[:,:norbs_frag]
	loc2emb  = loc2def[:,:norbs_emb]
	loc2froz = loc2def[:,norbs_emb:]
	idem_1RDM_def_basis = mrh.util.basis.represent_operator_in_subspace (idempotent_1RDM, loc2emb)
	corr_1RDM_def_basis = mrh.util.basis.represent_operator_in_subspace (correlated_1RDM, loc2emb)
	emb2dmeta_corr, norbs_bath_corr, nelec_imp_corr = schmidt_decompose_1RDM (corr_1RDM_def_basis, def2frag)
	emb2dmeta,      norbs_bath,      nelec_imp      = schmidt_decompose_1RDM (idem_1RDM_def_basis, def2frag)
	loc2dmet = np.append (np.dot (loc2emb, emb2dmeta), loc2froz, axis=1)

	# Count orbitals and arrange coefficients
	assert (mrh.util.basis.is_basis_orthonormal_and_complete (loc2dmet))
	assert (norbs_frag + norbs_bath <= norbs_emb)
	norbs_imp = norbs_frag + norbs_bath
	norbs_core = (norbs_emb - norbs_imp) + norbs_froz
	assert (norbs_imp + norbs_core == norbs_tot)
	loc2imp  = loc2dmet[:,:norbs_imp]
	loc2core = loc2dmet[:,norbs_imp:]
	norbs_imp_corr = norbs_frag + norbs_bath_corr

	# Partition up 1RDMs
	core_1RDM = mrh.util.basis.project_operator_into_subspace (idempotent_1RDM, loc2core) + (correlated_1RDM - idempotent_1RDM)
	imp_1RDM = correlated_1RDM - core_1RDM

	# Count electrons; compare results for schmidt-decomposing the whole thing to schmidt-decomposing only the idempotent 1RDM
	nelec_tot = np.trace (correlated_1RDM)
	nelec_bleed = mrh.util.basis.compute_nelec_in_subspace (core_1RDM, loc2imp)
	report_str1 = "Decomposing the correlated 1RDM in the {0} basis leads to a {1:.3f}-electron in {2} orbital impurity problem".format (imp_case, nelec_imp_corr, norbs_imp_corr)
	report_str2 = "Decomposing the idempotent 1RDM in the {0} basis leads to a {1:.3f}-electron in {2} orbital impurity problem".format (imp_case, nelec_imp, norbs_imp)
	report_str3 = report_str2 if imp_case == "raw" else report_str1 + "\n" + report_str2
	report_str4 = "in which {0} electrons from the correlated 1RDM were found bleeding on to the impurity space".format (nelec_bleed)
	print ("{0}\n{1}".format (report_str3, report_str4))
	for space, nelec in (("impurity", nelec_imp), ("total", nelec_tot)):
		err_str = "{0} number of {1} electrons not an even integer ({2})".format (imp_case, space, nelec)
		err_measure = abs (round (nelec/2) - (nelec/2))
		assert (err_measure < num_zero_atol), err_str
	nelec_imp = int (round (nelec_imp))

	# Perform the solver calculation and report the energy
	# All I want to do is read off the extra correlation energy, so I'll use pyscf_rhf and pyscf_mp2 together
	# The chemical potential shouldn't matter because this is a post-facto one-off correction, so there's no breaking the number
	# (As long as I passed the assertions a few lines above!)
	imp_OEI  = DMET_object.ints.dmet_oei  (loc2dmet, norbs_imp)
	imp_FOCK = DMET_object.ints.dmet_fock (loc2dmet, norbs_imp, core_1RDM)
	imp_TEI  = DMET_object.ints.dmet_tei  (loc2dmet, norbs_imp)
	chempot = 0.0
	DMguessRHF = DMET_object.ints.dmet_init_guess_rhf( loc2dmet, norbs_imp, nelec_imp//2, norbs_frag, chempot)
	imp_erhf, imp_rhf_1RDM = mrh.my_dmet.pyscf_rhf.solve( 0.0, imp_OEI, imp_FOCK, imp_TEI, norbs_imp, nelec_imp, norbs_frag, DMguessRHF, chempot )
	imp_emp2, imp_mp2_1RDM = mrh.my_dmet.pyscf_mp2.solve( 0.0, imp_OEI, imp_FOCK, imp_TEI, norbs_imp, nelec_imp, norbs_frag, DMguessRHF, chempot )

	return (imp_emp2 - imp_erhf)


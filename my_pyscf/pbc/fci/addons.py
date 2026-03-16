from pyscf.fci import addons

# Calling the spin-penalty function in addons file to keep it consistent with PySCF.
# I have added the contract_ss, so I hope I don't need to write the code for these
# function, rather these should work.

fix_spin = addons.fix_spin
fix_spin_ = addons.fix_spin_
SpinPenaltyFCISolver = addons.SpinPenaltyFCISolver


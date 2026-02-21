# It's time to meet the DMRG and LASSCF methods.


try:
    # DMRGSCF is a module which connects the PySCF MCSCF module to the various DMRG implementations.
    # For example: block2, block, CheMPS2, etc.
    from pyscf import dmrgscf
except ImportError:
    print("DMRGSCF module not found. \
        Please install that module: https://github.com/pyscf/dmrgscf")

try:
    # I think, block2 is the latest code, and active support is being provided for that code.
    # That's why I will tie the LASSCF to that code only.
    import block2
except ImportError:
    print("Block2 module not found. \
        Please install that module:https://github.com/block-hczhai/block2-preview.git")
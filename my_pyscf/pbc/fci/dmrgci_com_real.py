

# For the complex integrals in DMRG-CI, I need to make a subclass of
# the DMRGCI class that can handle the complex number. The block2 is really 
# good in handling the SU2 symmetry along with the complex numbers.

# The PySCF tools have FCIDUMP reader and writer but that only workds for the real-numbers.
# I need to write the complex version of the FCIDUMP reader and writer.


# DMRG-CI with complex spatial integrals for c-CASCI and c-CASSCF calculations.

import cmd
import os
from pyscf import lib
from pyscf.tools.fcidump import write_head, DEFAULT_FLOAT_FORMAT, TOL
import numpy as np                    

logger = lib.logger

try:
    from pyscf.dmrgscf import DMRGCI
    from pyscf.dmrgscf import dmrg_sym
    from pyscf.dmrgscf.dmrgci import make_schedule
    from pyscf.dmrgscf.dmrgci import block_version, check_call
except ImportError:
    raise ImportError("dmrgscf is not installed. Please install dmrgscf and block2 with USECOMPLEX=ON to use DMRGCIComplex.")


def write_hcore(fout, h, ncas, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    h = h.reshape(ncas, ncas)
    output_format = f"{float_format} {float_format}  %4d  %4d  0  0\n"
    for i in range(ncas):
        for j in range(i + 1):
            hij = h[i, j]
            if abs(hij.real) > tol: # or abs(hij.imag) > tol:
                fout.write(output_format % (hij.real, hij.imag, i + 1, j + 1))

def write_eri(fout, eri, ncas, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    eri = eri.reshape(ncas, ncas, ncas, ncas)
    output_format = f"{float_format} {float_format} %4d %4d %4d %4d\n"
    for i in range(ncas):
        for j in range(i + 1):
            for k in range(i + 1):
                lmax = (j + 1) if (k == i) else (k + 1)
                for l in range(lmax):
                    v = eri[i, j, k, l]
                    if abs(v.real) > tol: # or abs(v.imag) > tol:
                        fout.write(output_format % (v.real, v.imag, i + 1, j + 1, k + 1, l + 1))

def from_integrals(integralFile, h1e, h2e, ncas, nelec, nuc=0, ms=0, orbsym=None,
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    with open(integralFile, 'w') as fout:
        write_head(fout, ncas, nelec, ms, orbsym)
        write_eri(fout, h2e, ncas, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, ncas, tol=tol, float_format=float_format)
        output_format = f"{float_format}{float_format}  0  0  0  0\n"
        fout.write(output_format % (nuc.real, nuc.imag))


class DMRGCIComplex(DMRGCI):

    def writeDMRGConfFile(self, nelec, Restart,
                      maxIter=None, with_2pdm=True, extraline=[]):
        confFile = os.path.join(self.runtimeDir, self.configFile)

        f = open(confFile, 'w')

        if isinstance(nelec, (int, np.integer)):
            nelecb = (nelec-self.spin) // 2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec

        f.write('nelec %i\n'%(neleca+nelecb))
        f.write('spin %i\n' %(neleca-nelecb))
        f.write('use_complex\n')
        
        # I am just keeping this piece of the code:
        if self.groupname is not None:
            if isinstance(self.wfnsym, str):
                wfnsym = dmrg_sym.irrep_name2id(self.groupname, self.wfnsym)
            else:
                gpname = dmrg_sym.d2h_subgroup(self.groupname)
                assert(self.wfnsym in dmrg_sym.IRREP_MAP[gpname])
                wfnsym = self.wfnsym
            f.write('irrep %i\n' % wfnsym)

        if (not Restart):
            schedule = make_schedule(self.scheduleSweeps,
                                    self.scheduleMaxMs,
                                    self.scheduleTols,
                                    self.scheduleNoises,
                                    self.twodot_to_onedot)
            f.write('%s\n' % schedule)
        else:
            f.write('schedule\n')
            f.write('0 %6i  %8.4e  %8.4e \n' %(self.maxM, self.tol/10, 0e-6))
            f.write('end\n')
            f.write('fullrestart\n')
            f.write('onedot \n')
            if maxIter is None:
                maxIter = 8

        if self.groupname is not None:
            f.write('sym %s\n' % dmrg_sym.d2h_subgroup(self.groupname).lower())
        f.write('orbitals %s\n' % self.integralFile)
        if maxIter is None:
            maxIter = self.maxIter
        f.write('maxiter %i\n'%maxIter)
        f.write('sweep_tol %8.4e\n'%self.tol)

        f.write('outputlevel %s\n'%self.outputlevel)
        f.write('hf_occ %s\n'%self.hf_occ)
        if(with_2pdm and self.twopdm):
            f.write('twopdm\n')
        if(self.nonspinAdapted):
            f.write('nonspinAdapted\n')
        if(self.scratchDirectory):
            f.write('prefix  %s\n'%self.scratchDirectory)
        if (self.nroots !=1):
            f.write('nroots %d\n'%self.nroots)
            if (self.weights==[]):
                self.weights= [1.0/self.nroots]* self.nroots
            f.write('weights ')
            for weight in self.weights:
                f.write('%f '%weight)
            f.write('\n')

        block_extra_keyword = self.extraline + self.block_extra_keyword + extraline
        if block_version(self.executable).startswith('1.1'):
            for line in block_extra_keyword:
                if not ('num_thrds' in line or 'memory' in line):
                    f.write('%s\n'%line)
        else:
            if self.memory is not None:
                f.write('memory, %i, g\n'%(self.memory))
            if self.num_thrds > 1:
                f.write('num_thrds %d\n'%self.num_thrds)
            for line in block_extra_keyword:
                f.write('%s\n'%line)
        f.close()
        return confFile

    def writeIntegralFile(self, h1e, eri, ncas, nelec, ecore=0):
        if isinstance(nelec, (int, np.integer)):
            neleca = nelec//2 + nelec%2
            nelecb = nelec - neleca
        else :
            neleca, nelecb = nelec

        integralFile = os.path.join(self.runtimeDir, self.integralFile)
        if self.groupname is not None and self.orbsym is not []:
            # This is one last hook to avoid using orbital symmetries.
            raise NotImplementedError("Complex integrals with symmetry is not implemented yet.")

        assert h1e.shape == (ncas, ncas)
        assert eri.shape == (ncas, ncas, ncas, ncas)
        cmd = ' '.join((self.mpiprefix, "mkdir -p", self.scratchDirectory))
        check_call(cmd, shell=True)
        if not os.path.exists(self.runtimeDir):
            os.makedirs(self.runtimeDir)

        from_integrals(integralFile, h1e, eri, ncas,
                                 neleca+nelecb, ecore, ms=abs(neleca-nelecb),
                                 orbsym=self.orbsym)
        return integralFile
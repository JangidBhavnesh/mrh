from copy import deepcopy

import numpy as np
from pyscf import lib
from pyscf.csf_fci import csf_solver

from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver

from .excitations import apply_operator_string_fci


class LUSCC(LASSI):
    """Build a selected LUSCC product-state space and solve it with LASSI.

    ``las_or_lsi`` may be a LAS reference or a post-kernel LASSI reference.
    For a LASSI reference, significant product components are selected by
    ``threshold`` or ``top_m`` and every requested excitation is applied to
    each selected component.

    LUSCC is responsible only for constructing product states. Hamiltonian,
    overlap, S^2, spin projection, orthogonalization, and diagonalization are
    delegated to :class:`mrh.my_pyscf.lassi.LASSI`.
    """

    def __init__(self, las_or_lsi, a_idxs, i_idxs, state=0, threshold=0.01,
                 top_m=None, opt=1, smult_si=None, share_spectator_ci=False,
                 **kwargs):
        self.a_idxs = a_idxs
        self.i_idxs = i_idxs
        self._smult_si = smult_si
        self._share_spectator_ci = bool(share_spectator_ci)

        if isinstance(las_or_lsi, LASSI):
            self._ref_lsi = las_or_lsi
            self._lsi_state = state
            self._si_threshold = threshold
            self._top_m = top_m
            las = las_or_lsi._las
        else:
            self._ref_lsi = None
            self._lsi_state = 0
            self._si_threshold = threshold
            self._top_m = None
            las = las_or_lsi

        super().__init__(las, opt=opt, **kwargs)

    def _select_sig_indices(self):
        """Return selected product-state indices of the reference wavefunction."""
        if self._ref_lsi is None or self._ref_lsi.si is None:
            return [0]

        si = np.asarray(self._ref_lsi.si[:, self._lsi_state])
        if self._top_m is not None:
            nselect = min(self._top_m, len(si))
            return np.argsort(-np.abs(si))[:nselect].tolist()

        selected = np.where(np.abs(si) > self._si_threshold)[0].tolist()
        if not selected:
            selected = [int(np.argmax(np.abs(si)))]
        return selected

    def getAci(self, a_idx, i_idx, ref_ci=None, ref_nelecas_sub=None):
        """Apply one spin-orbital excitation to fragment CI product states."""
        frag_orbs_start = np.append(0, np.cumsum(self.ncas_sub[:-1]))
        if ref_ci is None:
            ref_ci = [frag_ci[0] for frag_ci in self.ci]
        if ref_nelecas_sub is None:
            ref_nelecas_sub = self.nelecas_sub

        # Excitations replace only active-fragment arrays. Sharing immutable
        # spectator arrays helps MRH identify equivalent product-state factors.
        aci = list(ref_ci) if self._share_spectator_ci else deepcopy(ref_ci)
        frag_ops = [[] for _ in range(self.nfrags)]
        for op_type, indices in (("ann", i_idx), ("cre", a_idx[::-1])):
            for index in indices:
                spatial = index % self.ncas
                spin = index // self.ncas
                ifrag = np.searchsorted(
                    frag_orbs_start, spatial, side="right") - 1
                local_orbital = spatial - frag_orbs_start[ifrag]
                frag_ops[ifrag].append((op_type, local_orbital, spin))

        nelecas_sub = deepcopy(list(ref_nelecas_sub))
        for ifrag, ci in enumerate(aci):
            if not frag_ops[ifrag]:
                continue
            ci, nelec = apply_operator_string_fci(
                ci, self.ncas_sub[ifrag], ref_nelecas_sub[ifrag],
                frag_ops[ifrag])
            if ci is None or not np.any(ci):
                return None, None
            ci = ci / np.linalg.norm(ci)
            aci[ifrag] = ci
            nelecas_sub[ifrag] = nelec
        return aci, nelecas_sub

    def prepare_states_(self):
        """Construct reference, excitation, and de-excitation root spaces."""
        from mrh.my_pyscf.mcscf.lasci import get_space_info
        from mrh.my_pyscf.lassi.citools import get_lroots, get_rootaddr_fragaddr

        selected = self._select_sig_indices()
        max_nroots = len(selected) * (1 + 2 * len(self.a_idxs))
        charges = np.zeros((max_nroots, self.nfrags), dtype=np.int32)
        spins = np.zeros_like(charges)
        smults = np.ones_like(charges)
        wfnsyms = np.zeros_like(charges)

        if self._ref_lsi is not None:
            ref = self._ref_lsi
            space_info = get_space_info(ref)
            nelec_frs = self.get_nelec_frs(ref)
            rootaddr, fragaddr = get_rootaddr_fragaddr(get_lroots(ref.ci))

            def ref_ci_nelec(index):
                iroot = rootaddr[index]
                ci = []
                for ifrag in range(self.nfrags):
                    ci_root = ref.ci[ifrag][iroot]
                    ci.append(ci_root[fragaddr[ifrag, index]]
                              if ci_root.ndim > 2 else ci_root)
                nelec = [tuple(nelec_frs[ifrag, iroot])
                         for ifrag in range(self.nfrags)]
                return ci, nelec, iroot
        else:
            ref = self._las
            space_info = get_space_info(ref)
            nelec_frs = self.get_nelec_frs(ref)
            rootaddr = None

            def ref_ci_nelec(index):
                ci = [self.ci[ifrag][index]
                      for ifrag in range(self.nfrags)]
                nelec = [tuple(nelec_frs[ifrag, index])
                         for ifrag in range(self.nfrags)]
                return ci, nelec, index

        ref_charges, ref_spins, ref_smults, ref_wfnsyms = space_info
        new_ci = [[] for _ in range(self.nfrags)]
        seen_rootspaces = set()
        self.nroots = 0

        # Preserve complete reference root spaces. Their local-root Cartesian
        # products are the original LASSI reference basis.
        for index in selected:
            _, _, iroot = ref_ci_nelec(index)
            if iroot in seen_rootspaces:
                continue
            seen_rootspaces.add(iroot)
            for ifrag in range(self.nfrags):
                new_ci[ifrag].append(ref.ci[ifrag][iroot])
            charges[self.nroots] = ref_charges[iroot]
            spins[self.nroots] = ref_spins[iroot]
            smults[self.nroots] = ref_smults[iroot]
            wfnsyms[self.nroots] = ref_wfnsyms[iroot]
            self.nroots += 1

        nref = self.nroots
        for a_idx, i_idx in zip(self.a_idxs, self.i_idxs):
            for index in selected:
                ref_ci, ref_nelec, _ = ref_ci_nelec(index)
                for a_op, i_op in ((a_idx, i_idx), (i_idx, a_idx)):
                    aci, nelec = self.getAci(
                        a_op, i_op, ref_ci=ref_ci,
                        ref_nelecas_sub=ref_nelec)
                    if aci is None:
                        continue
                    for ifrag, ci in enumerate(aci):
                        new_ci[ifrag].append(ci)
                        charges[self.nroots, ifrag] = (
                            self.ncas_sub[ifrag] - sum(nelec[ifrag]))
                        spins[self.nroots, ifrag] = (
                            nelec[ifrag][0] - nelec[ifrag][1])
                        # This is only nominal metadata. LASSI detects impure
                        # fragment vectors and passes smult_fr=None to opt=1.
                        smults[self.nroots, ifrag] = (
                            abs(spins[self.nroots, ifrag]) + 1)
                    self.nroots += 1

        charges = charges[:self.nroots]
        spins = spins[:self.nroots]
        smults = smults[:self.nroots]
        wfnsyms = wfnsyms[:self.nroots]
        self.ci = new_ci
        self.weights = np.zeros(self.nroots)
        self.weights[:nref] = 1.0
        self.e_states_meaningless = True

        self.fciboxes = [get_h1e_zipped_fcisolver(state_average_n_mix(
            self._las,
            [csf_solver(self._las.mol, smult=smult).set(
                charge=charge, spin=spin, wfnsym=wfnsym)
             for charge, spin, smult, wfnsym in zip(
                 charges_frag, spins_frag, smults_frag, wfnsyms_frag)],
            self.weights).fcisolver)
            for charges_frag, spins_frag, smults_frag, wfnsyms_frag in zip(
                charges.T, spins.T, smults.T, wfnsyms.T)]

        nstates = int(np.sum(np.prod(get_lroots(self.ci), axis=0)))
        lib.logger.info(
            self, "Prepared %d LUSCC product states in %d rootspaces",
            nstates, self.nroots)
        return self

    def kernel(self, **kwargs):
        """Build the LUSCC states and delegate the complete solve to LASSI."""
        self.prepare_states_()
        if self._smult_si is not None:
            kwargs.setdefault("smult_si", self._smult_si)
            kwargs.setdefault("davidson_only", True)
        return super().kernel(**kwargs)

    def filter_spaces(self, las):
        return las


# Backward-compatible public name.
LSI_LUSCC = LUSCC

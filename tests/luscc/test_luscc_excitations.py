import numpy as np

from mrh.exploratory.unitary_cc import lasuccsd


def test_h4_excitation_space_is_stable():
    uop = lasuccsd.gen_uccsd_op(4, (2, 2))
    a_idxs, i_idxs = uop.a_idxs, uop.i_idxs
    assert len(a_idxs) == len(i_idxs) == 146
    assert all(len(a) == len(i) for a, i in zip(a_idxs, i_idxs))
    assert all(sum(np.asarray(a) // 4) == sum(np.asarray(i) // 4)
               for a, i in zip(a_idxs, i_idxs))

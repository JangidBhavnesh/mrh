import numpy as np
from mrh.my_pyscf.pbc.fci.kcistrings import (
    AB_A0,
    AB_A1,
    AB_B0,
    AB_B1,
    AB_SIGN,
    AB_KA1,
    AB_KB1,
    AB_KPA,
    AB_KQA,
    AB_KRB,
    AB_PA,
    AB_QA,
    AB_RB,
    AB_SB,
    AB_KPB,
    AB_KQB,
    AB_KRA,
    AB_PB,
    AB_QB,
    AB_RA,
    AB_SA,
    SS_0,
    SS_1,
    SS_SIGN,
    SS_K1,
    SS_KP,
    SS_KQ,
    SS_KR,
    SS_P,
    SS_Q,
    SS_R,
    SS_S,
)


def contract_ab_pairs(eri, ci0_block, ci1_blocks, ab_pairs, ka, kb):
    pairtab = ab_pairs[ka][kb]

    for row in pairtab:
        a0 = row[AB_A0]
        a1 = row[AB_A1]
        b0 = row[AB_B0]
        b1 = row[AB_B1]
        sign = row[AB_SIGN]
        ka1 = row[AB_KA1]
        kb1 = row[AB_KB1]
        ci1_block = ci1_blocks[ka1][kb1]

        if ci1_block is None:
            continue

        val_ab = eri[
            row[AB_KPA], row[AB_KQA], row[AB_KRB],
            row[AB_PA], row[AB_QA], row[AB_RB], row[AB_SB],
        ]
        val_ba = eri[
            row[AB_KPB], row[AB_KQB], row[AB_KRA],
            row[AB_PB], row[AB_QB], row[AB_RA], row[AB_SA],
        ]

        ci1_block[a1, b1] += (val_ab + val_ba) * sign * ci0_block[a0, b0]


def contract_aa_pairs(eri, ci0_blocks, ci1_blocks, aa_pairs, ka, kb):
    ci0_block = ci0_blocks[ka][kb]
    if ci0_block is None:
        return

    pairtab = aa_pairs[ka]

    for row in pairtab:
        a0 = row[SS_0]
        a1 = row[SS_1]
        sign = row[SS_SIGN]
        ka1 = row[SS_K1]

        ci1_block = ci1_blocks[ka1][kb]
        if ci1_block is None:
            continue

        val = eri[
            row[SS_KP], row[SS_KQ], row[SS_KR],
            row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
        ]

        ci1_block[a1, :] += val * sign * ci0_block[a0, :]


def contract_bb_pairs(eri, ci0_blocks, ci1_blocks, bb_pairs, ka, kb):
    ci0_block = ci0_blocks[ka][kb]
    if ci0_block is None:
        return

    pairtab = bb_pairs[kb]

    for row in pairtab:
        b0 = row[SS_0]
        b1 = row[SS_1]
        sign = row[SS_SIGN]
        kb1 = row[SS_K1]

        ci1_block = ci1_blocks[ka][kb1]
        if ci1_block is None:
            continue

        val = eri[
            row[SS_KP], row[SS_KQ], row[SS_KR],
            row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
        ]

        ci1_block[:, b1] += val * sign * ci0_block[:, b0]

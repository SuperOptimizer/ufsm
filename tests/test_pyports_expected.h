/* Expected values of test_pyports (generated 2026-10-20): numpy 2.3 / scipy 1.16 percentiles and gaussian_filter, and the outputs of the former phase_score.py / check.py (archived in ~/ufsm-data/archive/py-20261020) on the inputs test_pyports writes (PYPORTS_DUMP=DIR). */
static const double exp_pct_in[11] = {3.25, -1.5, 7.125, 0.1, 22.0, 5.5, 5.5, -0.3, 13.7, 2.0, 9.9};
static const double exp_pct64[4] = {-0x1.3333333333333p-2, 0x1.6000000000000p+2, 0x1.b666666666666p+3, 0x1.595c28f5c28f4p+4};
static const float exp_pct32[4] = {-0x1.3333340000000p-2f, 0x1.6000000000000p+2f, 0x1.b666660000000p+3f, 0x1.595c280000000p+4f};
static const float exp_gauss[54] = {0x1.42b1040000000p+3f, 0x1.606e980000000p+3f, 0x1.7f380c0000000p+3f, 0x1.92153e0000000p+3f, 0x1.a0294e0000000p+3f, 0x1.a576080000000p+3f, 0x1.9dfe220000000p+3f, 0x1.9cad8c0000000p+3f, 0x1.a7355c0000000p+3f, 0x1.5cb3400000000p+3f, 0x1.7821a00000000p+3f, 0x1.96caca0000000p+3f, 0x1.a7ed760000000p+3f, 0x1.adc2260000000p+3f, 0x1.ac70ae0000000p+3f, 0x1.a907e60000000p+3f, 0x1.b11e520000000p+3f, 0x1.c1347e0000000p+3f, 0x1.94e8960000000p+3f, 0x1.a1942e0000000p+3f, 0x1.b1a2080000000p+3f, 0x1.b92fa40000000p+3f, 0x1.b676d40000000p+3f, 0x1.b23dfe0000000p+3f, 0x1.b5a9fe0000000p+3f, 0x1.c572be0000000p+3f, 0x1.d756580000000p+3f, 0x1.d9d13c0000000p+3f, 0x1.ccc1da0000000p+3f, 0x1.bf93060000000p+3f, 0x1.b723920000000p+3f, 0x1.b2ae360000000p+3f, 0x1.b556560000000p+3f, 0x1.be6ee80000000p+3f, 0x1.c8c75c0000000p+3f, 0x1.cf9c120000000p+3f, 0x1.070dbe0000000p+4f, 0x1.eb9c160000000p+3f, 0x1.c3e3a20000000p+3f, 0x1.ac8c940000000p+3f, 0x1.a9ef660000000p+3f, 0x1.b653da0000000p+3f, 0x1.c0b4de0000000p+3f, 0x1.bb238e0000000p+3f, 0x1.ae34f00000000p+3f, 0x1.130c420000000p+4f, 0x1.fa98be0000000p+3f, 0x1.c632d80000000p+3f, 0x1.a62c4e0000000p+3f, 0x1.a39db20000000p+3f, 0x1.b49fc60000000p+3f, 0x1.bda0dc0000000p+3f, 0x1.ab62a40000000p+3f, 0x1.90061c0000000p+3f};
static const long long exp_check[9] = {1, 5, 91, 3, 89, 1556, 269, 1421, 269};
static const char exp_score_plain[] =
    "wbox_w001: on-recto -0.031 (1463 px)\n"
    "wbox_w002: on-recto +0.430 (1587 px)\n"
    "wsA_r00000_c00000: on-recto -0.153 (1821 px)\n"
    "wsA_r00064_c00010: on-recto +0.842 (1648 px)\n"
    "wsB_r1_r00002_c00004: on-recto -0.242 (1552 px)\n"
    "all: on-recto +0.165 over 5 surfaces (median per surface -0.031, p10 -0.207, p90 +0.677)\n";
static const char exp_score_sheets[] =
    "wbox_w001: on-recto -0.031 (1463 px)\n"
    "wbox_w002: on-recto +0.430 (1587 px)\n"
    "wsA_r00000_c00000: on-recto -0.153 (1821 px) | one box +0.365, seam ok -0.211\n"
    "wsA_r00064_c00010: on-recto +0.842 (1648 px) | seam ok +0.885, spread +0.838, conflict +0.823\n"
    "wsB_r1_r00002_c00004: on-recto -0.242 (1552 px) | one box -0.345, seam ok -0.348, spread -0.300\n"
    "all: on-recto +0.165 over 5 surfaces (median per surface -0.031, p10 -0.207, p90 +0.677)\n"
    "  one box: +0.036 (841 px)\n"
    "  seam ok: -0.039 (1007 px)\n"
    "  spread: +0.242 (742 px)\n"
    "  conflict: +0.823 (291 px)\n";

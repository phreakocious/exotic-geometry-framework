#!/usr/bin/env python3
"""
Investigation: Protein Sequence Geometry — Folding Potential in 1D

DNA is the storage medium (investigated in dna.py), but proteins are the
functional machines. Their 1D amino acid sequence determines a 3D fold.
This investigation tests whether exotic geometries can distinguish sequences
that fold into globular structures from intrinsically disordered proteins (IDPs).

Five directions:
  D1: Detection landscape — globular vs IDP vs random (hydrophobicity encoding)
  D2: Encoding comparison — ordinal vs hydrophobicity vs molecular weight
  D3: Beyond composition — dist-matched and shuffled tests for ordering signal
  D4: Surrogate hierarchy — linear vs nonlinear via IAAFT
  D5: Helical periodicity — delay embedding at alpha-helix period (tau=4)

Budget: ~475 analyzer calls, estimated ~3 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import json
import time
import urllib.request
import urllib.error
import numpy as np
from collections import defaultdict
from scipy import stats
from exotic_geometry_framework import GeometryAnalyzer, delay_embed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

N_TRIALS = 25
DATA_SIZE = 500


# =============================================================================
# AMINO ACID DATA
# =============================================================================

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_SET = set(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydrophobicity scale
KD_HYDRO = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}
KD_MIN, KD_MAX = -4.5, 4.5

# Amino acid residue molecular weight (Da)
AA_MW = {
    'G': 57.02, 'A': 71.04, 'V': 99.07, 'L': 113.08, 'I': 113.08,
    'P': 97.05, 'F': 147.07, 'W': 186.08, 'M': 131.04, 'S': 87.03,
    'T': 101.05, 'C': 103.01, 'Y': 163.06, 'H': 137.06, 'D': 115.03,
    'E': 129.04, 'N': 114.04, 'Q': 128.06, 'K': 128.09, 'R': 156.10,
}
MW_MIN, MW_MAX = 57.02, 186.08

# AA frequencies: globular vs IDP (literature averages, Uversky et al.)
GLOB_AA_FREQ = {
    'A': 0.078, 'C': 0.017, 'D': 0.053, 'E': 0.063, 'F': 0.040,
    'G': 0.071, 'H': 0.023, 'I': 0.057, 'K': 0.059, 'L': 0.091,
    'M': 0.024, 'N': 0.043, 'P': 0.050, 'Q': 0.040, 'R': 0.051,
    'S': 0.068, 'T': 0.056, 'V': 0.065, 'W': 0.014, 'Y': 0.033,
}
IDP_AA_FREQ = {
    'A': 0.072, 'C': 0.006, 'D': 0.050, 'E': 0.093, 'F': 0.019,
    'G': 0.063, 'H': 0.022, 'I': 0.035, 'K': 0.078, 'L': 0.068,
    'M': 0.020, 'N': 0.035, 'P': 0.093, 'Q': 0.062, 'R': 0.055,
    'S': 0.088, 'T': 0.055, 'V': 0.048, 'W': 0.006, 'Y': 0.020,
}
for _fd in [GLOB_AA_FREQ, IDP_AA_FREQ]:
    _t = sum(_fd.values())
    for _k in _fd:
        _fd[_k] /= _t


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def encode_ordinal(seq):
    """Map each amino acid to evenly-spaced values in 0-255."""
    return np.array([int(AA_TO_IDX.get(c, 10) * 255 / 19) for c in seq],
                    dtype=np.uint8)


def encode_hydrophobicity(seq):
    """Map via Kyte-Doolittle hydrophobicity to 0-255."""
    vals = [int((KD_HYDRO.get(c, 0) - KD_MIN) / (KD_MAX - KD_MIN) * 255)
            for c in seq]
    return np.clip(vals, 0, 255).astype(np.uint8)


def encode_mw(seq):
    """Map via molecular weight to 0-255."""
    vals = [int((AA_MW.get(c, 110) - MW_MIN) / (MW_MAX - MW_MIN) * 255)
            for c in seq]
    return np.clip(vals, 0, 255).astype(np.uint8)


ENCODINGS = {
    'ordinal': encode_ordinal,
    'hydrophobicity': encode_hydrophobicity,
    'molecular_weight': encode_mw,
}


# =============================================================================
# PROTEIN DATA — CURATED ACCESSION LISTS
# =============================================================================

# Well-folded globular proteins (crystal structures available, diverse folds)
GLOBULAR_ACCESSIONS = [
    'P00698',  # Lysozyme C (chicken, 147 aa)
    'P00720',  # T4 lysozyme (phage, 164 aa)
    'P00918',  # Carbonic anhydrase 2 (human, 260 aa)
    'P00915',  # Carbonic anhydrase 1 (human, 261 aa)
    'P00760',  # Cationic trypsin (bovine, 247 aa)
    'P00441',  # SOD1 (human, 154 aa)
    'P62937',  # Cyclophilin A (human, 165 aa)
    'P10599',  # Thioredoxin (human, 105 aa)
    'P04406',  # GAPDH (human, 335 aa)
    'P06733',  # Alpha-enolase (human, 434 aa)
    'P14618',  # Pyruvate kinase (human, 531 aa)
    'P07195',  # LDH-B (human, 334 aa)
    'P00390',  # Glutathione reductase (human, 522 aa)
    'P0A9Q7',  # Adenylate kinase (E. coli, 214 aa)
    'P00321',  # Flavodoxin (D. vulgaris, 148 aa)
    'P00789',  # Chymotrypsin (bovine, 245 aa)
    'P02185',  # Myoglobin (sperm whale, 154 aa)
    'P69905',  # Hemoglobin alpha (human, 142 aa)
    'P68871',  # Hemoglobin beta (human, 147 aa)
    'P02768',  # Serum albumin (human, 609 aa)
    'P02753',  # Retinol-binding protein 4 (human, 201 aa)
    'P02754',  # Beta-lactoglobulin (bovine, 178 aa)
    'P02766',  # Transthyretin (human, 147 aa)
    'P00004',  # Cytochrome c (horse, 105 aa)
    'P62988',  # Ubiquitin (human, 76 aa)
    'P63165',  # SUMO-1 (human, 101 aa)
    'P42212',  # GFP (jellyfish, 238 aa)
    'P01112',  # KRAS (human, 189 aa)
    'P24941',  # CDK2 (human, 298 aa)
    'P61769',  # Beta-2-microglobulin (human, 119 aa)
    'P01009',  # Alpha-1-antitrypsin (human, 418 aa)
    'P60709',  # Actin beta (human, 375 aa)
    'P07437',  # Tubulin beta (human, 444 aa)
    'P02787',  # Transferrin (human, 698 aa)
    'P07339',  # Cathepsin D (human, 412 aa)
    'P07900',  # HSP90 alpha (human, 732 aa)
    'P11142',  # HSC70 (human, 646 aa)
    'P00352',  # ALDH1A1 (human, 501 aa)
]

# Intrinsically disordered proteins (established IDPs from DisProt/literature)
IDP_ACCESSIONS = [
    'P37840',  # Alpha-synuclein (human, 140 aa) — classic, ~100% disordered
    'P06454',  # Prothymosin alpha (human, 111 aa) — ~100% disordered
    'P38936',  # p21/CDKN1A (human, 164 aa) — ~80% disordered
    'P46527',  # p27/CDKN1B (human, 198 aa) — ~70% disordered
    'P10636',  # Tau/MAPT (human, 758 aa) — ~75% disordered
    'P01106',  # MYC (human, 439 aa) — ~70% disordered
    'P35637',  # FUS (human, 526 aa) — ~60% disordered (LCD/RGG)
    'Q92804',  # TAF15 (human, 592 aa) — FET family, LCD
    'Q13148',  # TARDBP/TDP-43 (human, 414 aa) — LCD C-terminal
    'P04637',  # p53 (human, 393 aa) — ~70% disordered (TAD+PRD+CTD)
    'P19338',  # Nucleolin (human, 710 aa) — GAR domain disordered
    'Q16665',  # HIF-1alpha (human, 826 aa) — ODD/TADs disordered
    'P09651',  # hnRNP A1 (human, 372 aa) — LCD
    'P08670',  # Vimentin (human, 466 aa) — head/tail disordered
    'Q04206',  # NF-kB p65/RELA (human, 551 aa) — TAD disordered
    'P06748',  # Nucleophosmin (human, 294 aa) — partially disordered
    'P02662',  # Alpha-S1-casein (bovine, 214 aa) — classic IDP
    'P02668',  # Kappa-casein (bovine, 190 aa) — classic IDP
    'P15502',  # Elastin/tropoelastin (human, 786 aa) — classic IDP
    'Q9UQ35',  # SRRM2 (human, 2752 aa) — RS domains, mostly disordered
    'P08047',  # SP1 (human, 785 aa) — Q-rich, mostly disordered
    'Q09472',  # EP300 (human, 2414 aa) — many disordered regions
    'P38398',  # BRCA1 (human, 1863 aa) — substantially disordered
    'P51587',  # BRCA2 (human, 3418 aa) — substantially disordered
]

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '.protein_cache.json')


# =============================================================================
# DATA LOADING
# =============================================================================

def fetch_uniprot_sequence(accession):
    """Fetch a single protein sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'exotic-geometry-framework/1.0')
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode('utf-8')
        lines = text.strip().split('\n')
        seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
        return ''.join(c for c in seq.upper() if c in AA_SET)
    except Exception as e:
        return None


def generate_synthetic_pools():
    """Fallback: synthetic sequences from known AA composition differences."""
    rng_g = np.random.RandomState(42)
    rng_i = np.random.RandomState(43)
    aas = sorted(AMINO_ACIDS)
    glob_probs = np.array([GLOB_AA_FREQ[aa] for aa in aas])
    idp_probs = np.array([IDP_AA_FREQ[aa] for aa in aas])
    pool_size = N_TRIALS * DATA_SIZE + 5000
    glob_pool = ''.join(rng_g.choice(aas, size=pool_size, p=glob_probs))
    idp_pool = ''.join(rng_i.choice(aas, size=pool_size, p=idp_probs))
    print(f"  Synthetic fallback: {pool_size} residues each (composition only, no ordering)")
    return glob_pool, idp_pool


def load_protein_data():
    """Load protein sequence pools from cache or UniProt."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            data = json.load(f)
        gp = data.get('globular_pool', '')
        ip = data.get('idp_pool', '')
        if len(gp) > 5000 and len(ip) > 5000:
            print(f"  Cached: {len(gp)} globular, {len(ip)} IDP residues")
            return gp, ip

    print("  Fetching from UniProt (one-time, will be cached)...")
    glob_seqs, idp_seqs = [], []

    for i, acc in enumerate(GLOBULAR_ACCESSIONS):
        seq = fetch_uniprot_sequence(acc)
        if seq:
            glob_seqs.append(seq)
        if (i + 1) % 10 == 0:
            print(f"    Globular: {i+1}/{len(GLOBULAR_ACCESSIONS)}", flush=True)
        time.sleep(0.1)

    for i, acc in enumerate(IDP_ACCESSIONS):
        seq = fetch_uniprot_sequence(acc)
        if seq:
            idp_seqs.append(seq)
        if (i + 1) % 10 == 0:
            print(f"    IDP: {i+1}/{len(IDP_ACCESSIONS)}", flush=True)
        time.sleep(0.1)

    gp = ''.join(glob_seqs)
    ip = ''.join(idp_seqs)

    if len(gp) < 5000 or len(ip) < 5000:
        print("  WARNING: UniProt fetch insufficient. Using synthetic fallback.")
        return generate_synthetic_pools()

    with open(CACHE_FILE, 'w') as f:
        json.dump({'globular_pool': gp, 'idp_pool': ip}, f)
    print(f"  Fetched: {len(gp)} globular ({len(glob_seqs)} proteins), "
          f"{len(ip)} IDP ({len(idp_seqs)} proteins)")
    return gp, ip


# Global pools (set in main)
GLOB_POOL = ''
IDP_POOL = ''


# =============================================================================
# WINDOW GENERATORS
# =============================================================================

def get_window(pool, trial_seed, size=DATA_SIZE):
    """Get a window of amino acids from the pool."""
    start = trial_seed * size
    if start + size <= len(pool):
        return pool[start:start + size]
    # Wrap if pool too small for non-overlapping
    start = (trial_seed * 137) % max(len(pool) - size, 1)
    return pool[start:start + size]


def generate_globular(trial_seed, encoding='hydrophobicity'):
    window = get_window(GLOB_POOL, trial_seed)
    return ENCODINGS[encoding](window)


def generate_idp(trial_seed, encoding='hydrophobicity'):
    window = get_window(IDP_POOL, trial_seed)
    return ENCODINGS[encoding](window)


def generate_random_aa(trial_seed, encoding='hydrophobicity'):
    """Random amino acid sequence (uniform over 20 AAs)."""
    rng = np.random.RandomState(trial_seed + 9000)
    seq = ''.join(rng.choice(list(AMINO_ACIDS), size=DATA_SIZE))
    return ENCODINGS[encoding](seq)


def generate_shuffled(trial_seed, pool, encoding='hydrophobicity'):
    """Same AAs as real window, shuffled order."""
    window = get_window(pool, trial_seed)
    rng = np.random.RandomState(trial_seed + 7000)
    aa_list = list(window)
    rng.shuffle(aa_list)
    return ENCODINGS[encoding](''.join(aa_list))


def generate_dist_matched(trial_seed, pool, encoding='hydrophobicity'):
    """Sample from pool's AA frequency (destroys ordering)."""
    aas = sorted(AA_SET)
    freq = defaultdict(int)
    for c in pool:
        if c in AA_SET:
            freq[c] += 1
    total = sum(freq.values())
    probs = [freq.get(aa, 1) / total for aa in aas]
    probs = np.array(probs)
    probs /= probs.sum()
    rng = np.random.RandomState(trial_seed + 8000)
    seq = ''.join(rng.choice(aas, size=DATA_SIZE, p=probs))
    return ENCODINGS[encoding](seq)


# =============================================================================
# IAAFT SURROGATE
# =============================================================================

def iaaft_surrogate(data, rng, n_iter=100):
    """IAAFT: preserves spectrum + distribution, destroys nonlinear structure."""
    data_f = data.astype(np.float64)
    n = len(data_f)
    target_amplitudes = np.abs(np.fft.rfft(data_f))
    sorted_data = np.sort(data_f)
    surrogate = data_f.copy()
    rng.shuffle(surrogate)
    for _ in range(n_iter):
        surr_fft = np.fft.rfft(surrogate)
        matched_fft = target_amplitudes * np.exp(1j * np.angle(surr_fft))
        surrogate = np.fft.irfft(matched_fft, n=n)
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_data[rank_order]
    return np.clip(surrogate, 0, 255).astype(np.uint8)


# =============================================================================
# STATISTICS
# =============================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d if np.isfinite(d) else 0.0


def count_significant(metrics_a, metrics_b, metric_names, n_total_tests):
    alpha = 0.05 / max(n_total_tests, 1)
    sig = 0
    for km in metric_names:
        va, vb = metrics_a.get(km, []), metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb, equal_var=False)
        if not np.isfinite(p):
            continue
        if abs(d) > 0.8 and p < alpha:
            sig += 1
    return sig


def per_metric_significance(metrics_a, metrics_b, metric_names, n_total_tests):
    alpha = 0.05 / max(n_total_tests, 1)
    result = {}
    for km in metric_names:
        va, vb = metrics_a.get(km, []), metrics_b.get(km, [])
        if len(va) < 2 or len(vb) < 2:
            result[km] = (0.0, 1.0, False)
            continue
        d = cohens_d(va, vb)
        _, p = stats.ttest_ind(va, vb, equal_var=False)
        if not np.isfinite(d) or not np.isfinite(p):
            result[km] = (0.0, 1.0, False)
            continue
        result[km] = (d, p, abs(d) > 0.8 and p < alpha)
    return result


def collect_metrics(analyzer, data, target_dict):
    results = analyzer.analyze(data)
    for r in results.results:
        for mn, mv in r.metrics.items():
            target_dict[f"{r.geometry_name}:{mn}"].append(mv)


# =============================================================================
# GEOMETRY FAMILY MAPPING
# =============================================================================

GEOMETRY_FAMILIES = {
    'Lattice/Discrete': ['E8 Lattice', 'Cantor (base 3)', '2-adic'],
    'Torus': ['Torus T^2', 'Clifford Torus'],
    'Curved': ['Hyperbolic (Poincaré)', 'Spherical S²'],
    'Heisenberg': ['Heisenberg (Nil)', 'Heisenberg (Nil) (centered)'],
    'Thurston': ['Sol (Thurston)', 'S² × ℝ (Thurston)',
                 'H² × ℝ (Thurston)', 'SL(2,ℝ) (Thurston)'],
    'Algebraic': ['Tropical', 'Projective ℙ²'],
    'Statistical': ['Wasserstein', 'Fisher Information', 'Persistent Homology'],
    'Physical': ['Lorentzian', 'Symplectic', 'Spiral (logarithmic)'],
    'Aperiodic': ['Penrose (Quasicrystal)', 'Ammann-Beenker (Octagonal)',
                  'Einstein (Hat Monotile)'],
    'Higher-Order': ['Higher-Order Statistics'],
}
GEOM_TO_FAMILY = {}
for _fam, _names in GEOMETRY_FAMILIES.items():
    for _n in _names:
        GEOM_TO_FAMILY[_n] = _fam

FAMILY_COLORS = {
    'Lattice/Discrete': '#E91E63', 'Torus': '#FF5722', 'Curved': '#FF9800',
    'Heisenberg': '#FFC107', 'Thurston': '#8BC34A', 'Algebraic': '#4CAF50',
    'Statistical': '#00BCD4', 'Physical': '#2196F3', 'Aperiodic': '#9C27B0',
    'Higher-Order': '#F44336',
}


def metric_to_family(metric_name):
    geom_name = metric_name.split(':')[0]
    return GEOM_TO_FAMILY.get(geom_name, 'Unknown')


# =============================================================================
# DIRECTION 1: DETECTION LANDSCAPE (100 calls)
# =============================================================================

def direction1_detection(analyzer, metric_names):
    """Globular vs IDP vs random, hydrophobicity encoding."""
    print("\n" + "=" * 78)
    print("DIRECTION 1: Detection Landscape (hydrophobicity encoding)")
    print("=" * 78)
    print(f"4 conditions x {N_TRIALS} trials = 100 calls")
    print(f"Question: are protein sequences non-random and distinguishable?\n")

    n_total = len(metric_names)
    glob_m = defaultdict(list)
    idp_m = defaultdict(list)
    rand_m = defaultdict(list)
    shuf_glob_m = defaultdict(list)

    print("  Globular (hydrophobicity)...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_globular(t), glob_m)
    print("done")

    print("  IDP (hydrophobicity)...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_idp(t), idp_m)
    print("done")

    print("  Random AA...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_random_aa(t), rand_m)
    print("done")

    print("  Shuffled globular...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_shuffled(t, GLOB_POOL), shuf_glob_m)
    print("done")

    n_glob_rand = count_significant(glob_m, rand_m, metric_names, n_total)
    n_idp_rand = count_significant(idp_m, rand_m, metric_names, n_total)
    n_glob_idp = count_significant(glob_m, idp_m, metric_names, n_total)
    n_glob_shuf = count_significant(glob_m, shuf_glob_m, metric_names, n_total)

    print(f"\n  Results:")
    print(f"    Globular vs Random:   {n_glob_rand:>3} sig")
    print(f"    IDP vs Random:        {n_idp_rand:>3} sig")
    print(f"    Globular vs IDP:      {n_glob_idp:>3} sig — ", end="")
    if n_glob_idp > 0:
        print("DISTINGUISHABLE!")
    else:
        print("not distinguishable")
    print(f"    Globular vs Shuffled: {n_glob_shuf:>3} sig (sequential ordering)")

    # AA composition comparison
    glob_aa = defaultdict(int)
    idp_aa = defaultdict(int)
    for c in get_window(GLOB_POOL, 0, min(5000, len(GLOB_POOL))):
        glob_aa[c] += 1
    for c in get_window(IDP_POOL, 0, min(5000, len(IDP_POOL))):
        idp_aa[c] += 1
    g_total = sum(glob_aa.values())
    i_total = sum(idp_aa.values())

    print(f"\n  AA composition (pool sample):")
    disorder_promoting = 'PESKQG'
    order_promoting = 'WCFIYLV'
    dp_glob = sum(glob_aa.get(c, 0) for c in disorder_promoting) / g_total
    dp_idp = sum(idp_aa.get(c, 0) for c in disorder_promoting) / i_total
    op_glob = sum(glob_aa.get(c, 0) for c in order_promoting) / g_total
    op_idp = sum(idp_aa.get(c, 0) for c in order_promoting) / i_total
    print(f"    Disorder-promoting ({disorder_promoting}): glob={dp_glob:.1%}, IDP={dp_idp:.1%}")
    print(f"    Order-promoting ({order_promoting}):    glob={op_glob:.1%}, IDP={op_idp:.1%}")

    # Top discriminating metrics (glob vs IDP)
    pms = per_metric_significance(glob_m, idp_m, metric_names, n_total)
    top_gi = [(km, d, p) for km, (d, p, sig) in pms.items() if sig]
    top_gi.sort(key=lambda x: -abs(x[1]))
    if top_gi:
        print(f"\n  Top globular-vs-IDP metrics:")
        for km, d, p in top_gi[:8]:
            print(f"    {km:<55} d={d:+8.2f}")

    return {
        'glob_m': glob_m, 'idp_m': idp_m, 'rand_m': rand_m,
        'shuf_glob_m': shuf_glob_m,
        'n_glob_rand': n_glob_rand, 'n_idp_rand': n_idp_rand,
        'n_glob_idp': n_glob_idp, 'n_glob_shuf': n_glob_shuf,
        'top_gi': top_gi,
    }


# =============================================================================
# DIRECTION 2: ENCODING COMPARISON (100 calls)
# =============================================================================

def direction2_encoding(analyzer, metric_names, d1):
    """Which encoding reveals the most structure?"""
    print("\n" + "=" * 78)
    print("DIRECTION 2: Encoding Comparison")
    print("=" * 78)
    print(f"2 new encodings x 2 classes x {N_TRIALS} = 100 calls")
    print(f"Question: which encoding best separates globular from IDP?\n")

    n_total = len(metric_names)
    enc_results = {}

    # Hydrophobicity reused from D1
    enc_results['hydrophobicity'] = d1['n_glob_idp']
    print(f"  hydrophobicity (from D1): {d1['n_glob_idp']} sig")

    for enc_name in ['ordinal', 'molecular_weight']:
        glob_m = defaultdict(list)
        idp_m = defaultdict(list)
        print(f"  {enc_name}...", end=" ", flush=True)
        for t in range(N_TRIALS):
            collect_metrics(analyzer, generate_globular(t, enc_name), glob_m)
            collect_metrics(analyzer, generate_idp(t, enc_name), idp_m)
        n_sig = count_significant(glob_m, idp_m, metric_names, n_total)
        enc_results[enc_name] = n_sig
        print(f"{n_sig} sig")

    best_enc = max(enc_results, key=enc_results.get)
    print(f"\n  Encoding ranking (glob vs IDP):")
    for enc_name in sorted(enc_results, key=enc_results.get, reverse=True):
        marker = " <-- best" if enc_name == best_enc else ""
        print(f"    {enc_name:<20} {enc_results[enc_name]:>3} sig{marker}")

    return {'enc_results': enc_results, 'best_encoding': best_enc}


# =============================================================================
# DIRECTION 3: BEYOND COMPOSITION (75 calls)
# =============================================================================

def direction3_composition(analyzer, metric_names, d1):
    """Dist-matched test: is there sequential ordering beyond AA composition?"""
    print("\n" + "=" * 78)
    print("DIRECTION 3: Beyond Composition")
    print("=" * 78)
    print(f"3 new conditions x {N_TRIALS} = 75 calls")
    print(f"Question: is there ordering information beyond AA frequencies?\n")

    n_total = len(metric_names)
    glob_m = d1['glob_m']
    idp_m = d1['idp_m']

    # Dist-matched globular
    dist_glob_m = defaultdict(list)
    print("  Dist-matched globular...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_dist_matched(t, GLOB_POOL), dist_glob_m)
    print("done")

    # Shuffled IDP
    shuf_idp_m = defaultdict(list)
    print("  Shuffled IDP...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_shuffled(t, IDP_POOL), shuf_idp_m)
    print("done")

    # Dist-matched IDP
    dist_idp_m = defaultdict(list)
    print("  Dist-matched IDP...", end=" ", flush=True)
    for t in range(N_TRIALS):
        collect_metrics(analyzer, generate_dist_matched(t, IDP_POOL), dist_idp_m)
    print("done")

    # Reuse shuffled-glob from D1
    shuf_glob_m = d1['shuf_glob_m']

    n_glob_shuf = d1['n_glob_shuf']  # from D1
    n_glob_dist = count_significant(glob_m, dist_glob_m, metric_names, n_total)
    n_idp_shuf = count_significant(idp_m, shuf_idp_m, metric_names, n_total)
    n_idp_dist = count_significant(idp_m, dist_idp_m, metric_names, n_total)

    print(f"\n  Sequential ordering tests:")
    print(f"    Globular vs shuffled:      {n_glob_shuf:>3} sig")
    print(f"    Globular vs dist-matched:  {n_glob_dist:>3} sig")
    print(f"    IDP vs shuffled:           {n_idp_shuf:>3} sig")
    print(f"    IDP vs dist-matched:       {n_idp_dist:>3} sig")

    if n_glob_shuf > 0 or n_idp_shuf > 0:
        print(f"\n  Sequential structure detected!")
        if n_glob_shuf > n_idp_shuf:
            print(f"    Globular has MORE ordering ({n_glob_shuf} vs {n_idp_shuf})")
        elif n_idp_shuf > n_glob_shuf:
            print(f"    IDP has MORE ordering ({n_idp_shuf} vs {n_glob_shuf})")
    else:
        print(f"\n  No sequential ordering detected in either class.")
        print(f"  All structure is compositional (AA frequencies).")

    return {
        'n_glob_shuf': n_glob_shuf, 'n_glob_dist': n_glob_dist,
        'n_idp_shuf': n_idp_shuf, 'n_idp_dist': n_idp_dist,
    }


# =============================================================================
# DIRECTION 4: SURROGATE HIERARCHY (50 calls)
# =============================================================================

def direction4_surrogates(analyzer, metric_names, d1):
    """Linear vs nonlinear structure via IAAFT."""
    print("\n" + "=" * 78)
    print("DIRECTION 4: Surrogate Hierarchy (IAAFT)")
    print("=" * 78)
    print(f"2 classes x {N_TRIALS} = 50 calls")
    print(f"Question: is protein sequential structure linear or nonlinear?\n")

    n_total = len(metric_names)
    glob_m = d1['glob_m']
    idp_m = d1['idp_m']

    # IAAFT surrogates for globular
    iaaft_glob_m = defaultdict(list)
    print("  IAAFT globular...", end=" ", flush=True)
    for t in range(N_TRIALS):
        data = generate_globular(t)
        rng = np.random.RandomState(t + 6000)
        surrogate = iaaft_surrogate(data, rng)
        collect_metrics(analyzer, surrogate, iaaft_glob_m)
    print("done")

    # IAAFT surrogates for IDP
    iaaft_idp_m = defaultdict(list)
    print("  IAAFT IDP...", end=" ", flush=True)
    for t in range(N_TRIALS):
        data = generate_idp(t)
        rng = np.random.RandomState(t + 6500)
        surrogate = iaaft_surrogate(data, rng)
        collect_metrics(analyzer, surrogate, iaaft_idp_m)
    print("done")

    n_glob_iaaft = count_significant(glob_m, iaaft_glob_m, metric_names, n_total)
    n_idp_iaaft = count_significant(idp_m, iaaft_idp_m, metric_names, n_total)

    print(f"\n  IAAFT results (nonlinear structure only):")
    print(f"    Globular vs IAAFT: {n_glob_iaaft:>3} sig")
    print(f"    IDP vs IAAFT:      {n_idp_iaaft:>3} sig")

    if n_glob_iaaft > 0:
        print(f"    Globular has nonlinear ordering beyond autocorrelation!")
    if n_idp_iaaft > 0:
        print(f"    IDP has nonlinear ordering beyond autocorrelation!")

    return {
        'n_glob_iaaft': n_glob_iaaft,
        'n_idp_iaaft': n_idp_iaaft,
    }


# =============================================================================
# DIRECTION 5: HELICAL PERIODICITY (150 calls)
# =============================================================================

def direction5_helix(analyzer, metric_names, d1):
    """Delay embedding at alpha-helix period. Does tau=4 amplify the signal?"""
    print("\n" + "=" * 78)
    print("DIRECTION 5: Helical Periodicity via Delay Embedding")
    print("=" * 78)
    taus = [2, 4, 7]
    print(f"tau = {taus}, {N_TRIALS} trials x 2 classes x {len(taus)} taus = {N_TRIALS*2*len(taus)} calls")
    print(f"Question: does tau=4 (alpha-helix period ~3.6) amplify the signal?\n")

    n_total = len(metric_names)
    raw_sig = d1['n_glob_idp']
    tau_results = {}

    for tau in taus:
        print(f"  tau={tau}...", end=" ", flush=True)
        glob_emb = defaultdict(list)
        idp_emb = defaultdict(list)

        for t in range(N_TRIALS):
            g_data = generate_globular(t)
            i_data = generate_idp(t)
            g_emb = delay_embed(g_data, tau)[:DATA_SIZE]
            i_emb = delay_embed(i_data, tau)[:DATA_SIZE]
            collect_metrics(analyzer, g_emb, glob_emb)
            collect_metrics(analyzer, i_emb, idp_emb)

        n_sig = count_significant(glob_emb, idp_emb, metric_names, n_total)
        tau_results[tau] = n_sig
        print(f"{n_sig} sig")

    print(f"\n  Delay embedding results (glob vs IDP):")
    print(f"    {'raw':<8} {raw_sig:>3} sig (D1 baseline)")
    for tau in taus:
        marker = ""
        if tau == 4 and tau_results[4] == max(tau_results.values()):
            marker = " <-- helix period!"
        elif tau_results[tau] == max(tau_results.values()):
            marker = " <-- peak"
        print(f"    tau={tau:<4} {tau_results[tau]:>3} sig{marker}")

    if tau_results.get(4, 0) > raw_sig:
        print(f"\n  tau=4 AMPLIFIES signal: {raw_sig} -> {tau_results[4]} (+{tau_results[4]-raw_sig})")
        print(f"  Consistent with alpha-helix periodicity (~3.6 residues/turn)")
    elif tau_results.get(4, 0) == max(tau_results.values()) and len(taus) > 1:
        print(f"\n  tau=4 is the best delay (helix period)")

    return {'tau_results': tau_results, 'raw_sig': raw_sig, 'taus': taus}


# =============================================================================
# MAIN
# =============================================================================

def main():
    global GLOB_POOL, IDP_POOL

    print("\n" + "=" * 78)
    print("INVESTIGATION: Protein Sequence Geometry — Folding Potential in 1D")
    print("=" * 78)

    print("\nLoading protein data...")
    GLOB_POOL, IDP_POOL = load_protein_data()
    min_needed = N_TRIALS * DATA_SIZE
    print(f"  Pool sizes: globular={len(GLOB_POOL)}, IDP={len(IDP_POOL)} "
          f"(need {min_needed} for {N_TRIALS} non-overlapping windows)")

    analyzer = GeometryAnalyzer().add_all_geometries()

    # Get metric names
    test_data = np.random.RandomState(0).randint(0, 256, DATA_SIZE, dtype=np.uint8)
    test_result = analyzer.analyze(test_data)
    metric_names = []
    for r in test_result.results:
        for mn in r.metrics:
            metric_names.append(f"{r.geometry_name}:{mn}")
    print(f"  Tracking {len(metric_names)} metrics across {len(test_result.results)} geometries\n")

    d1 = direction1_detection(analyzer, metric_names)
    d2 = direction2_encoding(analyzer, metric_names, d1)
    d3 = direction3_composition(analyzer, metric_names, d1)
    d4 = direction4_surrogates(analyzer, metric_names, d1)
    d5 = direction5_helix(analyzer, metric_names, d1)

    return metric_names, d1, d2, d3, d4, d5


# =============================================================================
# VISUALIZATION
# =============================================================================

def make_figure(metric_names, d1, d2, d3, d4, d5):
    print("\nGenerating figure...", flush=True)

    BG = '#181818'
    FG = '#e0e0e0'

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _dark_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # ── Panel 1: D1 — Detection landscape ──
    ax1 = fig.add_subplot(gs[0, 0])
    _dark_ax(ax1)
    labels = ['Glob vs\nRandom', 'IDP vs\nRandom', 'Glob vs\nIDP', 'Glob vs\nShuffled']
    vals = [d1['n_glob_rand'], d1['n_idp_rand'], d1['n_glob_idp'], d1['n_glob_shuf']]
    colors = ['#E91E63', '#2196F3', '#9C27B0', '#FF9800']
    ax1.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor='#333',
            width=0.6)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8, color=FG)
    ax1.set_ylabel('Significant metrics', color=FG, fontsize=9)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=11,
                 fontweight='bold')
    ax1.set_title('D1: Detection Landscape', fontsize=11, fontweight='bold',
                  color=FG)

    # ── Panel 2: D2 — Encoding comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    _dark_ax(ax2)
    enc_order = ['ordinal', 'hydrophobicity', 'molecular_weight']
    enc_labels = ['Ordinal\n(0-19)', 'Hydrophobicity\n(Kyte-Doolittle)', 'Molecular\nWeight']
    enc_vals = [d2['enc_results'].get(e, 0) for e in enc_order]
    enc_colors = ['#FF5722', '#4CAF50', '#00BCD4']
    ax2.bar(range(len(enc_order)), enc_vals, color=enc_colors, alpha=0.85,
            edgecolor='#333', width=0.55)
    ax2.set_xticks(range(len(enc_order)))
    ax2.set_xticklabels(enc_labels, fontsize=8, color=FG)
    ax2.set_ylabel('Sig metrics (Glob vs IDP)', color=FG, fontsize=9)
    for i, v in enumerate(enc_vals):
        ax2.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=11,
                 fontweight='bold')
    ax2.set_title('D2: Encoding Comparison', fontsize=11, fontweight='bold',
                  color=FG)

    # ── Panel 3: D3 — Beyond composition ──
    ax3 = fig.add_subplot(gs[0, 2])
    _dark_ax(ax3)
    comp_labels = ['Glob vs\nShuffled', 'Glob vs\nDist-match', 'IDP vs\nShuffled',
                   'IDP vs\nDist-match']
    comp_vals = [d3['n_glob_shuf'], d3['n_glob_dist'],
                 d3['n_idp_shuf'], d3['n_idp_dist']]
    comp_colors = ['#E91E63', '#E91E63', '#2196F3', '#2196F3']
    comp_alphas = [0.85, 0.55, 0.85, 0.55]
    for i, (v, c, a) in enumerate(zip(comp_vals, comp_colors, comp_alphas)):
        ax3.bar(i, v, color=c, alpha=a, edgecolor='#333', width=0.6)
    ax3.set_xticks(range(len(comp_labels)))
    ax3.set_xticklabels(comp_labels, fontsize=7, color=FG)
    ax3.set_ylabel('Significant metrics', color=FG, fontsize=9)
    for i, v in enumerate(comp_vals):
        ax3.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=10,
                 fontweight='bold')
    ax3.set_title('D3: Beyond Composition', fontsize=11, fontweight='bold',
                  color=FG)

    # ── Panel 4: D4 — Surrogate hierarchy ──
    ax4 = fig.add_subplot(gs[1, 0])
    _dark_ax(ax4)
    surr_labels = ['Glob vs\nShuffled', 'Glob vs\nIAAFT', 'IDP vs\nShuffled',
                   'IDP vs\nIAAFT']
    surr_vals = [d3['n_glob_shuf'], d4['n_glob_iaaft'],
                 d3['n_idp_shuf'], d4['n_idp_iaaft']]
    surr_colors = ['#E91E63', '#FF9800', '#2196F3', '#4CAF50']
    ax4.bar(range(len(surr_labels)), surr_vals, color=surr_colors, alpha=0.85,
            edgecolor='#333', width=0.55)
    ax4.set_xticks(range(len(surr_labels)))
    ax4.set_xticklabels(surr_labels, fontsize=7, color=FG)
    ax4.set_ylabel('Significant metrics', color=FG, fontsize=9)
    for i, v in enumerate(surr_vals):
        ax4.text(i, v + 0.3, str(v), ha='center', color=FG, fontsize=10,
                 fontweight='bold')

    # Annotate linear vs nonlinear for globular
    if d3['n_glob_shuf'] > 0:
        n_lin = d3['n_glob_shuf'] - d4['n_glob_iaaft']
        ax4.text(0.98, 0.98,
                 f"Globular:\n  Linear: ~{n_lin}\n  Nonlinear: ~{d4['n_glob_iaaft']}",
                 transform=ax4.transAxes, fontsize=7, va='top', ha='right',
                 color='#aaa', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#222',
                           edgecolor='#444'))
    ax4.set_title('D4: Surrogate Hierarchy', fontsize=11, fontweight='bold',
                  color=FG)

    # ── Panel 5: D5 — Helical periodicity ──
    ax5 = fig.add_subplot(gs[1, 1])
    _dark_ax(ax5)
    taus = d5['taus']
    tau_vals = [d5['tau_results'][t] for t in taus]
    raw_sig = d5['raw_sig']

    ax5.axhline(y=raw_sig, color='#FFB300', linestyle='--', alpha=0.7,
                label=f'Raw baseline ({raw_sig})')
    ax5.plot(range(len(taus)), tau_vals, 'o-', color='#E91E63',
             linewidth=2, markersize=10, alpha=0.85, label='Delay embedded')
    ax5.set_xticks(range(len(taus)))
    ax5.set_xticklabels([f'tau={t}' for t in taus], fontsize=9, color=FG)
    ax5.set_ylabel('Sig metrics (Glob vs IDP)', color=FG, fontsize=9)
    ax5.legend(fontsize=7, facecolor='#222', edgecolor='#444', labelcolor=FG)
    for i, v in enumerate(tau_vals):
        ax5.text(i, v + 0.5, str(v), ha='center', color=FG, fontsize=10,
                 fontweight='bold')

    # Annotate helix period
    if 4 in taus:
        idx_4 = taus.index(4)
        ax5.annotate('alpha-helix\nperiod ~3.6',
                     xy=(idx_4, tau_vals[idx_4]),
                     xytext=(idx_4 + 0.3, tau_vals[idx_4] + 3),
                     fontsize=7, color='#aaa',
                     arrowprops=dict(arrowstyle='->', color='#aaa'))

    ax5.set_title('D5: Helical Periodicity', fontsize=11, fontweight='bold',
                  color=FG)

    # ── Panel 6: Summary ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.axis('off')

    n_total = len(metric_names)
    lines = [
        "Protein Sequence Geometry — Summary",
        "",
        "D1: Detection (hydrophobicity)",
        f"  Glob vs Random:    {d1['n_glob_rand']:>3} / {n_total} sig",
        f"  IDP vs Random:     {d1['n_idp_rand']:>3} / {n_total} sig",
        f"  Glob vs IDP:       {d1['n_glob_idp']:>3} / {n_total} sig",
        f"  Glob vs Shuffled:  {d1['n_glob_shuf']:>3} / {n_total} sig",
        "",
        "D2: Best encoding",
    ]
    for e in ['ordinal', 'hydrophobicity', 'molecular_weight']:
        marker = " *" if e == d2['best_encoding'] else ""
        lines.append(f"  {e:<18} {d2['enc_results'].get(e,0):>3} sig{marker}")
    lines += [
        "",
        "D3: Beyond composition",
        f"  Glob vs shuffled:    {d3['n_glob_shuf']:>3} sig",
        f"  Glob vs dist-match:  {d3['n_glob_dist']:>3} sig",
        f"  IDP vs shuffled:     {d3['n_idp_shuf']:>3} sig",
        f"  IDP vs dist-match:   {d3['n_idp_dist']:>3} sig",
        "",
        "D4: Linear vs nonlinear",
        f"  Glob vs IAAFT: {d4['n_glob_iaaft']:>3} sig (nonlinear)",
        f"  IDP vs IAAFT:  {d4['n_idp_iaaft']:>3} sig (nonlinear)",
        "",
        "D5: Delay embedding",
        f"  Raw:   {d5['raw_sig']:>3} sig",
    ]
    for t in d5['taus']:
        lines.append(f"  tau={t}: {d5['tau_results'][t]:>3} sig")

    text = "\n".join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace', color=FG,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#222',
                       edgecolor='#444'))

    fig.suptitle('Protein Sequence Geometry: Can Folding Potential Be Detected in 1D?',
                 fontsize=15, fontweight='bold', color=FG, y=0.98)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'figures', 'proteins.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f"  Saved proteins.png")
    plt.close(fig)


if __name__ == '__main__':
    metric_names, d1, d2, d3, d4, d5 = main()
    make_figure(metric_names, d1, d2, d3, d4, d5)

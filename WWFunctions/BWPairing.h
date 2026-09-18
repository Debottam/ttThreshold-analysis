#ifndef BWPairing_H
#define BWPairing_H

// ── Standalone BW jet→W pairing discriminant ────────────────────────────────
//
// Given 4 jets, decide which of the 3 partitions into 2 di-jets is most likely
// to be the two W's — using ONLY the Breit-Wigner compatibility of the two
// di-jet masses with the W resonance. No kinematic fit, no Minuit: it is pure
// arithmetic on the raw jet 4-vectors and runs in ~microseconds.
//
// This is the discriminating core of the full WW→4q kinematic fit (where the
// per-term study showed the BW term is the ONLY part that separates pairings)
// distilled into a self-contained tool, usable as (a) a fast pairing chooser and
// (b) a WW-vs-background (e.g. ZZ→4q) χ²-like discriminant under the W hypothesis.
//
// Outputs per call:
//   pairing  — most probable partition ∈ {0,1,2}
//   gof[k]   — "goodness of fit" of partition k: −2·log[BW(m_a)·BW(m_b)] referenced
//              to the W pole (both di-jets exactly on mW), so gof ≥ 0 and gof = 0
//              means both masses sit on the pole. Lower = more W-like.
//   prob[k]  — posterior probability that partition k is the correct one, under a
//              flat prior: prob[k] = L_k / Σ_j L_j with L_k = BW(m_a)·BW(m_b).
//              The three probabilities sum to 1 by construction.
//   m_a[k], m_b[k] — the two di-jet masses of partition k (a = first pair).
//   dgof     — gof(2nd best) − gof(best): the pairing separation.
//
// Pairing index convention matches kinFit4q_bestpairing / pairing_index_from_groups:
//   0: (j1 j2)(j3 j4)   1: (j1 j3)(j2 j4)   2: (j1 j4)(j2 j3)

#include <TLorentzVector.h>
#include <cmath>
#include <limits>

namespace FCCAnalyses { namespace WWFunctions {

// PDG-ish defaults (kept independent of the Minuit-pulling WWKinReco.h so this
// header stays dependency-light). Override per call if desired.
static constexpr double BWPAIR_MW    = 80.385;
static constexpr double BWPAIR_GAMMA_W = 2.085;

static constexpr double BWPAIR_MZ    = 91.1876;
static constexpr double BWPAIR_GAMMA_Z = 2.4952;

struct BWPairingResult {
    int   pairing_WW;        // most probable partition ∈ {0,1,2}
    int   pairing_ZZ;
    float gof_WW[3];         // pole-referenced −2 log[BW_a·BW_b] (≥ 0)
    float gof_ZZ[3];
    float prob_WW[3];        // posterior over partitions, Σ = 1
    float prob_ZZ[3];
    float m_a[3], m_b[3]; // di-jet masses (a = first pair, b = second pair)
    float gof_best_WW;       // gof[pairing]
    float gof_best_ZZ; 
    float prob_best_WW;      // prob[pairing]
    float prob_best_ZZ; 
    float dgof_WW;           // gof(2nd best) − gof(best)
    float dgof_ZZ;
    float L_WW_best;      // BW_a·BW_b using W boson hypo
    float L_ZZ_best;      // BW_a·BW_b using Z boson hypo
    float L_ZZ_WW_ratio;  // L_ZZ/(L_ZZ+L_WW)
    float L_WW_ZZ_ratio;  // L_WW/(L_ZZ+L_WW)
    float Wa_best_cosTheta;
    float Wb_best_cosTheta;
    float Wa_best_dPhi;
    float Wb_best_dPhi;
    float Za_best_cosTheta;
    float Zb_best_cosTheta;
    float Za_best_dPhi;
    float Zb_best_dPhi;
    float deltaPhi_a[3], deltaPhi_b[3];
    float cosTheta_a[3], cosTheta_b[3];
    float p_a[3], p_b[3];
};

// Relativistic Breit-Wigner shape value at mass m (peak = 1/(mW·Γ) at m = mW).
inline double _bwpair_val(double m, double mW, double Gamma) {
    const double mwg = mW * Gamma;
    const double d   = m * m - mW * mW;
    return mwg / (d * d + mwg * mwg);
}

inline BWPairingResult bwPairing(const TLorentzVector& j1, const TLorentzVector& j2,
                                 const TLorentzVector& j3, const TLorentzVector& j4,
                                 double mW = BWPAIR_MW, double Gamma_W = BWPAIR_GAMMA_W,
                                 double mZ = BWPAIR_MZ, double Gamma_Z = BWPAIR_GAMMA_Z) {
    static const int order[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    const TLorentzVector* J[4] = {&j1, &j2, &j3, &j4};
    //std::cout<<"=====================mW================= "<<mW<<endl;
    //std::cout<<"=====================Gamma================= "<<Gamma_W<<endl;
    BWPairingResult R{};
    const double mwg       = mW * Gamma_W;
    const double mzg	   = mZ * Gamma_Z;
    const double pole_ref_W  = 4.0 * std::log(mwg);   // −2·2·log(1/mwg): both on pole
    const double pole_ref_Z  = 4.0 * std::log(mzg);   // −2·2·log(1/mwg): both on pole
    double L_W[3], gof_W[3], L_Z[3], gof_Z[3];
    for (int k = 0; k < 3; ++k) {
        const TLorentzVector Wa = *J[order[k][0]] + *J[order[k][1]];
        const TLorentzVector Wb = *J[order[k][2]] + *J[order[k][3]];
        const double ma = Wa.M(), mb = Wb.M();
        const double pa = Wa.P(), pb = Wb.P();
        TLorentzVector jb0,jb1,jb2,jb3;
        jb0 = *J[order[k][0]];
        jb1 = *J[order[k][1]];
        jb2 = *J[order[k][2]];
        jb3 = *J[order[k][3]];
        const double deltaPhia = jb0.DeltaPhi(jb1);
        const double deltaPhib = jb2.DeltaPhi(jb3);;
        const double cosThetaa = std::cos(jb0.Vect().Angle(jb1.Vect()));
        const double cosThetab = std::cos(jb2.Vect().Angle(jb3.Vect()));
        R.m_a[k] = static_cast<float>(ma);
        R.m_b[k] = static_cast<float>(mb);
        R.p_a[k] = static_cast<float>(pa);
        R.p_b[k] = static_cast<float>(pb);
        R.deltaPhi_a[k] = static_cast<float>(deltaPhia);
        R.deltaPhi_b[k] = static_cast<float>(deltaPhib);
        R.cosTheta_a[k] = static_cast<float>(cosThetaa);
        R.cosTheta_b[k] = static_cast<float>(cosThetab);
        const double bwa = _bwpair_val(ma, mW, Gamma_W);
        const double bwb = _bwpair_val(mb, mW, Gamma_W);
        const double bza = _bwpair_val(ma, mZ, Gamma_Z);
        const double bzb = _bwpair_val(mb, mZ, Gamma_Z);
        gof_W[k]   = -2.0 * (std::log(bwa) + std::log(bwb)) - pole_ref_W;
        L_W[k]     = bwa * bwb;
        R.gof_WW[k] = static_cast<float>(gof_W[k]);
        gof_Z[k]   = -2.0 * (std::log(bza) + std::log(bzb)) - pole_ref_Z;
        L_Z[k]     = bza * bzb;
        R.gof_ZZ[k] = static_cast<float>(gof_Z[k]);
    }

    // Posterior over partitions: softmax(−gof/2) ≡ L_k / Σ L_j (the pole_ref is a
    // common offset and cancels). Computed in the stable softmax form.
    double gmin = gof_W[0];
    for (int k = 1; k < 3; ++k) gmin = std::min(gmin, gof_W[k]);
    double w[3], wsum = 0.0;
    for (int k = 0; k < 3; ++k) { w[k] = std::exp(-0.5 * (gof_W[k] - gmin)); wsum += w[k]; }
    for (int k = 0; k < 3; ++k) R.prob_WW[k] = static_cast<float>(w[k] / wsum);

    int best = 0;
    for (int k = 1; k < 3; ++k) if (gof_W[k] < gof_W[best]) best = k;
    double second = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) if (k != best) second = std::min(second, gof_W[k]);

    R.pairing_WW   = best;
    R.gof_best_WW  = static_cast<float>(gof_W[best]);
    R.prob_best_WW = R.prob_WW[best];
    R.L_WW_best    = L_W[best];
    R.dgof_WW      = static_cast<float>(second - gof_W[best]);
    TLorentzVector jw0,jw1,jw2,jw3;
    jw0 = *J[order[best][0]];
    jw1 = *J[order[best][1]];
    jw2 = *J[order[best][2]];
    jw3 = *J[order[best][3]];
    R.Wa_best_cosTheta = std::cos(jw0.Vect().Angle(jw1.Vect()));
    R.Wb_best_cosTheta = std::cos(jw2.Vect().Angle(jw3.Vect()));
    R.Wa_best_dPhi = jw0.DeltaPhi(jw1);
    R.Wb_best_dPhi = jw2.DeltaPhi(jw3);
    
    double gmin_Z = gof_Z[0];
    for (int k = 1; k < 3; ++k) gmin_Z = std::min(gmin_Z, gof_Z[k]);
    double w_Z[3], wsum_Z = 0.0;
    for (int k = 0; k < 3; ++k) { w_Z[k] = std::exp(-0.5 * (gof_Z[k] - gmin_Z)); wsum_Z += w_Z[k]; }
    for (int k = 0; k < 3; ++k) R.prob_ZZ[k] = static_cast<float>(w_Z[k] / wsum_Z);

    int best_Z = 0;
    for (int k = 1; k < 3; ++k) if (gof_Z[k] < gof_Z[best_Z]) best_Z = k;
    double second_Z = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) if (k != best_Z) second_Z = std::min(second_Z, gof_Z[k]);

    R.pairing_ZZ   = best_Z;
    R.gof_best_ZZ  = static_cast<float>(gof_Z[best_Z]);
    R.prob_best_ZZ = R.prob_ZZ[best_Z];
    R.L_ZZ_best    = L_Z[best_Z];
    R.dgof_ZZ	   = static_cast<float>(second_Z - gof_Z[best_Z]);
    R.L_ZZ_WW_ratio = R.L_ZZ_best/(R.L_ZZ_best+R.L_WW_best);
    R.L_WW_ZZ_ratio = R.L_WW_best/(R.L_ZZ_best+R.L_WW_best);
    TLorentzVector jz0,jz1,jz2,jz3;
    jz0 = *J[order[best_Z][0]];
    jz1 = *J[order[best_Z][1]];
    jz2 = *J[order[best_Z][2]];
    jz3 = *J[order[best_Z][3]];
    R.Za_best_cosTheta = std::cos(jz0.Vect().Angle(jz1.Vect()));
    R.Zb_best_cosTheta = std::cos(jz2.Vect().Angle(jz3.Vect()));
    R.Za_best_dPhi = jz0.DeltaPhi(jz1);
    R.Zb_best_dPhi = jz2.DeltaPhi(jz3);

    return R;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif

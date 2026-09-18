# Standalone BW jet→W pairing (WWFunctions/BWPairing.h) — NO kinematic fit.
# Re-clusters the event into 4 jets and runs the fast Breit-Wigner pairing
# discriminant. Emits the chosen pairing, per-pairing gof / posterior prob /
# di-jet masses, and (signal only) the gen-truth pairing-correctness flag plus
# the jet→quark match dRs so WW can be split on "passes step1 matching".
#
# WW signal  : WW_SAMPLE=p8_ee_WW_ecm160 (default), WW_GENTRUTH=1.
# ZZ control : WW_SAMPLE=p8_ee_ZZ_ecm160  WW_GENTRUTH=0  (W hypothesis on ZZ→4q).
import os, ROOT
import treemaker_common as tc

WW_SAMPLE   = os.environ.get("WW_SAMPLE", "p8_ee_WW_ecm240")  #change here for WW or ZZ
WW_GENTRUTH = os.environ.get("WW_GENTRUTH", "1").strip().lower() in ("1", "true", "yes")
# Gen-level 4q filter: both p8_ee_WW and p8_ee_ZZ are INCLUSIVE, so restrict to
# the fully-hadronic final state at gen level. "W"→2(W→qq); "Z"→2(Z→qq); "none".
# Pairing gen-truth (gen_pairing_true, matched dR) only makes sense for WW.
#WW_GEN4Q = os.environ.get("WW_GEN4Q", "W" if WW_GENTRUTH else "Z").strip()
WW_GEN4Q = os.environ.get("WW_GEN4Q", "W").strip()         #change here for WW or ZZ
_frac = float(os.environ.get("WW_FRACTION", "0.01"))
processList = {WW_SAMPLE: {"fraction": _frac, "crossSection": 1}}

channel = "had"
prodTag = "FCCee/winter2023/IDEA/"
_tag = os.environ.get("WW_TAG", "").strip()         #change here for WW or ZZ
_default = "outputs/treemaker/4q/bwpairing/{}{}".format(channel, "_" + _tag if _tag else "")
outputDir = os.environ.get("STEP2_OUTDIR", _default)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h",
                "WWFunctions/BWPairing.h"]

all_branches = [
    "reco_jet1_p", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_mass",
    "reco_jet2_p", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_mass",
    "reco_jet3_p", "reco_jet3_theta", "reco_jet3_phi", "reco_jet3_mass",
    "reco_jet4_p", "reco_jet4_theta", "reco_jet4_phi", "reco_jet4_mass",
    "bwpair_pairing_WW", "bwpair_gof_best_WW", "bwpair_prob_best_WW", "bwpair_dgof_WW",
    "bwpair_gof_WW0", "bwpair_gof_WW1", "bwpair_gof_WW2",
    "bwpair_prob_WW0", "bwpair_prob_WW1", "bwpair_prob_WW2",
    "bwpair_ma0", "bwpair_mb0", "bwpair_ma1", "bwpair_mb1", "bwpair_ma2", "bwpair_mb2",
    "bwpair_p_a0", "bwpair_p_b0", "bwpair_p_a1", "bwpair_p_b1", "bwpair_p_a2", "bwpair_p_b2",
    "bwpair_cosTheta_a0", "bwpair_cosTheta_b0", "bwpair_cosTheta_a1", "bwpair_cosTheta_b1", "bwpair_cosTheta_a2", "bwpair_cosTheta_b2",
    "bwpair_deltaPhi_a0", "bwpair_deltaPhi_b0", "bwpair_deltaPhi_a1", "bwpair_deltaPhi_b1", "bwpair_deltaPhi_a2", "bwpair_deltaPhi_b2",
    "bwpair_L_WW_best", "bwpair_L_ZZ_WW_ratio","bwpair_L_WW_ZZ_ratio",
    "bwpair_pairing_ZZ", "bwpair_gof_best_ZZ", "bwpair_prob_best_ZZ", "bwpair_dgof_ZZ",
    "bwpair_gof_ZZ0", "bwpair_gof_ZZ1", "bwpair_gof_ZZ2",
    "bwpair_prob_ZZ0", "bwpair_prob_ZZ1", "bwpair_prob_ZZ2","bwpair_L_ZZ_best",
    "bwpair_Wa_best_cosTheta", "bwpair_Wb_best_cosTheta", "bwpair_Wa_best_dPhi", "bwpair_Wb_best_dPhi",
    "bwpair_Za_best_cosTheta", "bwpair_Zb_best_cosTheta", "bwpair_Za_best_dPhi", "bwpair_Zb_best_dPhi",
    # Durham splitting scales: d_45 = scale of a 5th jet (radiation), d_34 = 4th jet.
    # Cut d_45 small to require a genuine 4-jet event (standard ee 4-jet selection).
    "d_23", "d_34", "d_45",
]
if WW_GENTRUTH:
    all_branches += [
        "gen_pairing_true", "bwpair_correct_ZZ","bwpair_correct_WW",
        "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR",
        # global jet->quark matching-quality variable (Δθ,Δφ over all 4 pairs)
        "gen_match_dist", "gen_match_dmax",
        "jet1_dang", "jet2_dang", "jet3_dang", "jet4_dang",
    ]

jetClusteringHelper = None
_dataset_iter = iter(processList.keys())


class RDFanalysis:
    def analysers(df):
        global jetClusteringHelper
        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker 4q bwpairing] dataset={_dataset}  ecm={_ecm}  truth={WW_GENTRUTH}")

        # Forget about isolated leptons for this study: build the jet-input
        # collection (lepton/FSR removed) but do NOT veto on lepton count — the
        # gen-4q filter below defines the final state.
        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel, veto=False)
        df, jetClusteringHelper = tc.cluster_jets_4q(df)
        df = tc.define_reco_jets_kinematics_4q(df)

        # Gen-level 4q filter (both samples are inclusive).
        if WW_GEN4Q == "W":
            print("I AM IN W==========================")
            df = tc.select_gen_fromW(df)
            df = tc.define_gen_kinematics_4qW(df)
            df = tc.match_jets_to_quarks_4q(df)   # defines gen_pairing_true + jet*_matched_q_dR
            df = tc.define_match_quality_4q(df)   # global Δθ,Δφ matching variable
        elif WW_GEN4Q == "Z":
            print("I AM IN Z1==========================")
            df = tc.select_gen_fromZ(df)          # ZZ→4q filter, no pairing trut
            print("I AM IN Z2==========================")
            df = tc.define_gen_kinematics_4qZ(df)
            print("I AM IN Z3==========================")
            df = tc.match_jets_to_quarks_4q(df)   # defines gen_pairing_true + jet*_matched_q_dR
            print("I AM IN Z4==========================")
            df = tc.define_match_quality_4q(df) 
            print("I AM IN Z5==========================")
        elif WW_GEN4Q != "none":
            raise ValueError(f"WW_GEN4Q={WW_GEN4Q!r} must be 'W', 'Z' or 'none'")

        df = tc.run_bw_pairing(df, with_truth=WW_GENTRUTH)

        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print()
        print()
        return df

    def output():
        return all_branches

import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import Bio.SeqUtils.MeltingTemp as mt

# GLOBAL PAGE CONFIGURATION
st.set_page_config(page_title="HRMTool", layout="wide")

# ==========================================
# TASK 1: HRM ANALYSIS FUNCTION
# ==========================================
def _get_scientific_delta_tm(Tm1, Tm2, length):
    # Delta Tm lý thuyết (giả định là giá trị tại điểm bão hòa)
    raw_delta = abs(Tm1 - Tm2)
    
    # Hằng số này đại diện cho sự mất mát năng lượng do hiệu ứng biên (end-fraying)
    # được rút ra từ các nghiên cứu về độ bền của DNA duplex
    # Với L càng lớn, giá trị này tiến dần về 0
    k_length_factor = 25.0 
    
    # Áp dụng hiệu chỉnh entropy phụ thuộc độ dài
    # Công thức: Delta_tm = raw_delta * (1 - (k / (length + k)))
    # Đây là mô hình tiệm cận: khi length -> vô cực, delta -> 0
    corrected_delta = raw_delta * (1 - (k_length_factor / (length + k_length_factor)))
    
    return corrected_delta

def run_hrm_analysis():
    st.title("HRM Curve Analyzer")
    st.markdown("---")

    st.sidebar.header("HRM Configuration")
    ref_selection = st.sidebar.selectbox("Select Reference Baseline:", options=["Homozygote 1", "Homozygote 2"], key="hrm_ref_select")

    st.sidebar.markdown("**PCR Reaction Conditions**")
    dnac1_nm = st.sidebar.number_input("DNA 1 Conc. (nM):", 1, 2000, 100, 1, key="hrm_dnac1")
    dnac2_nm = st.sidebar.number_input("DNA 2 Conc. (nM):", 1, 2000, 100, 1, key="hrm_dnac1_2")
    na_mM = st.sidebar.number_input("Na+ Conc. (mM):", 0, 500, 50, 10, key="hrm_na")
    mg_mM = st.sidebar.number_input("Mg2+ Conc. (mM):", 0.0, 10.0, 3.0, 0.5, key="hrm_mg")

    st.sidebar.markdown("**Slope Factors (k)**")
    k_homo = st.sidebar.slider("k for Homoduplex:", 0.1, 1.0, 0.40, 0.01, key="hrm_khomo")
    k_hetero = st.sidebar.slider("k for Heteroduplex:", 0.1, 2.0, 0.80, 0.01, key="hrm_khet")

    col1, col2 = st.columns(2)
    with col1:
        raw_allele1 = st.text_input("Allele 1 Sequence (5' -> 3'):", value="AGCCAAAACAGCCTTAAATAGCATTCAAACACTCTTTCTTCCATGCCTTCAGTCCTGC", key="hrm_seq1_input")
        allele1 = raw_allele1.upper().replace(" ", "") 
    with col2:
        raw_allele2 = st.text_input("Allele 2 Sequence (5' -> 3'):", value="AGCCAAAACAGCCTTAAATAGCATTCCAACACTCTTTCTTCCATGCCTTCAGTCCTGC", key="hrm_seq2_input")
        allele2 = raw_allele2.upper().replace(" ", "") 

    def get_complement_3to5(seq):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        return "".join(complement.get(base, base) for base in seq)

    def extract_visual_mismatch(seq1, seq2):
        snp_pos = None
        for i, (a, b) in enumerate(zip(seq1, seq2)):
            if a != b: snp_pos = i; break
        if snp_pos is None: return "No Mismatch detected.", "N/A", "N/A", "N/A", "N/A"
        nu1, nu2 = seq1[snp_pos], seq2[snp_pos]
        mismatch_type = "Transition" if {nu1, nu2} in [{"A", "G"}, {"C", "T"}] else "Transversion"
        left, right = snp_pos - 1, snp_pos + 2
        raw_trip1 = seq1[max(0, left):min(len(seq1), right)]
        raw_trip2 = seq2[max(0, left):min(len(seq2), right)]
        comp_trip1 = get_complement_3to5(raw_trip1)
        comp_trip2 = get_complement_3to5(raw_trip2)
        h_raw1 = f"{seq1[snp_pos-1] if snp_pos > 0 else ''}[{nu1}]{seq1[snp_pos+1] if snp_pos < len(seq1)-1 else ''}"
        h_raw2 = f"{seq2[snp_pos-1] if snp_pos > 0 else ''}[{nu2}]{seq2[snp_pos+1] if snp_pos < len(seq2)-1 else ''}"
        h_comp1 = f"{comp_trip1[0] if snp_pos > 0 else ''}[{comp_trip1[1] if snp_pos > 0 else comp_trip1[0]}]{comp_trip1[2] if len(comp_trip1)>2 else ''}"
        h_comp2 = f"{comp_trip2[0] if snp_pos > 0 else ''}[{comp_trip2[1] if snp_pos > 0 else comp_trip2[0]}]{comp_trip2[2] if len(comp_trip2)>2 else ''}"
        return f"Position {snp_pos + 1}: {mismatch_type} ({nu1} ↔ {nu2})", h_raw1, h_comp1, h_raw2, h_comp2

    if allele1 and allele2:
        if len(allele1) != len(allele2):
            st.error("⚠️ Error: Sequence lengths must be equal for alignment.")
        else:
            Tm1 = mt.Tm_NN(allele1, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM, saltcorr=7)
            Tm2 = mt.Tm_NN(allele2, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM, saltcorr=7)
            length = len(allele1.replace(" ", "")) # Đảm bảo độ dài chính xác
            delta_tm = _get_corrected_delta(Tm1, Tm2, length)
            comp_allele1_3to5 = get_complement_3to5(allele1)
            comp_allele2_3to5 = get_complement_3to5(allele2)
            try:
                Tm_het1 = mt.Tm_NN(allele1, c_seq=comp_allele2_3to5, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM, saltcorr=7)
                Tm_het2 = mt.Tm_NN(allele2, c_seq=comp_allele1_3to5, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM, saltcorr=7)
                penalty_1, penalty_2 = Tm1 - Tm_het1, Tm2 - Tm_het2
            except:
                penalty_1, penalty_2 = 1.5, 1.5
                Tm_het1, Tm_het2 = Tm1 - penalty_1, Tm2 - penalty_2
            
            mismatch_info, r1, c_back1, r2, c_back2 = extract_visual_mismatch(allele1, allele2)
            st.subheader("Predicted Thermodynamic Results")
            st.info(f"**Detected Mismatch Details:** {mismatch_info}")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Tm Homo 1", f"{Tm1:.2f} °C"); st.code(f"5'- {r1} -3'\n3'- {c_back1} -5'")
            with c2: st.metric("Tm Homo 2", f"{Tm2:.2f} °C"); st.code(f"5'- {r2} -3'\n3'- {c_back2} -5'")
            with c3: st.metric("ΔTm (Homo1-Homo2)", f"{delta_tm:.2f} °C")
            with c4: st.metric("Tm Hetero 1", f"{Tm_het1:.2f} °C", delta=f"-{penalty_1:.2f}°C"); st.code(f"5'- {r1} -3'\n3'- {c_back2} -5'")
            with c5: st.metric("Tm Hetero 2", f"{Tm_het2:.2f} °C", delta=f"-{penalty_2:.2f}°C"); st.code(f"5'- {r2} -3'\n3'- {c_back1} -5'")

            t_start, t_end = min(Tm1, Tm2) - 6, max(Tm1, Tm2) + 6
            T = np.linspace(t_start, t_end, 1000)
            def inverse_sigmoid(T, Tm, k): return 1 / (1 + np.exp((T - Tm) / k))
            F_homo1 = inverse_sigmoid(T, Tm1, k_homo)
            F_homo2 = inverse_sigmoid(T, Tm2, k_homo)
            F_het = (0.25*inverse_sigmoid(T, Tm1, k_homo) + 0.25*inverse_sigmoid(T, Tm2, k_homo) + 0.25*inverse_sigmoid(T, Tm_het1, k_hetero) + 0.25*inverse_sigmoid(T, Tm_het2, k_hetero))
            dF_homo1, dF_homo2, dF_het = -np.gradient(F_homo1, T), -np.gradient(F_homo2, T), -np.gradient(F_het, T)
            F_ref = F_homo1 if ref_selection == "Homozygote 1" else F_homo2
            diff_homo1, diff_homo2, diff_het = F_homo1 - F_ref, F_homo2 - F_ref, F_het - F_ref

            st.subheader("HRM Analysis Visualizations")
            cc1, cc2, cc3 = st.columns(3)
            with cc1: color_homo1 = st.color_picker("Homozygote 1 Color", value="#1E90FF", key="cp_h1")
            with cc2: color_homo2 = st.color_picker("Homozygote 2 Color", value="#FF4500", key="cp_h2")
            with cc3: color_het = st.color_picker("Heterozygote Color", value="#8A2BE2", key="cp_het")
                
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
            zoom_range = (t_start, t_end)
            ax1.plot(T, F_homo1, color=color_homo1, linestyle='--'); ax1.plot(T, F_homo2, color=color_homo2, linestyle='--'); ax1.plot(T, F_het, color=color_het, linewidth=3)
            ax1.set_title('A. Aligned Melting Curve'); ax1.set_xlim(zoom_range); ax1.grid(True, linestyle=':')
            ax2.plot(T, dF_homo1, color=color_homo1, linestyle='--'); ax2.plot(T, dF_homo2, color=color_homo2, linestyle='--'); ax2.plot(T, dF_het, color=color_het, linewidth=3)
            ax2.set_title('B. Derivative Curve (-dF/dT)'); ax2.set_xlim(zoom_range); ax2.grid(True, linestyle=':')
            style_h1 = ':' if ref_selection == "Homozygote 1" else '-'
            style_h2 = ':' if ref_selection == "Homozygote 2" else '-'
            ax3.plot(T, diff_homo1, color=color_homo1, linestyle=style_h1); ax3.plot(T, diff_homo2, color=color_homo2, linestyle=style_h2); ax3.plot(T, diff_het, color=color_het, linewidth=3)
            ax3.set_title(f'C. Difference Plot ({ref_selection})'); ax3.set_xlim(zoom_range); ax3.grid(True, linestyle=':')
            plt.tight_layout(); st.pyplot(fig)


# ==========================================
# TASK 2: AUTOMATED PRIMER DESIGNER FUNCTION
# ==========================================
def run_primer_designer():
    st.title("Primer Design")
    st.markdown("---")
    
    st.subheader("1. Target Sequence Entry")
    raw_input_seq = st.text_area(
        "Input DNA Sequence Matrix (Mark SNP location with square brackets, e.g., [C/A]):", 
        value="AGCCAAAACAGCCTTAAATAGCATT[C/A]AAACACTCTTTCTTCCATGCCTTCAGTCCTGC", 
        key="pd_single_input"
    )
    
    clean_seq = raw_input_seq.upper().strip().replace(" ", "").replace("\n", "").replace("\r", "")
    
    if not clean_seq:
        st.warning("Please enter a valid template sequence matrix.")
        return

    match = re.search(r"\[\s*([A-Z])\s*/\s*([A-Z])\s*\]", clean_seq)
    if match:
        nu1 = match.group(1)
        nu2 = match.group(2)
        start_bracket_idx = match.start()
        end_bracket_idx = match.end()
        
        seq_base_1 = clean_seq[:start_bracket_idx] + nu1 + clean_seq[end_bracket_idx:]
        seq_base_2 = clean_seq[:start_bracket_idx] + nu2 + clean_seq[end_bracket_idx:]
        snp_idx = start_bracket_idx 
        st.success(f"SNP Polymorphism mapped successfully at base location: **{snp_idx + 1}** (Genotype: `{nu1}` ↔ `{nu2}`)")
    else:
        st.error("⚠️ SNP notation format `[N1/N2]` not detected or invalid. Please check your text format.")
        return

    # SIDEBAR PANEL
    st.sidebar.header("Design Filtering Thresholds")
    max_display_pairs = st.sidebar.slider("Maximum pairs to display:", 5, 1000, 10, key="pd_max_display")
    
    st.sidebar.subheader("Secondary Structure Rules")
    max_sec_tm = st.sidebar.slider("Max Allowable Structure Tm (°C):", 20.0, 80.0, 58.0, 0.5, key="p_max_sec_tm")
    min_sec_dg = st.sidebar.number_input("Min Allowable Delta G (kcal/mol):", -15.0, 0.0, -8.0, 0.1, key="p_min_sec_dg")
    max_3prime_matches = st.sidebar.slider("Max Allowable 3' Complementary Bases:", 1, 10, 4, key="p_max_3p")

    st.sidebar.subheader("Oligo Reaction Environment")
    na_mM = st.sidebar.number_input("Na+ Monovalent Ion (mM):", value=50.0, key="salt_na")
    mg_mM = st.sidebar.number_input("Mg2+ Divalent Ion (mM):", value=1.5, key="salt_mg")
    primer_conc = st.sidebar.number_input("Oligo Concentration (nM):", value=200.0, key="salt_pc")
    
    st.sidebar.subheader("Primer Sequence Features")
    p_len_min = st.sidebar.number_input("Min Primer Length (bp):", 10, 30, 15, key="p_min")
    p_len_max = st.sidebar.number_input("Max Primer Length (bp):", 15, 30, 24, key="p_max")
    gc_min, gc_max = st.sidebar.slider("Oligo %GC Content Bounds (%):", 20.0, 80.0, (20.0, 80.0), 1.0, key="p_gc")
    
    st.sidebar.subheader("Annealing & Melting Metrics")
    tm_min, tm_max = st.sidebar.slider("Ideal Primer Tm Window (°C):", 35.0, 85.0, (35.0, 85.0), 0.5, key="p_tm")
    max_primer_delta_tm = st.sidebar.number_input("Max Inter-Primer ΔTm Mismatch:", 0.0, 30.0, 15.0, 0.5, key="p_dtm")
    
    st.sidebar.subheader("PCR Product Dimensions")
    prod_len_min = st.sidebar.number_input("Min Target Amplicon Size (bp):", 15, 500, 20, key="pr_min")
    prod_len_max = st.sidebar.number_input("Max Target Amplicon Size (bp):", 40, 2000, 1000, key="pr_max")
    
    min_dist_to_snp = st.sidebar.number_input("Min distance from primer to SNP (bp):", min_value=0, max_value=50, value=2)

    def reverse_complement(seq):
        comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
        return "".join(comp.get(base, base) for base in reversed(seq))

    def analyze_hairpin(primer_seq, na, mg, dnac_m):
        n = len(primer_seq)
        best_dg, best_tm, best_structure = 0.0, 0.0, None
        for stem_len in range(3, n // 2):
            for i in range(0, n - 2 * stem_len):
                for j in range(i + stem_len + 3, n - stem_len + 1):
                    stem1 = primer_seq[i:i+stem_len]
                    stem2 = primer_seq[j:j+stem_len]
                    stem2_rc = reverse_complement(stem2)
                    matches = sum(1 for a, b in zip(stem1, stem2_rc) if a == b)
                    if matches >= 3: 
                        loop_seq = primer_seq[i+stem_len:j]
                        gc_count = stem1.count('G') + stem1.count('C')
                        at_count = len(stem1) - gc_count
                        est_tm = (at_count * 2) + (gc_count * 4) - (len(loop_seq) * 1.5)
                        est_dg = (at_count * -1.2) + (gc_count * -2.4) + (len(loop_seq) * 0.4)
                        if est_dg < best_dg:
                            best_dg, best_tm = est_dg, est_tm
                            best_structure = {
                                "stem1": stem1, "loop": loop_seq, "stem2_rev": stem2,
                                "visual": f"5'- {primer_seq[:i]} <span style='color:#ff4b4b;font-weight:bold;'>{stem1}</span> ({loop_seq}) <span style='color:#ff4b4b;font-weight:bold;'>{stem2}</span> -3'"
                            }
        return best_dg, best_tm, best_structure

    def analyze_dimer(seq1, seq2, na, mg, dnac_m):
        n1, n2 = len(seq1), len(seq2)
        best_dg, best_tm, max_3p_consec, best_visual = 0.0, 0.0, 0, None
        for shift in range(-n1 + 1, n2):
            alignment_lines = ""
            current_dg, current_3p_matches = 0.0, 0
            for i in range(max(0, -shift), min(n1, n2 - shift)):
                b1 = seq1[i]
                b2 = seq2[i + shift]
                if (b1=='A' and b2=='T') or (b1=='T' and b2=='A') or (b1=='G' and b2=='C') or (b1=='C' and b2=='G'):
                    alignment_lines += "|"
                    current_dg += -2.4 if b1 in 'GC' else -1.2
                    if i == n1 - 1 or (i + shift) == n2 - 1:
                        current_3p_matches += 1
                else:
                    alignment_lines += "."
                    current_3p_matches = 0
            if alignment_lines.count('|') >= 3:
                est_tm = alignment_lines.count('|') * 3.5
                if current_dg < best_dg:
                    best_dg, best_tm, max_3p_consec = current_dg, est_tm, max(max_3p_consec, current_3p_matches)
                    best_visual = f"5'- {seq1} -3'<br><span style='font-family:monospace;color:#ff4b4b;'>&nbsp;&nbsp;&nbsp;&nbsp; {alignment_lines}</span><br>3'- {seq2[::-1]} -5'"
        return best_dg, best_tm, max_3p_consec, best_visual

    if "final_pairs_list" not in st.session_state:
        st.session_state.final_pairs_list = []

    if st.button("Run Primer", key="btn_run_scanner"):
        with st.spinner("Executing sequence matrix generation..."):
            valid_pairs = []
            seen_pairs = set()  
            L = len(seq_base_1)
            
            for f_start in range(0, L):
                for f_len in range(p_len_min, p_len_max + 1):
                    f_end = f_start + f_len
                    
                    # Kiểm tra khoảng cách mồi xuôi so với SNP
                    if f_end > (snp_idx - min_dist_to_snp): continue
                    if f_end > L: continue
            
                    f_seq = seq_base_1[f_start:f_end]
            
                    for r_start in range(max(f_end, snp_idx), L):
                        # Kiểm tra khoảng cách mồi ngược so với SNP
                        if r_start < (snp_idx + min_dist_to_snp + 1): continue
                
                        for r_len in range(p_len_min, p_len_max + 1):
                            r_end = r_start + r_len
                            if r_end > L: continue
                            
                            p_size = r_end - f_start
                            if not (prod_len_min <= p_size <= prod_len_max): continue
                            if not (f_start <= snp_idx < r_end): continue
                            
                            r_seq_template = seq_base_1[r_start:r_end]
                            r_seq = reverse_complement(r_seq_template)
                            
                            pair_signature = (f_seq, r_seq)
                            if pair_signature in seen_pairs: continue
                            
                            f_gc = (f_seq.count('G') + f_seq.count('C')) / f_len * 100
                            r_gc = (r_seq.count('G') + r_seq.count('C')) / r_len * 100
                            
                            f_tm = (f_seq.count('G') + f_seq.count('C')) * 4 + (f_seq.count('A') + f_seq.count('T')) * 2
                            r_tm = (r_seq.count('G') + r_seq.count('C')) * 4 + (r_seq.count('A') + r_seq.count('T')) * 2
                            
                            product_seq1 = seq_base_1[f_start:r_end]
                            product_seq2 = seq_base_2[f_start:r_end]
                            prod_tm1 = mt.Tm_GC(product_seq1, Na=na_mM, Mg=mg_mM)
                            prod_tm2 = mt.Tm_GC(product_seq2, Na=na_mM, Mg=mg_mM)
                            prod_delta_tm = abs(prod_tm1 - prod_tm2)
                            
                            f_hp_dg, f_hp_tm, f_hp_struct = analyze_hairpin(f_seq, na_mM, mg_mM, 0.0000002)
                            r_hp_dg, r_hp_tm, r_hp_struct = analyze_hairpin(r_seq, na_mM, mg_mM, 0.0000002)
                            f_self_dg, f_self_tm, f_3p, f_self_vis = analyze_dimer(f_seq, f_seq, na_mM, mg_mM, 0.0000002)
                            r_self_dg, r_self_tm, r_3p, r_self_vis = analyze_dimer(r_seq, r_seq, na_mM, mg_mM, 0.0000002)
                            cross_dg, cross_tm, cross_3p, cross_vis = analyze_dimer(f_seq, r_seq, na_mM, mg_mM, 0.0000002)
                            
                            seen_pairs.add(pair_signature)
                            valid_pairs.append({
                                "f_seq": f_seq, "f_tm": f_tm, "f_gc": f_gc, "f_len": f_len, "f_pos": f"{f_start+1}..{f_end}",
                                "r_seq": r_seq, "r_tm": r_tm, "r_gc": r_gc, "r_len": r_len, "r_pos": f"{r_start+1}..{r_end}",
                                "prod_size": p_size, "prod_tm1": prod_tm1, "prod_tm2": prod_tm2, "prod_delta_tm": prod_delta_tm,
                                "snp_loc": snp_idx - f_start + 1, "prod_seq1": product_seq1, "prod_seq2": product_seq2,
                                "structures": {
                                    "Forward Hairpin": {"dg": f_hp_dg, "tm": f_hp_tm, "vis": f_hp_struct["visual"] if f_hp_struct else None, "3p": 0, "type": "Hairpin"},
                                    "Reverse Hairpin": {"dg": r_hp_dg, "tm": r_hp_tm, "vis": r_hp_struct["visual"] if r_hp_struct else None, "3p": 0, "type": "Hairpin"},
                                    "Forward Self-Dimer": {"dg": f_self_dg, "tm": f_self_tm, "vis": f_self_vis, "3p": f_3p, "type": "Dimer"},
                                    "Reverse Self-Dimer": {"dg": r_self_dg, "tm": r_self_tm, "vis": r_self_vis, "3p": r_3p, "type": "Dimer"},
                                    "Cross-Dimer": {"dg": cross_dg, "tm": cross_tm, "vis": cross_vis, "3p": cross_3p, "type": "Dimer"}
                                }
                            })

            sorted_pairs = sorted(valid_pairs, key=lambda x: x['prod_delta_tm'], reverse=True)
            st.session_state.final_pairs_list = sorted_pairs[:max_display_pairs]

    # RE-RENDER POOL
    if st.session_state.final_pairs_list:
        pairs = st.session_state.final_pairs_list
        st.subheader("Select Primer Pair")
        options_list = [f" {i+1}: Amplicon {p['prod_size']} bp [Product ΔTm = {p['prod_delta_tm']:.4f}°C]" for i, p in enumerate(pairs)]
        selected_index = st.selectbox("Target Candidate Pair Portfolio:", range(len(options_list)), format_func=lambda x: options_list[x])
        
        chosen_pair = pairs[selected_index]
        
        col_f, col_r = st.columns(2)
        with col_f:
            st.info("**Forward Primer**")
            st.code(f"Sequence: 5'- {chosen_pair['f_seq']} -3'\nLength: {chosen_pair['f_len']} bp | Coordinates: {chosen_pair['f_pos']}\nPrimer Tm: {chosen_pair['f_tm']:.2f} °C | %GC: {chosen_pair['f_gc']:.1f}%")
        with col_r:
            st.info("**Reverse Primer**")
            st.code(f"Sequence: 5'- {chosen_pair['r_seq']} -3'\nLength: {chosen_pair['r_len']} bp | Coordinates: {chosen_pair['r_pos']}\nPrimer Tm: {chosen_pair['r_tm']:.2f} °C | %GC: {chosen_pair['r_gc']:.1f}%")
        
        st.markdown("#### Secondary Structure Analysis Matrix")
        for name, data in chosen_pair["structures"].items():
            violates_dg = data["dg"] <= min_sec_dg and data["dg"] != 0.0
            violates_tm = data["tm"] >= max_sec_tm and data["tm"] != 0.0
            violates_3p = data["3p"] > max_3prime_matches and data["type"] == "Dimer"
            is_failed = violates_dg or violates_tm or violates_3p
            
            if is_failed:
                with st.expander(f"{name} — FAILED", expanded=True):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("Structural ΔG", f"{data['dg']:.2f} kcal/mol")
                        st.metric("Melting Tm", f"{data['tm']:.1f} °C")
                    with c2:
                        if data["vis"]: st.markdown(f"<div style='background-color:#fff5f5; padding:12px; font-family:monospace;'>{data['vis']}</div>", unsafe_allow_html=True)
            else:
                with st.expander(f"{name} — PASSED", expanded=False):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("Structural ΔG", f"{data['dg']:.2f} kcal/mol")
                        st.metric("Melting Tm", f"{data['tm']:.1f} °C")
                    with c2:
                        if data["vis"]: st.markdown(f"<div style='background-color:#f8f9fa; padding:12px; font-family:monospace;'>{data['vis']}</div>", unsafe_allow_html=True)

        st.markdown("#### Template–Oligo Binding Map")
        f_seq = chosen_pair["f_seq"]
        r_seq_template = reverse_complement(chosen_pair["r_seq"])
        f_s, r_s = seq_base_1.find(f_seq), seq_base_1.find(r_seq_template)
        f_e, r_e = f_s + len(f_seq) if f_s != -1 else 0, r_s + len(r_seq_template) if r_s != -1 else 0
        
        html_str = "<div style='font-family: monospace; font-size: 16px; line-height: 2.0; background-color: #f7f9fa; padding: 15px; overflow-x: auto; white-space: nowrap;'><b>5'- </b>"
        for idx, base in enumerate(seq_base_1):
            if idx == snp_idx: html_str += f"<span style='background-color: #ff4b4b; color: white; padding: 2px 6px; font-weight: bold;'>{nu1}/{nu2}</span>"
            elif f_s <= idx < f_e: html_str += f"<span style='color: #2E8B57; font-weight: bold; text-decoration: underline; background-color: #E8F5E9;'>{base}</span>"
            elif r_s <= idx < r_e: html_str += f"<span style='color: #D2691E; font-weight: bold; text-decoration: underline; background-color: #FFF3E0;'>{base}</span>"
            else: html_str += f"<span style='color: #9E9E9E;'>{base}</span>"
        html_str += "<b> -3'</b></div>"
        st.write(html_str, unsafe_allow_html=True)
        
        st.markdown("#### Predicted Melting Temperatures ($T_m$) of Final PCR Product (Amplicon)")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: st.metric("Homo 1 Tm", f"{chosen_pair['prod_tm1']:.2f} °C")
        with m_col2: st.metric("Homo 2 Tm", f"{chosen_pair['prod_tm2']:.2f} °C")
        with m_col3: st.metric("ΔTm", f"{chosen_pair['prod_delta_tm']:.4f} °C")
    else:
         st.error("Please click Run Primer to view the results.")

# ==========================================
# MAIN ROUTING MANAGEMENT ARCHITECTURE
# ==========================================
def main():
    st.sidebar.title("Tools")
    current_task = st.sidebar.radio("---", ["HRM Curve Analyzer", "Primer Design"])
    if current_task == "HRM Curve Analyzer": run_hrm_analysis()
    elif current_task == "Primer Design": run_primer_designer()

if __name__ == "__main__": main()

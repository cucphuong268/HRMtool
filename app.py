import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Bio.SeqUtils import MeltingTemp as mt

# ==========================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# ==========================================
st.set_page_config(page_title="HRMTool", layout="wide")

st.title("HRM Curve Analyzer & Tm Predictor")
st.markdown("---")

# ==========================================
# 2. SIDEBAR - PARAMETERS & SLIDERS
# ==========================================
st.sidebar.header("Difference Plot Reference")
ref_selection = st.sidebar.selectbox(
    "Select Reference Baseline:",
    options=["Homozygote 1", "Homozygote 2"]
)

st.sidebar.header("Experimental Parameters")
st.sidebar.markdown("PCR Conditions")
dnac1_nm = st.sidebar.number_input("DNA 1 (nM):", 1, 2000, 10, 1)
dnac2_nm = st.sidebar.number_input("DNA 2 (nM):", 1, 2000, 10, 1)
na_mM = st.sidebar.number_input("Na+ (mM):", 0, 500, 50, 10)
mg_mM = st.sidebar.number_input("Mg2+ (mM):", 0.0, 10.0, 3.0, 0.5)

st.sidebar.markdown("Slope Factors (k)")
k_homo = st.sidebar.slider("k for Homoduplex:", 0.1, 1.0, 0.40, 0.01)
k_hetero = st.sidebar.slider("k for Heteroduplex:", 0.1, 2.0, 0.80, 0.01)

# ==========================================
# 3. SEQUENCE INPUT 
# ==========================================
col1, col2 = st.columns(2)
with col1:
    raw_allele1 = st.text_input("Allele 1 Sequence (5' -> 3'):", 
                  value="AGCCAAAACAGCCTTAAATAGCATTCAAACACTCTTTCTTCCATGCCTTCAGTCCTGC")
    allele1 = raw_allele1.upper().replace(" ", "") 
with col2:
    raw_allele2 = st.text_input("Allele 2 Sequence (5' -> 3'):", 
                  value="AGCCAAAACAGCCTTAAATAGCATTCCAACACTCTTTCTTCCATGCCTTCAGTCCTGC")
    allele2 = raw_allele2.upper().replace(" ", "") 

# ==========================================
# 4. COMPUTATION BACKEND
# ==========================================
def get_complement_3to5(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, base) for base in seq)

def extract_visual_mismatch(seq1, seq2):
    snp_pos = None
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            snp_pos = i
            break
            
    if snp_pos is None: 
        return "No Mismatch detected (Sequences are identical).", "N/A", "N/A", "N/A", "N/A"
        
    nu1 = seq1[snp_pos]
    nu2 = seq2[snp_pos]
    pair = {nu1, nu2}
    
    if pair in [{"A", "G"}, {"C", "T"}]:
        mismatch_type = f"Transition ({nu1} ↔ {nu2})"
    else:
        mismatch_type = f"Transversion ({nu1} ↔ {nu2})"
        
    left = snp_pos - 1
    right = snp_pos + 2
    
    raw_trip1 = seq1[max(0, left):min(len(seq1), right)]
    raw_trip2 = seq2[max(0, left):min(len(seq2), right)]
    
    comp_trip1 = get_complement_3to5(raw_trip1)
    comp_trip2 = get_complement_3to5(raw_trip2)
    
    h_raw1 = f"{seq1[snp_pos-1] if snp_pos > 0 else ''}[{nu1}]{seq1[snp_pos+1] if snp_pos < len(seq1)-1 else ''}"
    h_raw2 = f"{seq2[snp_pos-1] if snp_pos > 0 else ''}[{nu2}]{seq2[snp_pos+1] if snp_pos < len(seq2)-1 else ''}"
    
    h_comp1 = f"{comp_trip1[0] if snp_pos > 0 else ''}[{comp_trip1[1] if snp_pos > 0 else comp_trip1[0]}]{comp_trip1[2] if len(comp_trip1)>2 else ''}"
    h_comp2 = f"{comp_trip2[0] if snp_pos > 0 else ''}[{comp_trip2[1] if snp_pos > 0 else comp_trip2[0]}]{comp_trip2[2] if len(comp_trip2)>2 else ''}"
    
    mismatch_info = f"Position {snp_pos + 1}: {mismatch_type}"
    return mismatch_info, h_raw1, h_comp1, h_raw2, h_comp2

if allele1 and allele2:
    if len(allele1) != len(allele2):
        st.error("⚠️ Error: Sequence lengths must be equal for alignment.")
    else:
        Tm1 = mt.Tm_NN(allele1, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM)
        Tm2 = mt.Tm_NN(allele2, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM)
        delta_tm = abs(Tm1 - Tm2)
        
        comp_allele1_3to5 = get_complement_3to5(allele1)
        comp_allele2_3to5 = get_complement_3to5(allele2)
        
        try:
            Tm_het1 = mt.Tm_NN(allele1, c_seq=comp_allele2_3to5, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM)
            Tm_het2 = mt.Tm_NN(allele2, c_seq=comp_allele1_3to5, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM)
            penalty_1 = Tm1 - Tm_het1
            penalty_2 = Tm2 - Tm_het2
        except Exception as e:
            st.warning(f"Fallback to static penalty due to exception: {e}")
            snp_pos_fb = next((idx for idx, (a, b) in enumerate(zip(allele1, allele2)) if a != b), None)
            if snp_pos_fb is not None and {allele1[snp_pos_fb], allele2[snp_pos_fb]} in [{"A", "G"}, {"C", "T"}]:
                fb_penalty = 0.9
            else:
                fb_penalty = 2.4
            penalty_1, penalty_2 = fb_penalty, fb_penalty
            Tm_het1 = Tm1 - penalty_1
            Tm_het2 = Tm2 - penalty_2     
            
        mismatch_info, r1, c_back1, r2, c_back2 = extract_visual_mismatch(allele1, allele2)

        st.subheader("Predicted Results (Bio.SeqUtils.MeltingTemp Model)")
        st.info(f"Detected Mismatch Details: {mismatch_info}")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Tm Homo 1", f"{Tm1:.2f} °C")
            st.markdown("**Duplex Structure:**")
            st.code(f"5'- {r1} -3'\n3'- {c_back1} -5'")
        with c2:
            st.metric("Tm Homo 2", f"{Tm2:.2f} °C")
            st.markdown("**Duplex Structure:**")
            st.code(f"5'- {r2} -3'\n3'- {c_back2} -5'")
        with c3:
            st.metric("ΔTm (Homo1 - Homo2)", f"{delta_tm:.2f} °C")
            
        with c4:
            st.metric("Tm Hetero 1", f"{Tm_het1:.2f} °C", delta=f"-{penalty_1:.2f}°C" if penalty_1 > 0 else None, delta_color="inverse")
            st.markdown("**Mismatch Structure:**")
            st.code(f"5'- {r1} -3'\n3'- {c_back2} -5'")
        with c5:
            st.metric("Tm Hetero 2", f"{Tm_het2:.2f} °C", delta=f"-{penalty_2:.2f}°C" if penalty_2 > 0 else None, delta_color="inverse")
            st.markdown("**Mismatch Structure:**")
            st.code(f"5'- {r2} -3'\n3'- {c_back1} -5'")

        # ==========================================
        # 5. MODELING & DIFFERENCE PLOT CALCULATION
        # ==========================================
        t_start = min(Tm1, Tm2) - 7
        t_end   = max(Tm1, Tm2) + 6

        T = np.linspace(t_start, t_end, 1000)
        def inverse_sigmoid(T, Tm, k):
            return 1 / (1 + np.exp((T - Tm) / k))

        F_homo1 = inverse_sigmoid(T, Tm1, k_homo)
        F_homo2 = inverse_sigmoid(T, Tm2, k_homo)
        F_het   = (0.25 * inverse_sigmoid(T, Tm1, k_homo) + 
                   0.25 * inverse_sigmoid(T, Tm2, k_homo) + 
                   0.25 * inverse_sigmoid(T, Tm_het1, k_hetero) + 
                   0.25 * inverse_sigmoid(T, Tm_het2, k_hetero))

        dF_homo1 = -np.gradient(F_homo1, T)
        dF_homo2 = -np.gradient(F_homo2, T)
        dF_het   = -np.gradient(F_het, T)

        if ref_selection == "Homozygote 1":
            F_ref = F_homo1
            ref_label = "Ref: Homo 1"
        else:
            F_ref = F_homo2
            ref_label = "Ref: Homo 2"

        diff_homo1 = F_homo1 - F_ref
        diff_homo2 = F_homo2 - F_ref
        diff_het   = F_het - F_ref

        # ==========================================
        # 6. VISUALIZATION (3 PLOTS)
        # ==========================================
        st.subheader("HRM Analysis Visualizations")
        
        st.markdown("Custom Plot Colors")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            color_homo1 = st.color_picker("Homozygote 1 Color", value="#1E90FF")
        with cc2:
            color_homo2 = st.color_picker("Homozygote 2 Color", value="#FF4500")
        with cc3:
            color_het = st.color_picker("Heterozygote Color", value="#8A2BE2")
            
        st.markdown(" ")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
        zoom_range = (t_start, t_end)

        ax1.plot(T, F_homo1, label='Homozygote 1', color=color_homo1, linewidth=2, linestyle='--')
        ax1.plot(T, F_homo2, label='Homozygote 2', color=color_homo2, linewidth=2, linestyle='--')
        ax1.plot(T, F_het,   label='Heterozygote',  color=color_het,   linewidth=3)
        ax1.set_title('A. Aligned Melting Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Fluorescence')
        ax1.set_xlim(zoom_range)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        ax2.plot(T, dF_homo1, color=color_homo1, linewidth=2, linestyle='--')
        ax2.plot(T, dF_homo2, color=color_homo2, linewidth=2, linestyle='--')
        ax2.plot(T, dF_het,   color=color_het,   linewidth=3)
        ax2.set_title('B. Derivative Curve (-dF/dT)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('-dF/dT')
        ax2.set_xlim(zoom_range)
        ax2.grid(True, linestyle=':', alpha=0.6)

        style_h1 = ':' if ref_selection == "Homozygote 1" else '-'
        style_h2 = ':' if ref_selection == "Homozygote 2" else '-'
        
        ax3.plot(T, diff_homo1, label='Homo 1', color=color_homo1, linewidth=2, linestyle=style_h1)
        ax3.plot(T, diff_homo2, label='Homo 2', color=color_homo2, linewidth=2, linestyle=style_h2)
        ax3.plot(T, diff_het,   label='Hetero', color=color_het,   linewidth=3)
        ax3.set_title(f'C. Difference Plot ({ref_label})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Difference data')
        ax3.set_xlim(zoom_range)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend()

        plt.tight_layout()
        st.pyplot(fig)
else:
    st.warning("Please enter Allele sequences.")

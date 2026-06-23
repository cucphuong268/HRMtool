import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Bio.SeqUtils import MeltingTemp as mt

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="HRMTool", layout="wide")

st.title("HRM Curve Analyzer & Tm Predictor")
st.markdown("---")

# ==========================================
# 2. SIDEBAR - PARAMETERS & CUSTOMIZATION
# ==========================================
st.sidebar.header("Colors")
# Color Pickers for User Customization
color_homo1 = st.sidebar.color_picker("Homozygote 1 Color", "#1E90FF")
color_homo2 = st.sidebar.color_picker("Homozygote 2 Color", "#FF4500")
color_het   = st.sidebar.color_picker("Heterozygote Color", "#8A2BE2") 

st.sidebar.header("Difference Plot Reference")
# User option to select the baseline reference for the Difference Plot
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
k_hetero = st.sidebar.slider("k for Heteroduplex:", 0.1, 1.5, 0.80, 0.01)

# ==========================================
# 3. SEQUENCE INPUT 
# ==========================================
col1, col2 = st.columns(2)
with col1:
    raw_allele1 = st.text_input("Allele 1 Sequence:", 
                  value="AGCCAAAACAGCCTTAAATAGCATTCAAACACTCTTTCTTCCATGCCTTCAGTCCTGC")
    allele1 = raw_allele1.upper().replace(" ", "") 
with col2:
    raw_allele2 = st.text_input("Allele 2 Sequence:", 
                  value="AGCCAAAACAGCCTTAAATAGCATTCCAACACTCTTTCTTCCATGCCTTCAGTCCTGC")
    allele2 = raw_allele2.upper().replace(" ", "") 

# ==========================================
# 4. COMPUTATION BACKEND
# ==========================================
def estimate_hetero_penalty(seq1, seq2):
    snp_pos = None
    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a != b:
            snp_pos = i
            break
    if snp_pos is None: return 0.0
    pair = {seq1[snp_pos], seq2[snp_pos]}
    return 1.0 if pair in [{"A", "G"}, {"C", "T"}] else 2.0

if allele1 and allele2:
    if len(allele1) != len(allele2):
        st.error("⚠️ Error: Sequence lengths must be equal for alignment.")
    else:
    
        # Calculate Tms using Biopython
        Tm1 = mt.Tm_NN(allele1, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM)
        Tm2 = mt.Tm_NN(allele2, nn_table=mt.DNA_NN4, dnac1=dnac1_nm, dnac2=dnac2_nm, Na=na_mM, Mg=mg_mM)
        
        penalty = estimate_hetero_penalty(allele1, allele2)
        Tm_het1, Tm_het2 = Tm1 - penalty, Tm2 - penalty

        # Metrics display
        st.subheader("Predicted Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tm Homo 1", f"{Tm1:.2f} °C")
        c2.metric("Tm Homo 2", f"{Tm2:.2f} °C")
        c3.metric("Tm Hetero 1", f"{Tm_het1:.2f} °C", delta=f"-{penalty}°C", delta_color="inverse")
        c4.metric("Tm Hetero 2", f"{Tm_het2:.2f} °C", delta=f"-{penalty}°C", delta_color="inverse")

        # ==========================================
        # 5. MODELING & DIFFERENCE PLOT CALCULATION
        # ==========================================
        T = np.linspace(65, 95, 1000)
        def inverse_sigmoid(T, Tm, k):
            return 1 / (1 + np.exp((T - Tm) / k))

        # Core Fluorescence Functions
        F_homo1 = inverse_sigmoid(T, Tm1, k_homo)
        F_homo2 = inverse_sigmoid(T, Tm2, k_homo)
        F_het   = (0.30 * inverse_sigmoid(T, Tm1, k_homo) + 
                   0.30 * inverse_sigmoid(T, Tm2, k_homo) + 
                   0.2 * inverse_sigmoid(T, Tm_het1, k_hetero) + 
                   0.2 * inverse_sigmoid(T, Tm_het2, k_hetero))

        # Derivative Curves
        dF_homo1 = -np.gradient(F_homo1, T)
        dF_homo2 = -np.gradient(F_homo2, T)
        dF_het   = -np.gradient(F_het, T)

        # Dynamic Reference Assignment
        if ref_selection == "Homozygote 1":
            F_ref = F_homo1
            ref_label = "Ref: Homo 1"
        else:
            F_ref = F_homo2
            ref_label = "Ref: Homo 2"

        # Difference Curves Calculation (F_target - F_ref)
        diff_homo1 = F_homo1 - F_ref
        diff_homo2 = F_homo2 - F_ref
        diff_het   = F_het - F_ref

        # ==========================================
        # 6. VISUALIZATION (3 PLOTS)
        # ==========================================
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
        zoom_range = (min(Tm1, Tm2) - 6, max(Tm1, Tm2) + 6)

        # Plot A: Aligned Melting Curve
        ax1.plot(T, F_homo1, label='Homozygote 1', color=color_homo1, linewidth=2, linestyle='--')
        ax1.plot(T, F_homo2, label='Homozygote 2', color=color_homo2, linewidth=2, linestyle='--')
        ax1.plot(T, F_het,   label='Heterozygote',  color=color_het,   linewidth=3)
        ax1.set_title('A. Aligned Melting Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Temperature (°C)'); ax1.set_ylabel('Fluorescence')
        ax1.set_xlim(zoom_range); ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # Plot B: Derivative Melt Curve
        ax2.plot(T, dF_homo1, color=color_homo1, linewidth=2, linestyle='--')
        ax2.plot(T, dF_homo2, color=color_homo2, linewidth=2, linestyle='--')
        ax2.plot(T, dF_het,   color=color_het,   linewidth=3)
        ax2.set_title('B. Derivative Curve (-dF/dT)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Temperature (°C)'); ax2.set_ylabel('-dF/dT')
        ax2.set_xlim(zoom_range); ax2.grid(True, linestyle=':', alpha=0.6)

        # Plot C: Difference Plot with Dynamic Reference Line Style
        style_h1 = ':' if ref_selection == "Homozygote 1" else '-'
        style_h2 = ':' if ref_selection == "Homozygote 2" else '-'
        
        ax3.plot(T, diff_homo1, label='Homo 1', color=color_homo1, linewidth=2, linestyle=style_h1)
        ax3.plot(T, diff_homo2, label='Homo 2', color=color_homo2, linewidth=2, linestyle=style_h2)
        ax3.plot(T, diff_het,   label='Hetero', color=color_het,   linewidth=3)
        
        ax3.set_title(f'C. Difference Plot ({ref_label})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Temperature (°C)'); ax3.set_ylabel('Difference data')
        ax3.set_xlim(zoom_range); ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend()

        plt.tight_layout()
        st.subheader("HRM Analysis Visualizations")
        st.pyplot(fig)
else:
    st.warning("Please enter Allele sequences.")

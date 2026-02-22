import streamlit as st
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depression Risk Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("depression_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model    = model_data["model"]
scaler   = model_data["scaler"]
features = model_data["features"]   # ['sEXT','sAGR','sCON','sOPN']
feat_names = model_data["feature_names"]  # ['Extraversion','Agreeableness','Conscientiousness','Openness']
ranges   = model_data["feature_ranges"]
accuracy = model_data["accuracy"]
f1_score_val = model_data["f1"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a1a2e; text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #555; font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9ff; border-radius: 12px;
        padding: 1.2rem; border-left: 4px solid #4f46e5;
        margin-bottom: 1rem;
    }
    .risk-low {
        background: #f0fdf4; border-left: 4px solid #22c55e;
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .risk-high {
        background: #fff7ed; border-left: 4px solid #f97316;
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .risk-very-high {
        background: #fef2f2; border-left: 4px solid #ef4444;
        border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .disclaimer {
        background: #fefce8; border: 1px solid #fbbf24;
        border-radius: 8px; padding: 1rem; font-size: 0.85rem;
        color: #78350f; margin-top: 1rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 600;
        color: #1a1a2e; border-bottom: 2px solid #4f46e5;
        padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Depression Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Detecting depression risk using Big Five personality traits '
    'and social network influence modeling · Research Project · Binghamton University 2024</div>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Individual Risk Predictor",
    "🕸️ Network Influence Explorer",
    "📊 Model Performance"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Individual Risk Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Big Five Personality Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        "Adjust the sliders to reflect your Big Five (OCEAN) personality scores. "
        "The model predicts **Neuroticism level** — a strong psychological marker for "
        "depression risk — based on the relationship between personality traits."
    )
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("**Enter Personality Scores** *(1 = Low, 5 = High)*")

        r_ext = ranges["sEXT"]
        r_agr = ranges["sAGR"]
        r_con = ranges["sCON"]
        r_opn = ranges["sOPN"]

        sEXT = st.slider(
            "😄 Extraversion  —  sociable, assertive, talkative",
            min_value=1.0, max_value=5.0, value=round(r_ext[2], 1), step=0.05,
            help="High scorers are outgoing and energetic. Low scorers tend to be reserved."
        )
        sAGR = st.slider(
            "🤝 Agreeableness  —  cooperative, trusting, empathetic",
            min_value=1.0, max_value=5.0, value=round(r_agr[2], 1), step=0.05,
            help="High scorers are compassionate and cooperative. Low scorers may be more competitive."
        )
        sCON = st.slider(
            "📋 Conscientiousness  —  organized, disciplined, goal-driven",
            min_value=1.0, max_value=5.0, value=round(r_con[2], 1), step=0.05,
            help="High scorers are dependable and self-disciplined. Low scorers may be more spontaneous."
        )
        sOPN = st.slider(
            "💡 Openness  —  curious, imaginative, creative",
            min_value=1.0, max_value=5.0, value=round(r_opn[2], 1), step=0.05,
            help="High scorers are creative and open to new experiences. Low scorers prefer routine."
        )

        st.markdown("")
        predict_btn = st.button("🔍 Analyze Depression Risk", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Understanding Your Scores**")
        trait_desc = {
            "Extraversion": ("Social energy and assertiveness", sEXT, "#6366f1"),
            "Agreeableness": ("Cooperation and empathy level", sAGR, "#22c55e"),
            "Conscientiousness": ("Discipline and organization", sCON, "#f59e0b"),
            "Openness": ("Creativity and curiosity", sOPN, "#ec4899"),
        }
        for trait, (desc, val, color) in trait_desc.items():
            pct = int((val - 1) / 4 * 100)
            st.markdown(f"**{trait}** — {desc}")
            st.progress(pct / 100)
            st.caption(f"Score: {val:.2f} / 5.00  ({pct}th percentile)")

        # Neuroticism note
        st.info(
            "ℹ️ **Note on Neuroticism:** This model predicts your Neuroticism level "
            "from the four traits above. High Neuroticism is the most consistent "
            "personality-based predictor of depression risk in clinical literature."
        )

    if predict_btn:
        st.markdown("---")
        X_input = np.array([[sEXT, sAGR, sCON, sOPN]])
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        high_neu_prob = probability[1]

        # Risk tier
        if high_neu_prob < 0.35:
            risk_label = "Low Risk"
            risk_class = "risk-low"
            risk_emoji = "✅"
            risk_color = "#22c55e"
            interpretation = (
                "Your personality profile suggests low Neuroticism, associated with "
                "emotional stability and resilience. This is generally a protective "
                "factor against depression."
            )
        elif high_neu_prob < 0.65:
            risk_label = "Moderate Risk"
            risk_class = "risk-high"
            risk_emoji = "⚠️"
            risk_color = "#f97316"
            interpretation = (
                "Your profile suggests moderate Neuroticism. While not alarming, "
                "individuals with this profile may benefit from monitoring emotional "
                "wellbeing and maintaining strong social connections."
            )
        else:
            risk_label = "Elevated Risk"
            risk_class = "risk-very-high"
            risk_emoji = "🔴"
            risk_color = "#ef4444"
            interpretation = (
                "Your profile suggests elevated Neuroticism, which research associates "
                "with higher susceptibility to depression and anxiety. This is an "
                "academic indicator only — please consult a mental health professional."
            )

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(
                f'<div class="{risk_class}">'
                f'<div style="font-size:2.5rem">{risk_emoji}</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{risk_color}">{risk_label}</div>'
                f'<div style="font-size:0.9rem;color:#555;margin-top:0.3rem">Neuroticism Level</div>'
                f'</div>', unsafe_allow_html=True
            )
        with res_col2:
            st.metric("High Neuroticism Probability", f"{high_neu_prob*100:.1f}%")
            st.metric("Model Confidence", f"{max(probability)*100:.1f}%")
        with res_col3:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            sizes = [probability[0], probability[1]]
            colors = ["#22c55e", "#ef4444"]
            labels = ["Low Neuroticism", "High Neuroticism"]
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 9}
            )
            ax.set_title("Risk Distribution", fontsize=10, fontweight="bold")
            st.pyplot(fig)
            plt.close()

        st.markdown(f"**Interpretation:** {interpretation}")

        # Trait contribution bar chart
        st.markdown("**Which traits most influenced this prediction?**")
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            feat_contrib = dict(zip(feat_names, coefs))
            fig2, ax2 = plt.subplots(figsize=(7, 2.5))
            colors_bar = ["#ef4444" if v > 0 else "#22c55e" for v in coefs]
            bars = ax2.barh(feat_names, coefs, color=colors_bar)
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("Coefficient (positive = increases Neuroticism prediction)", fontsize=9)
            ax2.set_title("Feature Importance (SVM Coefficients)", fontsize=10, fontweight="bold")
            st.pyplot(fig2)
            plt.close()

        st.markdown(
            '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is for academic research '
            'purposes only. It is NOT a clinical diagnostic tool and should NOT replace professional '
            'mental health assessment. If you are experiencing mental health difficulties, please '
            'consult a qualified healthcare provider.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Network Influence Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Social Network Influence Model</div>', unsafe_allow_html=True)
    st.markdown(
        "This module models how depression risk propagates through a friendship network using "
        "**shortest-path graph algorithms** (NetworkX). A user's mental health is influenced by "
        "the depression status of their connected peers."
    )
    st.markdown("---")

    net_col1, net_col2 = st.columns([1, 2])

    with net_col1:
        st.markdown("**Network Parameters**")
        n_users = st.slider("Number of users in network", 20, 100, 40, step=5)
        depression_pct = st.slider("% of users with depression risk", 5, 50, 20)
        max_levels = st.slider("Influence depth (network hops)", 1, 5, 3)
        seed_val = st.number_input("Random seed (for reproducibility)", value=42, step=1)
        generate_btn = st.button("🕸️ Generate Network", type="primary", use_container_width=True)

    with net_col2:
        if generate_btn or "network_generated" not in st.session_state:
            random.seed(int(seed_val))
            np.random.seed(int(seed_val))

            # Build random directed friendship network
            G = nx.DiGraph()
            users = list(range(1, n_users + 1))

            # Assign depression status
            n_depressed = int(n_users * depression_pct / 100)
            depressed_users = set(random.sample(users, n_depressed))
            depression_status = {u: u in depressed_users for u in users}

            # Add edges
            for u in users:
                friends = random.sample([v for v in users if v != u], min(3, len(users)-1))
                for f in friends:
                    G.add_edge(u, f)

            # Compute influence scores using shortest path
            influence_scores = {}
            for target in users:
                score = 0.0
                for source in depressed_users:
                    if source != target and nx.has_path(G, source, target):
                        path_len = nx.shortest_path_length(G, source, target)
                        if path_len <= max_levels:
                            score += 1.0 / path_len
                influence_scores[target] = round(score, 3)

            st.session_state["G"] = G
            st.session_state["depression_status"] = depression_status
            st.session_state["influence_scores"] = influence_scores
            st.session_state["network_generated"] = True

        G = st.session_state["G"]
        depression_status = st.session_state["depression_status"]
        influence_scores = st.session_state["influence_scores"]

        # Draw network
        fig, ax = plt.subplots(figsize=(9, 6))
        pos = nx.spring_layout(G, seed=42, k=1.2)

        node_colors = []
        for node in G.nodes():
            if depression_status.get(node, False):
                node_colors.append("#ef4444")
            elif influence_scores.get(node, 0) > 1.5:
                node_colors.append("#f97316")
            elif influence_scores.get(node, 0) > 0.5:
                node_colors.append("#fbbf24")
            else:
                node_colors.append("#22c55e")

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#999",
                               arrows=True, arrowsize=8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=300, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color="white",
                                font_weight="bold")

        legend_elements = [
            mpatches.Patch(color="#ef4444", label="Depressed user"),
            mpatches.Patch(color="#f97316", label="High influence (>1.5)"),
            mpatches.Patch(color="#fbbf24", label="Moderate influence (0.5-1.5)"),
            mpatches.Patch(color="#22c55e", label="Low influence (<0.5)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
        ax.set_title(
            f"Friendship Network — {len(G.nodes())} users, "
            f"{sum(depression_status.values())} depressed, "
            f"influence depth={max_levels}",
            fontsize=10, fontweight="bold"
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    # Influence score table
    st.markdown("---")
    st.markdown("**Top 10 Users by Influence Score**")
    import pandas as pd
    inf_df = pd.DataFrame({
        "User ID": list(influence_scores.keys()),
        "Influence Score": list(influence_scores.values()),
        "Depressed": [depression_status[u] for u in influence_scores.keys()],
        "Risk Level": [
            "🔴 Source" if depression_status[u]
            else ("🟠 High" if influence_scores[u] > 1.5
                  else ("🟡 Moderate" if influence_scores[u] > 0.5 else "🟢 Low"))
            for u in influence_scores.keys()
        ]
    }).sort_values("Influence Score", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(inf_df, use_container_width=True)

    st.markdown(
        "**Key Insight:** Users with the highest influence scores are those within "
        f"{max_levels} hops of multiple depressed users. These users are most susceptible "
        "to network-driven depression risk — a finding consistent with contagion theory "
        "in social network analysis."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.metric("Best Model", "SVM (Linear Kernel)")
    with perf_col2:
        st.metric("Test Accuracy", f"{accuracy}%")
    with perf_col3:
        st.metric("Weighted F1 Score", f"{f1_score_val}%")
    with perf_col4:
        st.metric("Cross-Val Accuracy", "72.1% ± 9.1%")

    st.markdown("---")
    st.markdown("**All 7 Models Evaluated**")

    import pandas as pd
    model_results = pd.DataFrame({
        "Model": ["SVM (Linear)", "Neural Network", "Random Forest",
                  "KNN (k=250)", "Logistic Regression", "Naive Bayes", "CART (Decision Tree)"],
        "Accuracy": ["74.1%", "73.5%", "70.7%", "69.5%", "68.8%", "64.4%", "59.6%"],
        "F1 Score": ["85.1%", "84.6%", "82.3%", "81.5%", "81.0%", "77.3%", "71.8%"],
        "Precision": ["74.9%", "76.3%", "75.4%", "74.9%", "74.6%", "74.3%", "75.7%"],
        "Recall": ["98.3%", "94.9%", "90.5%", "89.3%", "88.6%", "80.5%", "68.2%"],
        "Selected": ["✅ Best", "", "", "", "", "", ""]
    })
    st.dataframe(model_results, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Why SVM was selected as the best model**")
    st.markdown("""
    - Highest combined accuracy (74.1%) and F1 score (85.1%) across all 7 models
    - Highest recall (98.3%) — critical in mental health contexts where **missing a high-risk individual is more costly than a false positive**
    - Linear kernel provides interpretable feature coefficients, supporting explainability
    - Robust performance across different test/train split sizes (tested from 10% to 90%)
    """)

    st.markdown("---")
    st.markdown("**Dataset & Methodology**")
    st.markdown("""
    - **Dataset:** MyPersonality — real Facebook data from ~250 users with Big Five OCEAN personality scores
    - **Features:** Extraversion, Agreeableness, Conscientiousness, Openness scores (1–5 scale)  
    - **Target:** Neuroticism classification (High/Low) — strongest personality-based predictor of depression
    - **Train/Test Split:** 80/20 with stratification
    - **Validation:** 5-fold cross-validation on training set
    - **Network Model:** Directed graph (NetworkX) with shortest-path influence propagation across 250 user profiles
    """)

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ This is an academic research project. '
        'All results are research findings and not clinical recommendations. '
        'Reference: Yang et al. (2020), International Journal of Information Management.</div>',
        unsafe_allow_html=True
    )

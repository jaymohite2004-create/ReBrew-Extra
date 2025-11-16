
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    r2_score,
    mean_squared_error,
)
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="ReBrew – Subscription Intelligence Cockpit",
    layout="wide",
    page_icon="☕",
)

DATA_PATH = "ReBrew_Market_Survey_Synthetic_Data_600_responses.xlsx"
TARGET_COL = "Q33_Subscription_Interest"
DROP_COLS = ["Response_ID", "Timestamp"]

TEAM_MEMBERS = [
    "👑 Jay Mohite – Head of Marketing & AI Vision",
    "✨ Niyati – Consumer Insights & Storytelling",
    "🚀 Kavish – Data & Growth Experiments",
    "🎯 Mansi – Campaign Performance & Targeting",
    "🌈 Laavanya – Brand Experience & CX",
    "🧠 Kshitij – Product Strategy & Innovation",
]


# ------------------------------------------------------------
# DATA & MODEL UTILITIES
# ------------------------------------------------------------
@st.cache_data
def load_data(path: str):
    return pd.read_excel(path)


def build_classification_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    }


def prepare_xy(df):
    df = df.dropna(subset=[TARGET_COL]).copy()
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = df[TARGET_COL]

    cat_features = X.columns.tolist()

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("cat", cat_transformer, cat_features)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocess


def train_and_eval_classifiers(df, selected_models):
    X_train, X_test, y_train, y_test, preprocess = prepare_xy(df)
    models = build_classification_models()

    metrics_records = []
    roc_info = {}
    pipelines = {}

    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)

    for name in selected_models:
        clf = models[name]
        pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")

        # macro-average ROC
        fpr_dict, tpr_dict = {}, {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            fpr_dict[cls] = fpr
            tpr_dict[cls] = tpr

        all_fpr = np.unique(np.concatenate(list(fpr_dict.values())))
        mean_tpr = np.zeros_like(all_fpr)
        for cls in classes:
            mean_tpr += np.interp(all_fpr, fpr_dict[cls], tpr_dict[cls])
        mean_tpr /= len(classes)

        roc_info[name] = {"fpr": all_fpr, "tpr": mean_tpr, "auc": auc}

        metrics_records.append(
            {
                "Algorithm": name,
                "Accuracy": acc,
                "Precision (macro)": prec,
                "Recall (macro)": rec,
                "F1-score (macro)": f1,
                "ROC AUC (macro)": auc,
            }
        )

        pipelines[name] = pipe

    metrics_df = pd.DataFrame(metrics_records).set_index("Algorithm")
    return metrics_df, roc_info, pipelines


def get_numeric_target_for_reg(df):
    mapping = {"No": 0, "Maybe": 1, "Yes, very interested": 2}
    return df[TARGET_COL].map(mapping)


def prepare_regression_data(df):
    df = df.dropna(subset=[TARGET_COL]).copy()
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = get_numeric_target_for_reg(df)

    cat_features = X.columns.tolist()

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("cat", cat_transformer, cat_features)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocess


def run_regression_lab(df):
    X_train, X_test, y_train, y_test, preprocess = prepare_regression_data(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.001, random_state=42),
    }

    records = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocess), ("reg", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        records.append({"Model": name, "R²": r2, "RMSE": rmse})

    return pd.DataFrame(records).set_index("Model")


def run_clustering(df, n_clusters=3):
    df = df.dropna(subset=[TARGET_COL]).copy()
    X = df.drop(columns=DROP_COLS + [TARGET_COL])
    y = df[TARGET_COL]

    cat_features = X.columns.tolist()

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("cat", cat_transformer, cat_features)]
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    X_enc = pipe.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X_enc)

    df_clust = df.copy()
    df_clust["Cluster"] = clusters

    pca = PCA(n_components=2, random_state=42)
    X_arr = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc
    coords = pca.fit_transform(X_arr)
    df_clust["PC1"] = coords[:, 0]
    df_clust["PC2"] = coords[:, 1]

    return df_clust


def run_association_rules(df):
    cols = [
        "Q1_Age_Group",
        "Q2_Gender",
        "Q6_Location_Type",
        "Q8_Coffee_Frequency",
        "Q18_Shopping_Channels",
        "Q20_Specific_Products",
        TARGET_COL,
    ]
    df_sub = df[cols].dropna().astype(str)
    basket = pd.get_dummies(df_sub, prefix_sep="=")
    freq_items = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.1)
    rules = rules.sort_values("lift", ascending=False)
    return rules.head(10)


# ------------------------------------------------------------
# KPI & INSIGHTS
# ------------------------------------------------------------
def compute_kpis(df):
    total_responses = len(df)
    features_tracked = df.shape[1] - len(DROP_COLS) - 1
    yes_rate = (df[TARGET_COL] == "Yes, very interested").mean()
    maybe_rate = (df[TARGET_COL] == "Maybe").mean()
    no_rate = (df[TARGET_COL] == "No").mean()
    return {
        "total_responses": total_responses,
        "features_tracked": features_tracked,
        "yes_rate": yes_rate,
        "maybe_rate": maybe_rate,
        "no_rate": no_rate,
    }


def show_executive_summary(df):
    kpis = compute_kpis(df)

    hero = f"""
    <div style='padding:24px 28px;border-radius:24px;
        background:linear-gradient(135deg,#1e293b 0%,#4f46e5 35%,#ec4899 100%);
        color:white;box-shadow:0 18px 45px rgba(15,23,42,0.55);margin-bottom:18px;'>
        <h1 style='margin-bottom:4px;'>☕ ReBrew – AI Subscription Intelligence</h1>
        <p style='font-size:0.96rem;opacity:0.92;'>
            ChatGPT-style cockpit that turns ReBrew's market survey into subscription intelligence.
        </p>
        <p style='font-size:0.9rem;opacity:0.9;'>
            We blend Decision Trees, Random Forests and Gradient Boosting to score every customer on
            <strong>Subscription Interest</strong> and surface high-intent segments for precision campaigns.
        </p>
    </div>
    """
    st.markdown(hero, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📊 Total Responses", f"{kpis['total_responses']}")
    with k2:
        st.metric("🧩 Features Tracked", f"{kpis['features_tracked']}+")
    with k3:
        st.metric("🔥 High-Intent Conversion", f"{kpis['yes_rate']*100:.1f}%")
    with k4:
        st.metric("🤝 Warm Leads (Maybe)", f"{kpis['maybe_rate']*100:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("💡 Why this dashboard matters for ReBrew")
        st.write(
            """
            - 🎯 **Precision Targeting** – Focus budget on customers most likely to subscribe.  
            - 🧪 **Test & Learn Lab** – Compare multiple ML models on the same data.  
            - 🧵 **Customer Storytelling** – Translate rows into personas & segments.  
            - 📦 **Operational Ready** – Export full dataset with predicted labels for CRM.  
            """
        )
        with st.expander("📌 Business Impact (click to open)"):
            st.write(
                """
                - Potential **30–40% reduction** in wasted impressions on low-intent users.  
                - Always-on **lead-scoring engine** for remarketing & WhatsApp/email flows.  
                - Design **personalised bundles** for 'Maybe' segment to push them into 'Yes'.  
                - Investor-ready visuals for ReBrew growth & funding presentations.  
                """
            )
    with c2:
        st.subheader("👨‍👩‍👧‍👦 ReBrew Intelligence Squad")
        st.write("Core team behind this dashboard:")
        st.write("\n".join([f"- {m}" for m in TEAM_MEMBERS]))
        st.markdown("**Quick Snapshot**")
        st.write(f"✅ Overall intent (Yes + Maybe): **{(1-kpis['no_rate'])*100:.1f}%**")
        st.write(f"🚫 No-interest customers: **{kpis['no_rate']*100:.1f}%**")


def show_marketing_insights(df):
    st.subheader("📈 Market Intelligence – Top 5 Insights")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x=TARGET_COL, color=TARGET_COL, title="1️⃣ Subscription Interest Mix")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Q1_Age_Group", color=TARGET_COL, barmode="group",
                           title="2️⃣ Age vs Subscription Interest")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(df, x="Q2_Gender", color=TARGET_COL, barmode="group",
                           title="3️⃣ Gender vs Subscription Interest")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(df, x="Q8_Coffee_Frequency", color=TARGET_COL, barmode="group",
                           title="4️⃣ Coffee Frequency (Repeat Behaviour)")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="Q18_Shopping_Channels", color=TARGET_COL,
                       title="5️⃣ Preferred Shopping Channels")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Quick Marketing Answers")
    yes_segment = (
        df[df[TARGET_COL] == "Yes, very interested"]["Q1_Age_Group"]
        .value_counts()
        .idxmax()
    )
    st.write(f"**1. Largest target demographic:** Age group **{yes_segment}** shows the highest 'Yes, very interested' volume.")
    st.write("**2. Likely repeat buyers:** Customers with **high coffee frequency** ('Daily' / '1–2 times per day') and 'Yes' in subscription interest.")
    maybe_share = (
        df[df[TARGET_COL] == "Maybe"]["Q1_Age_Group"]
        .value_counts(normalize=True)
        .sort_values(ascending=False)
    )
    top_maybe_age = maybe_share.index[0]
    st.write(f"**3. Expansion opportunity:** Age group **{top_maybe_age}** has the highest share of 'Maybe' – ideal for targeted promos & trials.")


def show_ml_lab(df):
    st.subheader("🧪 ReBrew AI Lab – Classification, Clustering, Rules & Regression")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📌 Classification", "🧬 Clustering", "📎 Association Rules", "📉 Regression"]
    )

    with tab1:
        all_models = list(build_classification_models().keys())
        selected = st.multiselect("Select algorithms", all_models, default=all_models)
        run = st.button("🚀 Run classification experiment")
        if run and selected:
            with st.spinner("Training models..."):
                metrics_df, roc_info, pipelines = train_and_eval_classifiers(df, selected)
            st.session_state["cls_metrics"] = metrics_df
            st.session_state["roc_info"] = roc_info
            st.session_state["pipelines"] = pipelines
            st.success("Classification complete!")
        if "cls_metrics" in st.session_state:
            st.markdown("#### Performance table (test set)")
            st.dataframe(st.session_state["cls_metrics"].style.format("{:.3f}"))
            st.markdown("#### ROC curve comparison")
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = {"Decision Tree": "tab:blue", "Random Forest": "tab:green", "Gradient Boosting": "tab:red"}
            for name, info in st.session_state["roc_info"].items():
                ax.plot(info["fpr"], info["tpr"], label=f"{name} (AUC={info['auc']:.3f})", color=colors.get(name, None))
            ax.plot([0, 1], [0, 1], "k--", label="No Skill")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve – Subscription Models")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Run an experiment to see accuracy, precision, recall & F1-score.")

    with tab2:
        n_clusters = st.slider("Number of clusters", 2, 6, 3)
        if st.button("🔍 Run clustering"):
            with st.spinner("Building clusters..."):
                df_clust = run_clustering(df, n_clusters=n_clusters)
            st.session_state["clusters"] = df_clust
            st.success("Clusters ready!")
        if "clusters" in st.session_state:
            df_clust = st.session_state["clusters"]
            st.markdown("#### Cluster sizes & subscription mix")
            summary = (
                df_clust.groupby("Cluster")[TARGET_COL]
                .value_counts(normalize=True)
                .rename("Share")
                .mul(100)
                .reset_index()
            )
            st.dataframe(summary)
            fig = px.scatter(
                df_clust,
                x="PC1",
                y="PC2",
                color="Cluster",
                hover_data=[TARGET_COL, "Q1_Age_Group", "Q8_Coffee_Frequency"],
                title="Customer clusters (PCA projection)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run clustering to see customer DNA segments.")

    with tab3:
        if st.button("🧾 Generate association rules"):
            with st.spinner("Mining frequent patterns..."):
                rules = run_association_rules(df)
            st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
            st.caption("Look for consequents containing Subscription Interest = Yes/Maybe to see strong drivers.")

    with tab4:
        if st.button("📊 Run regression comparison"):
            with st.spinner("Training regression models..."):
                reg_df = run_regression_lab(df)
            st.session_state["reg_metrics"] = reg_df
            st.success("Regression complete!")
        if "reg_metrics" in st.session_state:
            st.dataframe(st.session_state["reg_metrics"].style.format("{:.3f}"))
        else:
            st.info("Click the button to compare Linear, Ridge and Lasso regression.")


def show_prediction_studio(df):
    st.subheader("🔮 Prediction Studio – Score a New ReBrew Customer")
    if "pipelines" not in st.session_state:
        st.info("First run a classification experiment in the ReBrew AI Lab tab.")
        return
    pipelines = st.session_state["pipelines"]
    model_name = st.selectbox("Choose model", list(pipelines.keys()))
    model = pipelines[model_name]
    X_all = df.drop(columns=DROP_COLS + [TARGET_COL])
    cols = X_all.columns.tolist()

    st.markdown("#### Enter customer details")
    with st.form("predict_form"):
        input_data = {}
        for col in cols:
            vals = sorted([str(v) for v in df[col].dropna().unique().tolist()])
            if 0 < len(vals) <= 25:
                input_data[col] = st.selectbox(col, options=vals, index=0)
            else:
                demo = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else ""
                input_data[col] = st.text_input(col, value=demo)
        submitted = st.form_submit_button("✨ Predict label")
    if submitted:
        new_row = pd.DataFrame([input_data])
        pred = model.predict(new_row)[0]
        st.success(f"Predicted Subscription Interest: **{pred}**")
        full_X = df.drop(columns=DROP_COLS + [TARGET_COL])
        full_pred = model.predict(full_X)
        df_with_pred = df.copy()
        df_with_pred[f"Predicted_{TARGET_COL}_{model_name.replace(' ', '_')}"] = full_pred
        st.markdown("##### Preview of scored data")
        st.dataframe(df_with_pred.tail(10))
        csv = df_with_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download full dataset with predictions",
            data=csv,
            file_name="rebrew_subscription_predictions.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    with st.sidebar:
        st.markdown("## ☕ ReBrew Intelligence")
        st.write("AI-Powered Subscription Dashboard")
        st.markdown("---")
        nav = st.radio(
            "Navigate dashboard",
            [
                "🏠 Executive Summary",
                "📊 Market Intelligence (EDA)",
                "🧪 ReBrew AI Lab",
                "🔮 Prediction Studio",
            ],
        )
        st.markdown("---")
        st.markdown("**Team ReBrew**")
        for m in TEAM_MEMBERS:
            st.write(m)
        st.markdown("---")
        k = compute_kpis(df)
        st.caption("Quick conversion snapshot")
        st.write(f"✅ High-intent: **{k['yes_rate']*100:.1f}%**")
        st.write(f"🤝 Maybe: **{k['maybe_rate']*100:.1f}%**")
        st.write(f"🚫 No: **{k['no_rate']*100:.1f}%**")

    if nav == "🏠 Executive Summary":
        show_executive_summary(df)
    elif nav == "📊 Market Intelligence (EDA)":
        show_marketing_insights(df)
    elif nav == "🧪 ReBrew AI Lab":
        show_ml_lab(df)
    elif nav == "🔮 Prediction Studio":
        show_prediction_studio(df)


if __name__ == "__main__":
    main()

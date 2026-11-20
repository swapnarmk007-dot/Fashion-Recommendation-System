from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Fashion Product Recommender & Analytics",
    page_icon="👗",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "styles.csv"
IMAGES_DIR = BASE_DIR / "images" if (BASE_DIR / "images").exists() else BASE_DIR
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@st.cache_data
def index_images(images_dir: Path) -> dict[str, str]:
    """Index product images by filename stem (product ID) inside the folder."""
    image_map: dict[str, str] = {}
    if not images_dir.exists() or not images_dir.is_dir():
        return image_map

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = path.stem.strip()
            if stem not in image_map:
                image_map[stem] = str(path)
    return image_map


def find_product_image(product_id, image_map: dict[str, str], images_dir: Path) -> str | None:
    pid_str = str(product_id).strip()
    if pid_str.endswith(".0"):
        pid_str = pid_str[:-2]

    if pid_str in image_map:
        return image_map[pid_str]

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{pid_str}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def show_product_image(product_id, image_map: dict[str, str], images_dir: Path, caption: str | None = None):
    img_path = find_product_image(product_id, image_map, images_dir)
    if img_path:
        st.image(img_path, caption=caption, width="stretch")
    else:
        st.caption("🖼️ No image found")


@st.cache_data
def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df = df.drop_duplicates().copy()
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    for col in [
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "usage",
        "productDisplayName",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df["combined_features"] = (
        df["gender"].astype(str)
        + " "
        + df["masterCategory"].astype(str)
        + " "
        + df["subCategory"].astype(str)
        + " "
        + df["articleType"].astype(str)
        + " "
        + df["baseColour"].astype(str)
        + " "
        + df["season"].astype(str)
        + " "
        + df["usage"].astype(str)
        + " "
        + df["productDisplayName"].astype(str)
    )

    return df.reset_index(drop=True)


@st.cache_resource
def build_recommender_models(combined_texts: pd.Series):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15000)
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    nn_model = NearestNeighbors(n_neighbors=15, metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)
    return vectorizer, tfidf_matrix, nn_model


def get_recommendations_nn(df: pd.DataFrame, item_idx: int, tfidf_matrix, nn_model, top_n: int = 8) -> pd.DataFrame:
    distances, indices = nn_model.kneighbors(tfidf_matrix[item_idx], n_neighbors=top_n + 1)
    rec_rows = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == item_idx:
            continue
        similarity = round(float(1 - dist), 4)
        row = df.iloc[idx].to_dict()
        row["similarity_score"] = similarity
        rec_rows.append(row)
        if len(rec_rows) >= top_n:
            break
    return pd.DataFrame(rec_rows)


def get_recommendations_cosine_subset(subset_df: pd.DataFrame, selected_idx_in_subset: int, top_n: int = 8) -> pd.DataFrame:
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(subset_df["combined_features"])
    sim_scores = cosine_similarity(mat[selected_idx_in_subset], mat).flatten()
    ranked_indices = np.argsort(-sim_scores)

    rec_rows = []
    for idx in ranked_indices:
        if idx == selected_idx_in_subset:
            continue
        row = subset_df.iloc[idx].to_dict()
        row["similarity_score"] = round(float(sim_scores[idx]), 4)
        rec_rows.append(row)
        if len(rec_rows) >= top_n:
            break
    return pd.DataFrame(rec_rows)


# --- Load Permanent Dataset & Images ---
if not CSV_PATH.exists():
    st.error(f"Dataset not found at: {CSV_PATH}")
    st.info("Place `styles.csv` inside the same folder as `app.py` and refresh.")
    st.stop()

df = load_and_clean_data(CSV_PATH)
image_map = index_images(IMAGES_DIR)

# --- Sidebar: Catalog Filters Only ---
st.sidebar.title("🎛️ Catalog Filters")
st.sidebar.caption(f"Products: {len(df):,} • Images indexed: {len(image_map):,}")

gender_options = ["All"] + sorted(df["gender"].unique().tolist())
category_options = ["All"] + sorted(df["masterCategory"].unique().tolist())
season_options = ["All"] + sorted(df["season"].unique().tolist())

selected_gender = st.sidebar.selectbox("Gender", gender_options)
selected_category = st.sidebar.selectbox("Master Category", category_options)
selected_season = st.sidebar.selectbox("Season", season_options)

filtered_df = df.copy()
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["masterCategory"] == selected_category]
if selected_season != "All":
    filtered_df = filtered_df[filtered_df["season"] == selected_season]

filtered_df = filtered_df.reset_index(drop=True)

if filtered_df.empty:
    st.warning("No products match the current filter selection. Try broadening your filters.")
    st.stop()

# --- Main App ---
st.title("👗 Fashion Product Recommendation & Catalog Explorer")
st.caption("Content-based fashion recommendations with local product imagery and catalog analytics.")

tab1, tab2, tab3 = st.tabs(["🎯 Product Recommender", "🔍 Catalog Search", "📊 Exploratory Data Analysis"])

# --- Tab 1: Recommender ---
with tab1:
    st.subheader("Find Similar Fashion Items")

    product_labels = [
        f"[{row['id']}] {row['productDisplayName']} ({row['articleType']} | {row['baseColour']})"
        for _, row in filtered_df.iterrows()
    ]

    col_sel, col_n = st.columns([3, 1])
    with col_sel:
        chosen_label = st.selectbox("Select a reference product", options=product_labels)
    with col_n:
        top_n = st.slider("Number of recommendations", min_value=3, max_value=12, value=6)

    selected_pos = product_labels.index(chosen_label)
    ref_item = filtered_df.iloc[selected_pos]

    st.markdown("### Selected Reference Item")
    ref_img_col, ref_info_col = st.columns([1, 3])
    with ref_img_col:
        show_product_image(
            ref_item["id"],
            image_map,
            IMAGES_DIR,
            caption=f"ID: {ref_item['id']}",
        )
    with ref_info_col:
        st.markdown(f"**{ref_item['productDisplayName']}**")
        m_cols = st.columns(4)
        m_cols[0].metric("Gender", ref_item["gender"])
        m_cols[1].metric("Category", ref_item["subCategory"])
        m_cols[2].metric("Article Type", ref_item["articleType"])
        m_cols[3].metric("Base Colour", ref_item["baseColour"])

    if st.button("Generate Recommendations", type="primary"):
        if len(filtered_df) == len(df):
            _, tfidf_matrix, nn_model = build_recommender_models(df["combined_features"])
            full_idx = df.index[df["id"] == ref_item["id"]].tolist()[0]
            recs = get_recommendations_nn(df, full_idx, tfidf_matrix, nn_model, top_n=top_n)
        else:
            recs = get_recommendations_cosine_subset(filtered_df, selected_pos, top_n=top_n)

        st.markdown("### Recommended Items")
        cards_per_row = 3
        rec_records = recs.to_dict(orient="records")

        for start_idx in range(0, len(rec_records), cards_per_row):
            row_items = rec_records[start_idx : start_idx + cards_per_row]
            cols = st.columns(cards_per_row)
            for col, item in zip(cols, row_items):
                with col:
                    with st.container(border=True):
                        show_product_image(item["id"], image_map, IMAGES_DIR)
                        st.markdown(f"**{item['productDisplayName']}**")
                        st.caption(
                            f"ID: `{item['id']}` • Similarity: **{item['similarity_score']}**\n\n"
                            f"{item['articleType']} | {item['baseColour']} | {item['gender']}"
                        )

# --- Tab 2: Catalog Search ---
with tab2:
    st.subheader("Search & Browse Products")
    search_query = st.text_input(
        "Search by product name, colour, article type, or usage",
        placeholder="e.g., Navy Blue Shirt, Titan Watch, Casual",
    )

    search_df = filtered_df
    if search_query.strip():
        q = search_query.strip().lower()
        mask = search_df["combined_features"].str.lower().str.contains(q, na=False)
        search_df = search_df[mask]

    st.write(f"Showing **{len(search_df):,}** matching items (displaying top 24 visual cards)")
    preview_items = search_df.head(24).to_dict(orient="records")
    cards_per_row = 4

    for start_idx in range(0, len(preview_items), cards_per_row):
        row_items = preview_items[start_idx : start_idx + cards_per_row]
        cols = st.columns(cards_per_row)
        for col, item in zip(cols, row_items):
            with col:
                with st.container(border=True):
                    show_product_image(item["id"], image_map, IMAGES_DIR)
                    st.markdown(f"**{item['productDisplayName']}**")
                    st.caption(f"ID: `{item['id']}` • {item['articleType']} • {item['baseColour']}")

# --- Tab 3: EDA ---
with tab3:
    st.subheader("Dataset Overview & Distributions")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Rows", f"{len(filtered_df):,}")
    m2.metric("Unique Article Types", f"{filtered_df['articleType'].nunique():,}")
    m3.metric("Unique Colours", f"{filtered_df['baseColour'].nunique():,}")
    m4.metric("Top Season", filtered_df["season"].mode()[0] if not filtered_df["season"].empty else "N/A")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top 10 Article Types**")
        top_articles = filtered_df["articleType"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_articles.values, y=top_articles.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Article Type")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Top 10 Base Colours**")
        top_colours = filtered_df["baseColour"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_colours.values, y=top_colours.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Base Colour")
        st.pyplot(fig)
        plt.close(fig)
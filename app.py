"""
Análisis de Discografía y Letras — Web App
Asignatura: Introducción a Sistemas Cognitivos

Apartado 1: filtrar la discografía de un artista para mostrar solo sus
álbumes de estudio.
Apartado 2: comparar el estilo lírico de varios artistas.

Fuentes: API pública de lyrics.ovh (sin autenticación).
"""

import re
import time
from collections import Counter

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud


# ---------------------------------------------------------------------------
# Configuración general (tema oscuro)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Análisis de Letras",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para tema oscuro y acentos rojos
st.markdown("""
<style>
    /* Fondo principal oscuro */
    .stApp {
        background-color: #0f0f0f;
        color: #f1f1f1;
    }
    /* Sidebar más oscuro */
    section[data-testid="stSidebar"] {
        background-color: #030303 !important;
    }
    /* Títulos en blanco */
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    /* Acento rojo (estilo YouTube Music) en botones */
    .stButton > button[kind="primary"] {
        background-color: #ff0033 !important;
        color: white !important;
        border: none !important;
        border-radius: 18px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #cc0029 !important;
    }
    /* Botones secundarios */
    .stButton > button {
        background-color: #272727 !important;
        color: #f1f1f1 !important;
        border: 1px solid #333 !important;
        border-radius: 18px !important;
    }
    /* Inputs y textareas */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background-color: #121212 !important;
        color: #f1f1f1 !important;
        border: 1px solid #303030 !important;
    }
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #aaaaaa !important;
    }
    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #aaaaaa !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f1f1f;
        border-radius: 18px;
        padding: 6px 16px;
        color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f0f0f !important;
    }
    /* Tablas */
    .stDataFrame {
        background-color: #1f1f1f !important;
    }
    /* Cards de álbumes */
    .album-card {
        background-color: #1f1f1f;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        text-align: center;
        transition: background-color 0.2s;
    }
    .album-card:hover {
        background-color: #2a2a2a;
    }
    .album-card img {
        border-radius: 6px;
        width: 100%;
        margin-bottom: 8px;
    }
    .album-title {
        color: #ffffff;
        font-weight: 600;
        font-size: 14px;
        margin: 4px 0;
        line-height: 1.2;
    }
    .album-meta {
        color: #aaaaaa;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Tema de seaborn coherente con el fondo oscuro
plt.rcParams.update({
    "axes.facecolor": "#0f0f0f",
    "figure.facecolor": "#0f0f0f",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#f1f1f1",
    "text.color": "#f1f1f1",
    "xtick.color": "#f1f1f1",
    "ytick.color": "#f1f1f1",
    "grid.color": "#222",
    "axes.grid": True,
})
sns.set_palette("Set2")


st.title("🎵 Análisis de Discografía y Letras")
st.caption(
    "Práctica ISC | Filtrado automático de álbumes de estudio + análisis comparativo "
    "de letras (lyrics.ovh)"
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
NON_STUDIO_KEYWORDS = [
    "live", "en vivo", "en directo", "unplugged", "mtv",
    "greatest hits", "best of", "the very best", "hits",
    "anthology", "collection", "compilation", "essentials",
    "acoustic", "remix", "remixes", "karaoke", "tribute", "covers",
    "soundtrack", "ost", "b-sides", "demos", "sessions",
    "singles", "complete", "rarities",
    "my worlds", "never say never", "journals",
]

MIN_TRACKS_PER_ALBUM = 4

STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","at","by","for","with",
    "is","it","its","i","you","he","she","we","they","me","my","your","our",
    "be","am","are","was","were","been","being","have","has","had","do","does",
    "did","will","would","can","could","should","this","that","these","those",
    "so","if","as","not","no","yes","oh","ah","la","de","que","y","el","en",
    "don't","i'm","it's","you're","there","then","now","just","all","any",
    "about","from","up","out","down","into","over","like","what","when","where",
    "who","why","how","got","get","go","going","gonna","wanna","yo","hey",
}


# ---------------------------------------------------------------------------
# Llamadas a lyrics.ovh
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_suggest(query, max_retries=3, retry_delay=2):
    """Endpoint /suggest: lista canciones del artista con metadatos de Deezer."""
    url = f"https://api.lyrics.ovh/suggest/{query}"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json().get("data", [])
        except requests.RequestException:
            pass
        if attempt < max_retries:
            time.sleep(retry_delay)
    return []


def clean_track_title(title):
    t = re.sub(
        r"\s*-\s*(remaster(ed)?|live|mono|stereo|remix|acoustic|version|\d{4}).*$",
        "", title, flags=re.I,
    )
    t = re.sub(
        r"\(.*?(feat|featuring|with|remix|live|version|acoustic).*?\)",
        "", t, flags=re.I,
    )
    return t.strip()


@st.cache_data(show_spinner=False)
def get_lyrics(artist, title, max_retries=3, retry_delay=2, timeout=8):
    """Endpoint /v1/{artist}/{title}: letra completa con reintentos."""
    clean = clean_track_title(title)
    url = f"https://api.lyrics.ovh/v1/{artist}/{clean}"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                lyrics = (r.json().get("lyrics") or "").strip()
                return lyrics or None
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        if attempt < max_retries:
            time.sleep(retry_delay)
    return None


# ---------------------------------------------------------------------------
# Lógica de filtrado y métricas
# ---------------------------------------------------------------------------
def normalize_album(name):
    n = re.sub(r"\(.*?\)|\[.*?\]", "", name).strip().lower()
    n = re.sub(r"\s+", " ", n)
    n = re.sub(
        r"\b(deluxe|remaster(ed)?|expanded|edition|version|anniversary|complete|\d{4})\b.*$",
        "", n,
    ).strip()
    return n


def is_studio(album_name, album_type):
    if album_type and album_type != "album":
        return False
    return not any(k in album_name.lower() for k in NON_STUDIO_KEYWORDS)


def metrics(text):
    if not text:
        return 0, 0, 0.0
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
    return len(words), len(set(words)), (len(set(words)) / len(words)) if words else 0.0


@st.cache_data(show_spinner=False)
def fetch_artist_discography(artist_name, extra_queries):
    """Apartado 1: descarga toda la discografía y aplica los filtros."""
    raw = fetch_suggest(artist_name)
    for q in extra_queries:
        raw += fetch_suggest(q)
        time.sleep(0.2)

    raw = [d for d in raw if d.get("artist", {}).get("name", "").lower() == artist_name.lower()]

    if not raw:
        return pd.DataFrame(), pd.DataFrame()

    tracks_raw = pd.DataFrame([
        {
            "track": d["title"],
            "album": d["album"]["title"],
            "album_type": d["album"].get("type", ""),
            "duration": d["duration"],
            "cover": d["album"].get("cover_medium") or d["album"].get("cover", ""),
        }
        for d in raw
    ]).drop_duplicates(subset=["track", "album"]).reset_index(drop=True)

    # Filtro
    tracks = tracks_raw.copy()
    tracks = tracks[tracks.apply(lambda r: is_studio(r["album"], r["album_type"]), axis=1)]
    tracks["album_norm"] = tracks["album"].apply(normalize_album)
    sizes = tracks.groupby("album_norm").size()
    valid = sizes[sizes >= MIN_TRACKS_PER_ALBUM].index
    tracks = tracks[tracks["album_norm"].isin(valid)]

    counts = tracks.groupby(["album_norm", "album"]).size().reset_index(name="cnt")
    canon = (
        counts.sort_values("cnt", ascending=False)
        .drop_duplicates("album_norm")
        .set_index("album_norm")["album"]
        .to_dict()
    )
    tracks = tracks[tracks.apply(lambda r: r["album"] == canon[r["album_norm"]], axis=1)]
    tracks["album"] = tracks["album_norm"].map(canon)
    tracks = tracks.drop(columns=["album_norm"]).reset_index(drop=True)

    return tracks_raw, tracks


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎧 Navegación")
    SECTION = st.radio(
        "",
        ["📀 Apartado 1: Filtrado", "📊 Apartado 2: Comparativa"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa'>Datos: lyrics.ovh + Deezer.<br>"
        "La descarga de letras puede tardar varios minutos la primera vez; "
        "los resultados se cachean para visitas posteriores.</small>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# APARTADO 1
# ===========================================================================
if "Apartado 1" in SECTION:

    st.header("📀 Apartado 1: filtrado de álbumes de estudio")
    st.markdown(
        "Para un artista dado, lyrics.ovh/Deezer devuelve numerosos lanzamientos: "
        "álbumes de estudio, ediciones deluxe, recopilatorios, álbumes en directo, "
        "remixes, etc. Esta aplicación aplica un filtro combinado (tipo de álbum + "
        "lista negra de palabras + umbral mínimo de pistas + deduplicación) para "
        "mostrar únicamente los álbumes de estudio. Al pulsar una portada se "
        "muestran las canciones del álbum y se puede consultar su letra completa."
    )

    col_a, col_b = st.columns([2, 3])
    with col_a:
        artist = st.text_input("Artista", value="Justin Bieber")
    with col_b:
        extra_input = st.text_area(
            "Búsquedas adicionales (una por línea, opcional)",
            value="My World 2.0 Justin Bieber\nUnder the Mistletoe Justin Bieber\n"
                  "Believe Justin Bieber\nPurpose Justin Bieber\nChanges Justin Bieber\n"
                  "Justice Justin Bieber\nSwag Justin Bieber",
            height=100,
        )

    extra_queries = [q.strip() for q in extra_input.split("\n") if q.strip()]

    # Estado de sesión: artista buscado y álbum seleccionado
    if "search_done" not in st.session_state:
        st.session_state.search_done = False
    if "selected_album" not in st.session_state:
        st.session_state.selected_album = None

    if st.button("🔍 Buscar discografía", type="primary"):
        with st.spinner(f"Consultando lyrics.ovh para {artist}..."):
            tracks_raw, tracks = fetch_artist_discography(artist, extra_queries)
        st.session_state.tracks_raw = tracks_raw
        st.session_state.tracks = tracks
        st.session_state.search_artist = artist
        st.session_state.search_done = True
        st.session_state.selected_album = None  # reset al buscar de nuevo

    # Mostrar resultados si ya se ha buscado
    if st.session_state.search_done:
        tracks_raw = st.session_state.tracks_raw
        tracks = st.session_state.tracks
        artist = st.session_state.search_artist

        if tracks_raw.empty:
            st.error(f"No se han encontrado resultados para '{artist}'.")
        else:
            # Métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Lanzamientos brutos", tracks_raw["album"].nunique())
            m2.metric("Pistas brutas", len(tracks_raw))
            m3.metric("Álbumes de estudio", tracks["album"].nunique())
            m4.metric("Pistas en estudio", len(tracks))

            st.markdown("---")
            st.subheader("💿 Álbumes de estudio detectados")
            st.caption("Pulsa una portada para ver sus canciones")

            studio_albums_info = (
                tracks.groupby("album")
                .agg(pistas=("track", "count"), cover=("cover", "first"))
                .reset_index()
                .sort_values("album")
            )

            # Galería de portadas con botones
            n_cols = 4
            for row_start in range(0, len(studio_albums_info), n_cols):
                cols = st.columns(n_cols)
                row_albums = studio_albums_info.iloc[row_start:row_start + n_cols]
                for col, (_, alb) in zip(cols, row_albums.iterrows()):
                    with col:
                        cover_url = alb["cover"] if alb["cover"] else None
                        # Imagen del álbum
                        if cover_url:
                            st.image(cover_url, use_container_width=True)
                        # Botón debajo con el nombre y nº pistas
                        is_selected = st.session_state.selected_album == alb["album"]
                        button_label = (
                            f"✓ {alb['album']}" if is_selected else f"{alb['album']}"
                        )
                        if st.button(
                            f"{button_label}\n{alb['pistas']} pistas",
                            key=f"album_btn_{alb['album']}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                        ):
                            st.session_state.selected_album = alb["album"]
                            st.session_state.selected_track = None
                            st.rerun()

            # Si hay un álbum seleccionado, mostrar canciones
            if st.session_state.selected_album:
                st.markdown("---")
                st.subheader(f"🎵 Canciones de: {st.session_state.selected_album}")

                album_tracks = tracks[tracks["album"] == st.session_state.selected_album].copy()

                # Selector de canción
                track_options = album_tracks["track"].tolist()
                selected_track = st.selectbox(
                    "Selecciona una canción para ver su letra",
                    options=["— elige una canción —"] + track_options,
                )

                if selected_track and selected_track != "— elige una canción —":
                    with st.spinner("Descargando letra..."):
                        lyrics = get_lyrics(artist, selected_track)

                    if lyrics:
                        # Métricas de la letra
                        n, u, lex = metrics(lyrics)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Palabras totales", n)
                        c2.metric("Palabras únicas", u)
                        c3.metric("Riqueza léxica", f"{lex:.3f}")

                        st.markdown("##### Letra")
                        st.markdown(
                            f'<div style="background-color:#1f1f1f;padding:20px;'
                            f'border-radius:8px;white-space:pre-wrap;color:#f1f1f1;'
                            f'max-height:500px;overflow-y:auto;font-size:14px;'
                            f'line-height:1.6">{lyrics}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(
                            "No se ha podido recuperar la letra de esta canción "
                            "en lyrics.ovh."
                        )

            # Tablas comparativas al final
            st.markdown("---")
            with st.expander("Ver tablas brutas (antes / después del filtro)"):
                t1, t2 = st.tabs(["Antes del filtro", "Después del filtro"])
                with t1:
                    st.dataframe(
                        tracks_raw["album"].value_counts().rename_axis("álbum").reset_index(name="pistas"),
                        use_container_width=True, height=400,
                    )
                with t2:
                    st.dataframe(
                        tracks.groupby("album").size().reset_index(name="pistas"),
                        use_container_width=True, height=400,
                    )


# ===========================================================================
# APARTADO 2
# ===========================================================================
else:

    st.header("📊 Apartado 2: comparativa lírica entre artistas")
    st.markdown(
        "Para cada artista se descargan las letras de un sample de canciones "
        "populares y se calculan métricas léxicas y de sentimiento."
    )

    DEFAULT_TRACKS = {
        "Justin Bieber": "Baby, Sorry, Love Yourself, What Do You Mean, Yummy, Holy, Peaches, Stay, Boyfriend, Beauty And A Beat, As Long As You Love Me, All Around The World, Mistletoe, Where Are U Now, Company, Cold Water, Let Me Love You, Friends, 10000 Hours, Intentions",
        "Taylor Swift": "Love Story, You Belong With Me, Shake It Off, Blank Space, Bad Blood, Wildest Dreams, Style, Look What You Made Me Do, Delicate, ME, Lover, Cardigan, Willow, Anti-Hero, Cruel Summer, Lavender Haze, Bejeweled, Karma, All Too Well, I Knew You Were Trouble",
        "Katy Perry": "I Kissed A Girl, Hot N Cold, California Gurls, Teenage Dream, Firework, E.T., Last Friday Night, Part Of Me, Wide Awake, Roar, Dark Horse, Birthday, This Is How We Do, Chained To The Rhythm, Bon Appetit, Swish Swish, Never Really Over, Daisies, Smile, Harleys In Hawaii",
    }

    st.subheader("🎤 Artistas y canciones")
    artists_tracks = {}
    for default_artist, default_songs in DEFAULT_TRACKS.items():
        col1, col2 = st.columns([1, 4])
        with col1:
            artist_name = st.text_input("Artista", value=default_artist, key=f"name_{default_artist}")
        with col2:
            songs_str = st.text_area(
                "Canciones (separadas por comas)",
                value=default_songs,
                key=f"songs_{default_artist}",
                height=80,
            )
        if artist_name and songs_str:
            artists_tracks[artist_name] = [s.strip() for s in songs_str.split(",") if s.strip()]

    if st.button("🚀 Descargar letras y analizar", type="primary"):

        if not artists_tracks:
            st.error("Define al menos un artista con canciones.")
            st.stop()

        progress = st.progress(0.0, text="Descargando letras...")
        rows = []
        total = sum(len(v) for v in artists_tracks.values())
        done = 0
        for artist, songs in artists_tracks.items():
            for title in songs:
                lyr = get_lyrics(artist, title)
                time.sleep(0.2)
                n, u, lex = metrics(lyr)
                rows.append({
                    "artist": artist, "track": title, "lyrics": lyr,
                    "n_words": n, "n_unique": u, "lexical_richness": lex,
                })
                done += 1
                progress.progress(done / total, text=f"Descargando letras... {done}/{total}")

        progress.empty()
        df = pd.DataFrame(rows)

        from textblob import TextBlob
        df["sentiment"] = df["lyrics"].apply(
            lambda t: TextBlob(t).sentiment.polarity if t else None
        )

        found = df["lyrics"].notna().sum()
        st.success(f"✅ Letras encontradas: **{found}/{len(df)}** ({found / len(df):.0%})")

        st.markdown("---")
        st.subheader("📈 Resumen por artista")
        summary = (
            df[df["lyrics"].notna()].groupby("artist").agg(
                canciones=("track", "count"),
                palabras_medias=("n_words", "mean"),
                riqueza_lexica=("lexical_richness", "mean"),
                sentimiento=("sentiment", "mean"),
            ).round(3)
        )
        st.dataframe(summary, use_container_width=True)

        plot_df = df[df["lyrics"].notna()].copy()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔤 Riqueza léxica", "😊 Sentimiento", "📋 Top palabras",
            "☁️ Nubes", "🎯 Huella lírica",
        ])

        with tab1:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.boxplot(data=plot_df, x="artist", y="lexical_richness", ax=ax)
            sns.stripplot(data=plot_df, x="artist", y="lexical_richness", ax=ax, color="white", alpha=0.5, size=4)
            ax.set_ylabel("Riqueza léxica (únicas / totales)")
            ax.set_xlabel("")
            ax.set_title("Riqueza léxica por artista")
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.boxplot(data=plot_df, x="artist", y="sentiment", ax=ax)
            sns.stripplot(data=plot_df, x="artist", y="sentiment", ax=ax, color="white", alpha=0.5, size=4)
            ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
            ax.set_ylabel("Polaridad (-1 negativo, +1 positivo)")
            ax.set_xlabel("")
            ax.set_title("Sentimiento por artista")
            st.pyplot(fig)

        with tab3:
            artists = list(artists_tracks.keys())
            fig, axes = plt.subplots(1, len(artists), figsize=(5 * len(artists), 5))
            if len(artists) == 1:
                axes = [axes]
            for ax, art in zip(axes, artists):
                sub = plot_df[plot_df["artist"] == art]
                words = []
                for txt in sub["lyrics"]:
                    words += [
                        w for w in re.findall(r"[a-zà-ÿ']+", txt.lower())
                        if w not in STOPWORDS and len(w) > 2
                    ]
                top = Counter(words).most_common(10)
                if top:
                    ws, cs = zip(*top)
                    sns.barplot(x=list(cs), y=list(ws), ax=ax, palette="Set2")
                ax.set_title(art)
                ax.set_xlabel("Frecuencia")
                ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)

        with tab4:
            artists = list(artists_tracks.keys())
            fig, axes = plt.subplots(1, len(artists), figsize=(6 * len(artists), 5))
            fig.patch.set_facecolor("#0f0f0f")
            if len(artists) == 1:
                axes = [axes]
            for ax, art in zip(axes, artists):
                sub = plot_df[plot_df["artist"] == art]
                words = []
                for txt in sub["lyrics"]:
                    words += [
                        w for w in re.findall(r"[a-zà-ÿ']+", txt.lower())
                        if w not in STOPWORDS and len(w) > 2
                    ]
                if words:
                    wc = WordCloud(
                        width=600, height=400, background_color="#0f0f0f",
                        colormap="Set2", stopwords=STOPWORDS, collocations=False,
                    ).generate(" ".join(words))
                    ax.imshow(wc, interpolation="bilinear")
                ax.set_title(art, fontsize=14, color="white")
                ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)

        with tab5:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.scatterplot(
                data=plot_df.dropna(subset=["sentiment"]),
                x="sentiment", y="lexical_richness",
                hue="artist", palette="Set2",
                s=80, alpha=0.85, ax=ax,
            )
            ax.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
            ax.set_xlabel("Sentimiento (TextBlob)")
            ax.set_ylabel("Riqueza léxica")
            ax.set_title("Huella lírica: cada punto una canción, color por artista")
            st.pyplot(fig)


st.markdown("---")
st.caption("Práctica ISC · Datos: lyrics.ovh + Deezer · Sin autenticación")

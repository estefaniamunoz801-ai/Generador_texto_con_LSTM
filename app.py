"""
Generador de Texto con LSTM
Aplicacion Streamlit - Curso Agentes de IA e Interfaces Multimodales
"""

import streamlit as st
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="Generador LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #6c757d; font-size: 1rem; }
    .generated-text {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px; padding: 1.5rem;
        font-family: Georgia, serif; font-size: 1.05rem;
        line-height: 1.8; color: #2c3e50;
        border-left: 5px solid #667eea; min-height: 120px;
    }
    .info-box {
        background: #e8f4f8; border-radius: 8px; padding: 1rem;
        border: 1px solid #bee3f8; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Funciones ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)


def is_embedding_model(model):
    """Detecta si el modelo usa Embedding (nuevo) o entrada normalizada (viejo)."""
    first = model.layers[0]
    return hasattr(first, "input_dim") or first.__class__.__name__ == "Embedding"


def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))


def prepare_input(window, char_to_idx, vocab_size, use_embedding):
    indices = [char_to_idx.get(c, 0) for c in window]
    if use_embedding:
        return np.array([indices], dtype=np.int32)
    else:
        x = np.array(indices, dtype=np.float32) / float(vocab_size)
        return x.reshape(1, len(window), 1)


def generate_full_text(model, seed_text, char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars=200, temperature=0.8):
    use_emb = is_embedding_model(model)
    seed_text = seed_text.lower()
    if len(seed_text) < seq_length:
        seed_text = seed_text.rjust(seq_length)
    seed_text = seed_text[-seq_length:]
    seed_text = "".join(c if c in char_to_idx else " " for c in seed_text)

    generated = ""
    window = list(seed_text)

    for _ in range(n_chars):
        x = prepare_input(window[-seq_length:], char_to_idx, vocab_size, use_emb)
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temperature)]
        generated += next_char
        window.append(next_char)

    return generated


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Configuracion")
    st.markdown("---")
    st.markdown("### Cargar Modelo")

    model_file    = st.file_uploader("Modelo (.keras o .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("Metadatos (.json)",    type=["json"])

    st.markdown("---")
    st.markdown("### Parametros de Generacion")

    temperature = st.slider(
        "Temperatura", min_value=0.1, max_value=2.0,
        value=0.8, step=0.05,
        help="Bajo = conservador | Alto = creativo/caotico"
    )

    if temperature < 0.5:
        st.caption("Frio: texto conservador y repetitivo")
    elif temperature < 1.0:
        st.caption("Templado: balance coherencia / variedad")
    elif temperature < 1.4:
        st.caption("Caliente: texto mas creativo")
    else:
        st.caption("Muy caliente: puede inventar palabras")

    n_chars = st.slider("Longitud del texto", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("### Semillas predefinidas")
    seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "con estas razones perdia",
        "el hidalgo tomo la espada",
    ]
    selected_seed = st.selectbox("Elige una semilla:", ["(personalizada)"] + seeds)

    st.markdown("---")
    st.markdown("""<div class="info-box">
    Modelo LSTM entrenado caracter a caracter con el <b>Quijote</b>.
    Genera texto aprendiendo P(siguiente_char | contexto_previo).
    </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">Generador de Texto LSTM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Redes Neuronales Recurrentes · Agentes de IA e Interfaces Multimodales</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Generar Texto", "Explorar Temperatura", "Teoria"])

# ── Tab 1: Generar ────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Texto Semilla")
        default = selected_seed if selected_seed != "(personalizada)" else ""
        seed_input = st.text_area(
            "Escribe el inicio del texto:",
            value=default, height=100,
            placeholder="Ejemplo: en un lugar de la mancha...",
        )
        gen_btn = st.button("Generar Texto", type="primary", use_container_width=True)

        if model_file and metadata_file:
            st.success("Modelo cargado correctamente")
        else:
            st.warning("Sube tu modelo (.keras) y metadatos (.json) para generacion real. "
                       "Sin ellos se mostrara modo demo.")

    with col2:
        st.markdown("### Texto Generado")
        output = st.empty()
        output.markdown(
            '<div class="generated-text"><em style="color:#aaa">El texto aparecera aqui...</em></div>',
            unsafe_allow_html=True
        )

    if gen_btn:
        if not seed_input.strip():
            st.error("Escribe un texto semilla")
        elif model_file and metadata_file:
            # Guardar archivos temporalmente
            ext = "keras" if model_file.name.endswith(".keras") else "h5"
            with open(f"/tmp/model.{ext}", "wb") as f:
                f.write(model_file.read())
            with open("/tmp/metadata.json", "wb") as f:
                f.write(metadata_file.read())

            model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

            if err:
                st.error(f"Error al cargar modelo: {err}")
            else:
                char_to_idx = meta["char_to_idx"]
                idx_to_char = meta["idx_to_char"]
                vocab_size  = meta["vocab_size"]
                seq_length  = meta["seq_length"]

                with st.spinner("Generando..."):
                    texto = generate_full_text(
                        model, seed_input,
                        char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars, temperature
                    )

                output.markdown(
                    f'<div class="generated-text">{texto}</div>',
                    unsafe_allow_html=True
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Caracteres generados", len(texto))
                m2.metric("Temperatura", f"{temperature:.2f}")
                m3.metric("Palabras aprox.", len(texto.split()))
        else:
            # Demo sin modelo
            st.info("Modo Demo - carga tu modelo para generacion real")
            demos = {
                0.3: "en un lugar de la mancha de cuyo nombre no quiero acordarme no ha mucho tiempo que vivia un hidalgo de la mancha",
                0.8: "en un lugar de la mancha donde el sol caia sobre las piedras del camino el caballero alzo su espada y miro al horizonte",
                1.5: "en un lugar de la mancha xilofon caballeo partio con su rocin hacia las montanas del sueno donde los gigantes bailaban",
            }
            closest = min(demos.keys(), key=lambda k: abs(k - temperature))
            output.markdown(
                f'<div class="generated-text">{demos[closest]}</div>',
                unsafe_allow_html=True
            )


# ── Tab 2: Comparar temperaturas ──────────────────────────────────────────────
with tab2:
    st.markdown("### Comparacion de Temperaturas")
    st.markdown("Genera el mismo texto con 5 temperaturas distintas para ver el efecto.")

    compare_seed = st.text_input("Semilla:", value="en un lugar de la mancha de cuyo nombre")
    n_cmp = st.slider("Longitud", 50, 200, 100, key="ncmp")
    cmp_btn = st.button("Comparar", type="primary")

    if cmp_btn and model_file and metadata_file:
        ext = "keras" if model_file.name.endswith(".keras") else "h5"
        with open(f"/tmp/model.{ext}", "wb") as f:
            f.write(model_file.getvalue()
        )
        with open("/tmp/metadata.json", "wb") as f:
            f.write(metadata_file.getvalue())

        model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

        if model and meta:
            temps = [0.3, 0.6, 0.9, 1.2, 1.5]
            cols  = st.columns(len(temps))
            labels = ["Frio", "Templado bajo", "Templado alto", "Caliente", "Muy caliente"]

            with st.spinner("Generando comparacion..."):
                for col, temp, label in zip(cols, temps, labels):
                    text = generate_full_text(
                        model, compare_seed,
                        meta["char_to_idx"], meta["idx_to_char"],
                        meta["seq_length"], meta["vocab_size"],
                        n_cmp, temp
                    )
                    with col:
                        hue = int(240 - temps.index(temp) * 40)
                        st.markdown(f"**T={temp} — {label}**")
                        st.markdown(
                            f'<div style="background:#f8f9fa; border-radius:8px; padding:0.8rem; font-family:Georgia,serif; font-size:0.85rem; line-height:1.6; min-height:150px; border-top:3px solid hsl({hue},70%,55%);">{text}</div>',
                            unsafe_allow_html=True
                        )
    elif cmp_btn:
        st.warning("Necesitas cargar un modelo. Sube .keras y .json en el sidebar.")


# ── Tab 3: Teoria ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Fundamentos Teoricos")

    with st.expander("Que es una RNN?", expanded=True):
        st.markdown("""
Una **Red Neuronal Recurrente** mantiene un estado oculto que actua como memoria:

```
h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
```

Problema: los gradientes se multiplican en cada paso y desaparecen
si son menores a 1 (vanishing gradient).""")

    with st.expander("Por que LSTM resuelve el vanishing gradient?"):
        st.markdown("""
La LSTM introduce tres puertas y un estado de celda `C_t`:

| Puerta | Funcion |
|--------|---------|
| Forget | Que olvidar del pasado? |
| Input  | Que nueva info guardar? |
| Output | Que parte del estado exponer? |

El estado `C_t` actua como una cinta transportadora:
el gradiente fluye sin interrupciones a traves del tiempo.""")

    with st.expander("Como funciona la temperatura?"):
        st.markdown("""
```
p_i = exp(log(p_i) / T) / sum(exp(log(p_j) / T))
```

- **T < 1.0:** distribucion aguda, elige siempre lo mas probable
- **T = 1.0:** distribucion original del modelo
- **T > 1.0:** distribucion plana, mayor variedad y riesgo""")

    with st.expander("RNN vs LSTM vs GRU"):
        st.markdown("""
| | RNN | LSTM | GRU |
|---|---|---|---|
| Memoria larga | Pobre | Excelente | Buena |
| Velocidad CPU | Rapida | Lenta | Media |
| Parametros | Pocos | Muchos | Moderados |""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Generador LSTM - Agentes de IA e Interfaces Multimodales</div>",
    unsafe_allow_html=True
)

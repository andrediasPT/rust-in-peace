"""
PROJETO "RUST IN PEACE" - FASE III
Aplicacao web (Streamlit) para classificar o tipo de dano num veiculo a
partir de uma fotografia, com o modelo guardado em modelo/.

Correr localmente:   streamlit run app.py
Publicar:            Streamlit Community Cloud (ver LEIA-ME_Fase_III.txt)
"""

import os
import io
import json
import time
import hashlib

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image, ImageOps

PASTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo')
LIMIAR_CONFIANCA = 0.50

DESCRICOES = {
    'Amolgadela': 'Deformação da chapa sem rotura.',
    'Farol_Partido': 'Farol ou farolim danificado/partido.',
    'Peca_em_Falta': 'Peça da carroçaria em falta (para-choques, espelho, ...).',
    'Rasgao': 'Rasgão/rotura no material da carroçaria.',
    'Risco': 'Riscos na pintura.',
    'Vidro_Partido': 'Para-brisas ou vidro lateral partido.',
}

st.set_page_config(page_title='Rust in Peace - Danos em veículos', page_icon='🚗', layout='wide')


# ---------------------------------------------------------------
# Modelo e artefactos (carregados uma unica vez por servidor)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner='A carregar o modelo...')
def carregar_modelo():
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from tensorflow import keras
    modelo = keras.models.load_model(os.path.join(PASTA_MODELO, 'modelo_final.keras'), compile=False)
    with open(os.path.join(PASTA_MODELO, 'classes.json'), encoding='utf-8') as f:
        classes = json.load(f)
    with open(os.path.join(PASTA_MODELO, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    t = cfg['tamanho_imagem']
    modelo.predict(np.zeros((1, t, t, 3), np.float32), verbose=0)    # aquecimento
    return modelo, classes, cfg


def preprocessar(dados, tamanho):
    """MESMO pre-processamento do treino (comum.carregar_imagem): rotacao EXIF,
    RGB, resize bilinear para tamanho x tamanho, pixeis 0-255 em float32."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(dados))).convert('RGB')
    x = np.asarray(im.resize((tamanho, tamanho), Image.BILINEAR), dtype=np.float32)
    return im, x[None]


def classificar(modelo, x):
    t0 = time.perf_counter()
    p = modelo.predict(x, verbose=0)[0]
    return p, (time.perf_counter() - t0) * 1000


def grafico_probabilidades(prob, classes, legiveis):
    df = pd.DataFrame({'Classe': [legiveis.get(c, c) for c in classes], 'Probabilidade': prob})
    df['Prevista'] = df['Probabilidade'] == df['Probabilidade'].max()
    df['Texto'] = (100 * df['Probabilidade']).map(lambda v: f'{v:.1f}%')
    base = alt.Chart(df).encode(
        y=alt.Y('Classe:N', sort='-x', title=None),
        x=alt.X('Probabilidade:Q', scale=alt.Scale(domain=[0, 1.18]),
                axis=alt.Axis(format='%', values=[0, 0.25, 0.5, 0.75, 1]), title=None),
        tooltip=['Classe', alt.Tooltip('Probabilidade:Q', format='.1%')])
    barras = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.condition('datum.Prevista', alt.value('#2a78d6'), alt.value('#b9cfee')))
    texto = base.mark_text(align='left', dx=4, color='#52514e').encode(text='Texto:N')
    return (barras + texto).properties(height=34 * len(classes))


# ---------------------------------------------------------------
# Interface
# ---------------------------------------------------------------
try:
    modelo, info_classes, cfg = carregar_modelo()
except Exception as e:
    st.error(f'Não foi possível carregar o modelo de {PASTA_MODELO}: {e}')
    st.stop()

CLASSES = info_classes['classes']
LEGIVEIS = info_classes.get('nomes_legiveis', {})
TAM = cfg['tamanho_imagem']

if 'historico' not in st.session_state:
    st.session_state.historico = []

with st.sidebar:
    st.header('Sobre o modelo')
    met = cfg.get('metricas_teste', {})
    st.markdown(
        f"**Arquitetura:** {cfg.get('backbone', '?')} (ImageNet) + rede densa"
        f"{' + fine-tuning' if cfg.get('fine_tuning_mantido') else ''}  \n"
        f"**Parâmetros:** {cfg.get('parametros_total', 0):,}".replace(',', ' ') + "  \n"
        f"**Accuracy no teste:** {met.get('Accuracy', float('nan')):.1%}  \n"
        f"**F1 macro no teste:** {met.get('F1_macro', float('nan')):.3f}  \n"
        f"**Imagens de teste:** {met.get('N_imagens', '?')}")
    st.subheader('Classes')
    for c in CLASSES:
        st.markdown(f"- **{LEGIVEIS.get(c, c)}** — {DESCRICOES.get(c, '')}")
    st.info('O modelo assume que a fotografia mostra um dano. Não tem a classe "sem dano", '
            'por isso uma imagem sem danos (ou que não seja de um carro) é sempre atribuída '
            'a uma das 6 classes — veja a confiança.')
    if st.button('Limpar histórico', use_container_width=True):
        st.session_state.historico = []

st.title('🚗 Rust in Peace — classificação de danos em veículos')
st.caption('Inteligência Computacional 2026/27 · ISEC · Fase III (transfer learning) · '
           'João Pinheiro e André Dias')

origem = st.radio('Origem da imagem', ['📁 Carregar ficheiro(s)', '📷 Câmara'], horizontal=True)
if origem.startswith('📁'):
    ficheiros = st.file_uploader('Escolha uma ou mais fotografias (JPG, PNG, WEBP, BMP)',
                                 type=['jpg', 'jpeg', 'png', 'webp', 'bmp'], accept_multiple_files=True)
else:
    foto = st.camera_input('Tire uma fotografia ao dano')
    ficheiros = [foto] if foto is not None else []

if not ficheiros:
    st.markdown('Carregue uma imagem ou use a câmara para obter a classificação. '
                'Pode testar várias imagens seguidas: os resultados ficam no histórico abaixo.')

for f in ficheiros:
    dados = f.getvalue()
    try:
        imagem, x = preprocessar(dados, TAM)
    except Exception:
        st.error(f'{f.name}: não é uma imagem válida.')
        continue
    prob, ms = classificar(modelo, x)
    i = int(np.argmax(prob))
    classe, conf = CLASSES[i], float(prob[i])

    st.divider()
    col_img, col_res = st.columns([1, 1.2], gap='large')
    with col_img:
        st.image(imagem, caption=f'{f.name} ({imagem.width}×{imagem.height})', use_container_width=True)
    with col_res:
        st.caption('Classe prevista')
        st.markdown(f"### {LEGIVEIS.get(classe, classe)}")
        st.markdown(f"**{conf:.1%}** de confiança")
        if conf < LIMIAR_CONFIANCA:
            seg = int(np.argsort(prob)[-2])
            st.warning(f'Confiança baixa: o modelo hesita entre **{LEGIVEIS.get(classe, classe)}** e '
                       f'**{LEGIVEIS.get(CLASSES[seg], CLASSES[seg])}** ({prob[seg]:.1%}).')
        st.altair_chart(grafico_probabilidades(prob, CLASSES, LEGIVEIS), use_container_width=True)
        st.caption(f'Tempo de inferência: {ms:.0f} ms')

    chave = hashlib.md5(dados).hexdigest()
    if chave not in {h['_id'] for h in st.session_state.historico}:
        st.session_state.historico.insert(0, {
            '_id': chave, 'Hora': time.strftime('%H:%M:%S'), 'Imagem': f.name,
            'Classe prevista': LEGIVEIS.get(classe, classe), 'Confiança': conf,
            **{LEGIVEIS.get(c, c): float(p) for c, p in zip(CLASSES, prob)}})

if st.session_state.historico:
    st.divider()
    st.subheader(f'Histórico desta sessão ({len(st.session_state.historico)} imagens)')
    hist = pd.DataFrame(st.session_state.historico).drop(columns='_id')
    colunas_pct = [c for c in hist.columns if c not in ('Hora', 'Imagem', 'Classe prevista')]
    st.dataframe(hist, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.ProgressColumn(c, format='%.2f', min_value=0, max_value=1)
                                for c in colunas_pct})

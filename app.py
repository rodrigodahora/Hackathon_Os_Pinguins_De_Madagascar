import re
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# -------------------------
# Extrair artigos
# -------------------------
def extrair_artigos(texto, lei_nome):
    artigos = re.split(r'(Art\.\s*\d+[^\n]*)', texto)
    resultado = []
    for i in range(1, len(artigos), 2):
        titulo = artigos[i].strip()
        texto_artigo = artigos[i+1].strip() if i+1 < len(artigos) else ""
        resultado.append({
            "lei": lei_nome,
            "titulo": titulo,
            "texto": texto_artigo
        })
    return resultado

# -------------------------
# Carregar leis
# -------------------------
def carregar_leis():
    leis = []
    with open("constituicao.txt", "r", encoding="utf-8") as f:
        leis += extrair_artigos(f.read(), "Constituição Federal")
    with open("ctb.txt", "r", encoding="utf-8") as f:
        leis += extrair_artigos(f.read(), "Código de Trânsito Brasileiro")
    return leis

# -------------------------
# Criar índice FAISS
# -------------------------
def criar_indice(leis):
    modelo = SentenceTransformer("./modelos/all-mpnet-base-v2")
    corpus = [f"{a['lei']} {a['titulo']} {a['texto']}" for a in leis]
    embeddings = modelo.encode(corpus, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return modelo, index, leis

# -------------------------
# Buscar artigos
# -------------------------
def buscar_artigo(pergunta, modelo, index, leis, k=3):
    query_emb = modelo.encode([pergunta], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    return [leis[idx] for idx in I[0]]

# -------------------------
# Gerar resposta humanizada dinamicamente
# -------------------------
summarizer = pipeline("summarization", model="./modelos/bart-large-cnn")

def gerar_resposta(resultados, pergunta):
    textos = [f"{r['lei']} - {r['titulo']}: {r['texto']}" for r in resultados]
    concatenado = " ".join(textos)

    # Resumir o conteúdo
    resumo = summarizer(concatenado, max_length=150, min_length=60, do_sample=False)[0]['summary_text']

    # Transformar em frases humanizadas
    resposta_final = (
        f"Baseando-se na legislação relevante, aqui está o que encontramos:\n"
        f"{resumo}\n\n"
        
    )
    return resposta_final

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

print("📚 Carregando legislações...")
leis = carregar_leis()
print(f"✅ {len(leis)} artigos indexados (Constituição + CTB).")
modelo, index, leis = criar_indice(leis)

@app.route("/", methods=["GET", "POST"])
def home():
    resposta = None
    pergunta = None
    if request.method == "POST":
        pergunta = request.form["pergunta"]
        resultados = buscar_artigo(pergunta, modelo, index, leis, k=3)
        resposta = gerar_resposta(resultados, pergunta)
    return render_template("index.html", resposta=resposta, pergunta=pergunta)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)


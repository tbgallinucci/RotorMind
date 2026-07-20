# RAG Vetorial — como foi feito, como usar, e como explicar na entrevista

> Módulo de recuperação por embeddings (`assistant/app/wiki_vector.py`), ativável
> pelo toggle **Lexical / Vector** no cabeçalho do chat. Quando desativado (ou
> quando a dependência opcional não está instalada), o sistema usa o RAG lexical
> que sempre existiu (`wiki_logic.py`) — nada muda no resto do pipeline.

---

## 1. O que foi construído (resumo de 30 segundos)

O RotorMind agora tem **dois retrievers intercambiáveis** sobre o mesmo corpus
(as páginas de `assistant/wiki/`):

| | Lexical (padrão) | Vetorial (toggle "Vector") |
|---|---|---|
| Mecanismo | Pontuação por palavra-chave (tipo BM25 caseiro) | Similaridade de cosseno entre embeddings |
| Modelo | Nenhum — regex + contagem | `all-MiniLM-L6-v2` (sentence-transformers, 384 dim, local) |
| Dependências | Nenhuma extra | `pip install -e ".[vector]"` (traz torch) |
| Ganha quando | A pergunta usa os termos exatos do corpus (identificadores, "Class 600", números) | A pergunta é **parafraseada** ("o filme de óleo se comporta como mola?" acha "stiffness and damping") |
| Falha quando | Sinônimos/paráfrase — zero acerto de palavra = zero score | Identificadores raros que o modelo nunca viu; corpus com jargão fora da distribuição de treino |

Os dois compartilham **chunking, orçamento de tokens e formato de saída** — a
única coisa que muda é o ranking. Isso foi deliberado (ver §3.1).

## 2. Como usar

```bash
# 1. Instalar a dependência opcional (uma vez; baixa torch, ~2 GB)
pip install -e ".[vector]"

# 2. Subir o app normalmente
python -m uvicorn assistant.app.main:app --port 8000
```

- Na UI: o toggle **Lexical | Vector** fica no cabeçalho do chat, ao lado do
  toggle Local/Cloud. Se o extra não estiver instalado, o botão "Vector" aparece
  desabilitado com o motivo no tooltip (mesmo padrão do botão "Cloud" sem chave).
- A escolha persiste em `localStorage` e vale por requisição (`ChatRequest.rag`).
- **Primeira consulta vetorial**: baixa o modelo de embedding (~80 MB) e embeda o
  corpus inteiro — leva alguns segundos. Depois disso, os vetores das seções ficam
  em cache em disco (`wiki/.vector_cache/embeddings.npz`) e só seções novas ou
  alteradas são re-embedadas (ex.: um run de FEA recém-ingerido).
- Modelo customizável via env: `EMBEDDING_MODEL_NAME` (ver `.env.example`) — ex.
  `paraphrase-multilingual-MiniLM-L12-v2` para corpus multilíngue.
- Endpoint de status: `GET /api/rag-status` → `{"vector": bool, "embedding_model", "reason"}`.

### 2.1 Como PROVAR qual retriever respondeu (ótimo para demo ao vivo)

Três níveis de evidência, do visível ao irrefutável:

1. **Selo na resposta (UI)**: cada resposta do chat carrega um selo discreto
   ("vector retrieval · local LLM"). Ele vem do primeiro evento do stream — um
   `{"type": "meta", "rag": ..., "llm": ...}` emitido pelo **servidor** com o
   modo que **realmente** construiu o contexto. Não é eco do toggle: se você
   pedir vetor num servidor sem o extra instalado, o selo diz "lexical",
   porque o fallback degradou. É declaração do backend, não estado do frontend.
2. **Wire (DevTools → Network → `/api/chat`)**: o payload do request mostra o
   que foi *pedido* (`"rag": "vector"`); a primeira linha do response stream
   mostra o que foi *usado* (o evento `meta`). Pedido ≠ usado é exatamente o
   caso que o fallback cobre.
3. **Comportamental**: faça uma pergunta parafraseada sem nenhuma palavra-chave
   do corpus — ex. *"does the oil film behave like a spring?"* ("spring" não
   está nos títulos). No modo vetorial ela recupera `journal-bearing-theory`;
   no lexical, o score de palavras é fraco e o contexto vem diferente. Rodar a
   mesma pergunta nos dois modos, lado a lado, é a demo mais convincente.

> Ponto de entrevista: "o modo usado é **observável no produto** — o servidor
> declara no stream qual retriever construiu o contexto, porque com fallback
> silencioso 'o que foi pedido' e 'o que rodou' podem divergir, e depurar um
> sistema de RAG exige saber a diferença." Isso é observabilidade aplicada a
> LLM ops, e bancas gostam.

## 3. Decisões técnicas (o coração da entrevista)

### 3.1 Mesmo chunking, mesmo contrato de saída — comparação limpa

O retriever vetorial reusa **exatamente** o pipeline de chunking do lexical
(`flatten_tables_generic` → `split_into_sections`, um chunk por seção) e emite o
mesmo formato (`### WIKI PAGE: <slug>` + seção). Por quê:

1. **Comparabilidade científica**: se chunking e formato são idênticos, qualquer
   diferença de resultado entre os modos é atribuível *só ao ranking*. É a versão
   de engenharia de "mude uma variável por vez".
2. **Drop-in**: o system prompt exige citação `(wiki: slug, Seção)` e o guard de
   grounding valida números contra o marker `### WIKI PAGE:`. Como o vetorial
   emite o mesmo contrato, **nem o prompt, nem o agente, nem o guard precisaram
   mudar** — o resto do pipeline não sabe qual retriever montou o contexto.

### 3.2 Embeddings locais, não API

`sentence-transformers/all-MiniLM-L6-v2` roda na máquina (CPU serve; usa torch
por baixo). Coerente com a tese local-first do projeto: dados de engenharia
sensíveis não saem da máquina. Trade-off honesto: um modelo de embedding de API
(ex. `text-embedding-3-small`) é melhor em qualidade, mas cria dependência de
rede/custo e vaza o corpus. Para um corpus técnico pequeno, o MiniLM é suficiente
e o custo marginal é zero.

### 3.3 Busca exata por força bruta, não FAISS — e saber defender isso

A "índice" é uma matriz numpy; a busca é `matriz @ vetor_da_query` (produto
escalar = cosseno, porque tudo é normalizado) seguido de `argsort`. Para um
corpus de **centenas** de seções isso é ótimo e exato:

- ANN (FAISS/HNSW/IVF) troca exatidão por velocidade e só compensa a partir de
  ~10⁵–10⁶ vetores; abaixo disso o overhead do índice supera o ganho.
- A resposta de entrevista: *"eu sei o que o FAISS faz e onde ele entra; a
  fronteira está isolada em duas funções (`_embed_chunks` devolve a matriz,
  `build_context` faz o ranking), então trocar por FAISS/Chroma quando o corpus
  crescer é mudança local, não rearquitetura."*

### 3.4 Cache incremental por hash de conteúdo

Cada chunk tem id = SHA-1 do próprio texto. O cache em disco (`.npz`) guarda
`id → vetor`. Na consulta: chunks cujo hash já está no cache não são re-embedados;
só o delta (ex.: as ~10 seções de um run novo) passa pelo modelo. Se uma seção
muda, o hash muda, e ela é re-embedada automaticamente — **invalidação de cache
de graça**, sem timestamps nem versionamento manual.

### 3.5 Piso de similaridade ≈ fallback do lexical

O lexical descarta seções com score 0 e, sem nenhum match, devolve o índice da
wiki. O vetorial espelha isso com `MIN_SIMILARITY = 0.20`: cosseno nunca é zero
entre textos reais, então sem um piso o retriever encheria o contexto com as
seções "menos irrelevantes" de um corpus inteiro irrelevante. O valor é
permissivo de propósito — o objetivo é cortar ruído óbvio, não segundo-adivinhar
o ranking.

### 3.6 Degradação graciosa em três camadas

Recuperação **nunca** derruba um turno de chat:

1. **Instalação**: sem `sentence-transformers`, `/api/rag-status` reporta
   indisponível e a UI desabilita o botão (mesmo padrão do toggle Cloud sem
   `CLOUD_LLM_API_KEY`).
2. **Requisição**: se um cliente de API pedir `rag: "vector"` sem o extra
   instalado, o servidor silenciosamente usa o lexical (não é erro do usuário).
3. **Query**: se o vetorial falhar em runtime (ex. download do modelo sem
   internet), `search_knowledge` captura e cai para o lexical — que não tem modo
   de falha (é regex + leitura de arquivo).

### 3.7 O modo viaja por closure, não por estado global

O toggle chega ao backend como `ChatRequest.rag` e é amarrado à tool via
`make_tool_dispatch(rag_mode)` — uma tabela de dispatch **por requisição** onde
`search_knowledge` captura o modo numa closure. Alternativas rejeitadas:

- *Variável global/módulo*: dois chats simultâneos com toggles diferentes se
  atropelariam (race condition clássica de estado compartilhado).
- *ContextVar*: funciona, mas é implícito; a closure torna o fluxo do dado
  visível na assinatura (`run_agent(..., tool_dispatch=...)`).

É o mesmo princípio de injeção de dependência já usado para o cliente LLM (que
é injetado para os testes rodarem com um mock).

### 3.8 Embedding roda no threadpool

`build_context` vetorial é CPU-bound (e a primeira chamada carrega o modelo).
No endpoint async do FastAPI ele roda via `run_in_threadpool` para não travar o
event loop — mesma razão pela qual `/api/run` (FEA) é `def` síncrono de
propósito.

### 3.9 Testes sem torch, sem rede, sem download

`build_context(query, budget, embed_fn=None)` aceita um **embedder injetável**.
Os testes (`assistant/tests/test_vector.py`) injetam um fake determinístico
(bag-of-words com hash md5) e cobrem: ranking, contrato de saída, orçamento de
tokens, cache (segunda consulta embeda só a query), incrementalidade (página
nova → só ela é embedada), fallback em indisponibilidade E em erro, roteamento
do dispatch por requisição, e os dois endpoints. Filosofia idêntica ao LLM
mockado: o CI testa **o que é nosso** (mecânica), não a semântica do modelo de
terceiros.

Detalhe fino: um teste do piso de similaridade não pode confiar no fake de hash
(colisões dão similaridade espúria) — ele injeta um embedder que torna a query
**ortogonal por construção** aos chunks. Saber explicar isso mostra que você
entende o que o teste testa.

## 4. Perguntas prováveis de banca — respostas prontas

**"Seu RAG é vetorial?"**
> "Tem os dois modos, com um toggle na UI. O padrão é lexical determinístico —
> para um corpus técnico pequeno, cheio de identificadores e valores exatos tipo
> 'Class 600', busca por palavra-chave é precisa e 100% explicável. O modo
> vetorial usa sentence-transformers local com cache incremental por hash de
> conteúdo, e ganha quando a pergunta é parafraseada em vez de usar os termos do
> documento. Os dois compartilham chunking e contrato de saída, então a
> comparação entre eles isola só o efeito do ranking."

**"Por que não usou FAISS/Chroma/Pinecone?"**
> "Corpus de centenas de seções: busca exata por numpy é mais rápida E mais
> precisa que ANN nessa escala — índice aproximado só paga a partir de ~10⁵
> vetores. A fronteira está isolada em duas funções, então plugar FAISS quando
> escalar é mudança local. Escolhi não adicionar uma dependência de
> infraestrutura para um problema que ainda não existe."

**"Como você invalida o cache de embeddings?"**
> "Hash do conteúdo como chave. Seção mudou → hash mudou → re-embeda sozinha.
> Sem timestamp, sem versão, sem cron."

**"E se o serviço de embedding cair em produção?"**
> "Aqui é local, mas o princípio está implementado: três camadas de degradação —
> status endpoint desabilita o toggle, o servidor degrada requisições pedindo
> vetor sem backend, e falha em runtime cai para o retriever lexical, que não
> tem modo de falha. Recuperação nunca derruba a resposta."

**"Como testa isso sem GPU no CI?"**
> "Embedder injetado por parâmetro, fake determinístico nos testes — mesmo
> padrão do cliente LLM mockado. O CI testa a mecânica que é minha (chunking,
> cache, ranking, fallback), não a qualidade do modelo da HuggingFace."

**"Qual a evolução natural?"**
> "Híbrido: rodar os dois rankings e fundir com Reciprocal Rank Fusion — lexical
> segura os identificadores exatos, vetorial segura a paráfrase. Depois, um
> re-ranker cross-encoder no top-k se a precisão do primeiro estágio virar
> gargalo. E FAISS/Chroma quando o corpus justificar."

## 5. Mapa de arquivos

| Arquivo | Papel |
|---|---|
| `assistant/app/wiki_vector.py` | Retriever vetorial: chunking compartilhado, embeddings, cache, ranking |
| `assistant/app/wiki_logic.py` | Retriever lexical (inalterado) — o fallback universal |
| `assistant/app/tools.py` | `search_knowledge(query, mode)` + `make_tool_dispatch(rag_mode)` |
| `assistant/app/agent.py` | Loop do agente; agora aceita `tool_dispatch` por requisição |
| `assistant/app/main.py` | `ChatRequest.rag`, `/api/rag-status`, seleção do retriever + threadpool |
| `assistant/static/index.html` / `script.js` / `style.css` | Toggle Lexical/Vector (espelho do toggle Local/Cloud) |
| `assistant/tests/test_vector.py` | 10 testes, embeddings 100% fakes |
| `pyproject.toml` | Extra opcional `[vector]` |

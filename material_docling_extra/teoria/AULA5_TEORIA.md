# Aula 5: Docling e Ingestão Inteligente de Documentos
## Estratégias Avançadas de Processamento Documental para Sistemas RAG Jurídicos
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 5 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática  
**Pré-requisito:** Aulas 1–4 concluídas (RAG básico, chunking, embeddings, buscas híbridas)  
**Stack:** Docling ≥2.0 · Apache Tika · PIL · LangChain · BGE-M3 · OpenSearch · vLLM

---

## 1. Motivação e Contexto Jurídico

O processamento de documentos jurídicos e de segurança pública apresenta desafios únicos que vão muito além do que parsers de texto tradicionais conseguem resolver. Pense nas situações reais que um sistema RAG jurídico precisa enfrentar:

- **Laudos periciais escaneados** do Instituto de Criminalística, muitas vezes com timbres, carimbos e manuscritos sobrepostos
- **Acórdãos em PDF** com estruturas complexas: ementas, votos divergentes, tabelas de jurisprudência e citações em formato específico
- **Formulários de boletim de ocorrência** preenchidos à mão ou digitados em campos não-uniformes
- **Relatórios de inteligência** com tabelas de análise de rede criminosa, gráficos e diagramas
- **Legislação** com estrutura hierárquica complexa: artigos, parágrafos, incisos, alíneas

Um sistema que não consiga extrair fielmente o conteúdo desses documentos compromete a qualidade de todo o pipeline RAG. Se o contexto recuperado contiver texto corrompido, colunas misturadas ou tabelas perdidas, o LLM produzirá respostas incorretas — o que em contexto jurídico pode significar erros graves.

### 1.1 O Problema da Ingestão Ingênua

A abordagem mais simples de ingestão — ler o PDF com `pdfplumber` ou `PyPDF2` — falha em cenários críticos:

```
Documento original (PDF com 2 colunas):
┌─────────────────────┬─────────────────────┐
│ Art. 1º O réu foi   │ Art. 2º A pena base │
│ condenado pela       │ foi fixada em 6     │
│ prática de tráfico  │ anos de reclusão     │
└─────────────────────┴─────────────────────┘

Extração ingênua (leitura linha a linha):
"Art. 1º O réu foi Art. 2º A pena base"
"condenado pela foi fixada em 6"
"prática de tráfico anos de reclusão"
```

O resultado é texto incompreensível que invalida qualquer chunking subsequente.

### 1.2 Por que Docling?

O **Docling** (IBM Research, 2024) foi desenvolvido especificamente para resolver esses problemas em documentos corporativos e técnicos complexos. Suas principais capacidades incluem:

- Detecção e reconstituição de layout (colunas, tabelas, figuras)
- OCR integrado com modelos de reconhecimento de layout (DocLayNet)
- Exportação estruturada em Markdown, JSON e DoclingDocument
- Suporte nativo a PDF, DOCX, PPTX, HTML, imagens
- Pipeline configurável com diferentes backends de OCR

> **Referência:** AUER, Peter et al. *Docling Technical Report*. arXiv:2408.09869, 2024. IBM Research.

---

## 2. Arquitetura do Docling

### 2.1 Visão Geral do Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DOCLING                          │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │  Input   │──▶│  Document    │──▶│   Layout Model    │  │
│  │  Files   │   │  Backend     │   │   (DocLayNet)     │  │
│  │ PDF/DOCX │   │  (pdfium,    │   │                   │  │
│  │ HTML/IMG │   │   docx2txt)  │   │  Detecta:         │  │
│  └──────────┘   └──────────────┘   │  - Títulos        │  │
│                                    │  - Parágrafos     │  │
│                                    │  - Tabelas        │  │
│                                    │  - Figuras        │  │
│                                    │  - Listas         │  │
│                                    └────────┬──────────┘  │
│                                             │              │
│  ┌──────────────────┐   ┌──────────────────▼───────────┐  │
│  │  DoclingDocument │◀──│  Table Structure Model +     │  │
│  │  (formato unif.) │   │  OCR Engine (EasyOCR/        │  │
│  │                  │   │  Tesseract/RapidOCR)         │  │
│  └────────┬─────────┘   └──────────────────────────────┘  │
│           │                                                 │
│           ▼                                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │              EXPORTADORES                          │    │
│  │  Markdown  │  JSON  │  DoclingDocument  │  HTML   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Instalação e Configuração Básica

```python
# Instalação completa com suporte a OCR
# pip install docling[ocr] --quiet

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrOptions

# Configuração padrão — adequada para PDFs nativos (texto selecionável)
converter = DocumentConverter()

# Converter um único documento
resultado = converter.convert("acordao_stj.pdf")
doc = resultado.document

# Exportar para Markdown (ideal para chunking posterior)
markdown = doc.export_to_markdown()

# Exportar para JSON estruturado
json_doc = doc.export_to_dict()

print(f"Páginas: {len(doc.pages)}")
print(f"Elementos detectados: {len(list(doc.iterate_items()))}")
```

**Aplicação jurídica:** Esta configuração básica é adequada para acórdãos de tribunais superiores (STJ, STF, TJ) que são publicados como PDFs nativos de texto.

---

## 3. Processamento de PDFs Escaneados com OCR

### 3.1 O Desafio do OCR em Documentos Jurídicos

Documentos como laudos periciais do IC, formulários de BO preenchidos à mão, e cópias digitalizadas de processos físicos requerem OCR. O problema é que OCR genérico falha em:

- **Termos jurídicos latinos** ("fumus boni iuris", "periculum in mora")
- **Números de processo CNJ** com formato específico
- **Timbres e carimbos** sobrepostos ao texto
- **Formulários com campos mistos** (pré-impresso + manuscrito)

### 3.2 Configuração OCR para Documentos Jurídicos

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    EasyOcrOptions,
    TesseractOcrOptions
)

# ─── Opção 1: EasyOCR (melhor para documentos mistos)
ocr_options_easyocr = EasyOcrOptions(
    lang=["pt", "en"],        # Português + inglês para termos latinos
    gpu=False,                 # False para Colab sem GPU
    confidence_threshold=0.5   # Threshold de confiança do OCR
)

# ─── Opção 2: Tesseract (mais leve, pior qualidade)
ocr_options_tesseract = TesseractOcrOptions(
    lang=["por"],              # Idioma português
)

# ─── Pipeline para PDFs escaneados
pipeline_options = PdfPipelineOptions(
    do_ocr=True,                          # Habilita OCR
    do_table_structure=True,               # Detecta estrutura de tabelas
    ocr_options=ocr_options_easyocr,       # Engine OCR
    generate_page_images=True,             # Gera imagens das páginas
    generate_picture_images=True,          # Extrai figuras
)

# ─── Converter com OCR habilitado
from docling.document_converter import DocumentConverter, PdfFormatOption

converter_ocr = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# Converter documento escaneado
resultado_ocr = converter_ocr.convert("laudo_pericial_escaneado.pdf")
texto_extraido = resultado_ocr.document.export_to_markdown()
```

### 3.3 Fluxo de Decisão: Nativo vs. OCR

```
                    ┌─────────────────────┐
                    │  Documento PDF       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Texto selecionável? │
                    │  (pdfplumber test)   │
                    └──────────┬──────────┘
                          ┌────┴─────┐
                         SIM        NÃO
                          │          │
               ┌──────────▼─┐   ┌───▼──────────────┐
               │  Pipeline  │   │  Pipeline OCR     │
               │  Padrão    │   │  (EasyOCR/        │
               │  Docling   │   │   Tesseract)      │
               └──────────┬─┘   └───┬──────────────┘
                          │         │
                    ┌─────▼─────────▼─────┐
                    │  DoclingDocument     │
                    │  Exportar Markdown   │
                    └─────────────────────┘
```

```python
# Código para detectar automaticamente se PDF precisa de OCR
import pdfplumber

def precisa_ocr(caminho_pdf: str, paginas_amostra: int = 3) -> bool:
    """
    Verifica se um PDF precisa de OCR baseado na quantidade de texto extraível.
    Retorna True se o documento for provavelmente escaneado.
    """
    total_chars = 0
    
    with pdfplumber.open(caminho_pdf) as pdf:
        # Analisa as primeiras N páginas como amostra
        for i, pagina in enumerate(pdf.pages[:paginas_amostra]):
            texto = pagina.extract_text() or ""
            total_chars += len(texto.strip())
    
    # Heurística: menos de 100 chars por página → provavelmente escaneado
    media_chars = total_chars / min(paginas_amostra, 1)
    precisa = media_chars < 100
    
    print(f"📄 Média de chars por página: {media_chars:.0f}")
    print(f"🔍 Precisa OCR: {'SIM' if precisa else 'NÃO'}")
    
    return precisa
```

**Aplicação jurídica:** Esta função de detecção automática é essencial em pipelines que processam documentos de diferentes origens — desde acórdãos digitais até cópias escaneadas de processos físicos do arquivo judicial.

---

## 4. Extração de Tabelas Estruturadas

### 4.1 Por que Tabelas são Críticas em Documentos Jurídicos

Tabelas aparecem em contextos jurídicos e de segurança pública de formas variadas:

- **Laudos periciais:** tabelas de resultados de exames (substâncias, medidas, comparações)
- **Relatórios de inteligência:** tabelas de análise de rede, movimentação financeira
- **Decisões administrativas:** tabelas de critérios de avaliação, pontuações
- **Sumários de jurisprudência:** tabelas comparativas de precedentes

A extração incorreta de uma tabela pode inverter o sentido de um resultado pericial ou confundir dados financeiros.

### 4.2 Extração e Exportação de Tabelas com Docling

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.document import DoclingDocument
import pandas as pd

def extrair_tabelas_documento(caminho_pdf: str) -> list[dict]:
    """
    Extrai todas as tabelas de um documento PDF usando Docling.
    Retorna lista de dicts com posição, markdown e DataFrame.
    """
    converter = DocumentConverter()
    resultado = converter.convert(caminho_pdf)
    doc = resultado.document
    
    tabelas = []
    
    # Iterar por todos os elementos do documento
    for item, nivel in doc.iterate_items():
        # Verificar se é uma tabela
        from docling_core.types.doc import TableItem
        if isinstance(item, TableItem):
            tabela_info = {
                "caption": item.caption_text(doc) if hasattr(item, 'caption_text') else "",
                "linhas": item.data.num_rows if item.data else 0,
                "colunas": item.data.num_cols if item.data else 0,
                "markdown": item.export_to_markdown(doc),
                "dataframe": None
            }
            
            # Tentar converter para DataFrame pandas
            try:
                # Extrair dados da tabela como lista de listas
                dados = []
                if item.data and item.data.grid:
                    for linha in item.data.grid:
                        linha_dados = [celula.text for celula in linha]
                        dados.append(linha_dados)
                
                if dados:
                    df = pd.DataFrame(dados[1:], columns=dados[0])
                    tabela_info["dataframe"] = df
            except Exception as e:
                print(f"⚠️ Não foi possível converter tabela para DataFrame: {e}")
            
            tabelas.append(tabela_info)
    
    print(f"✅ {len(tabelas)} tabela(s) extraída(s) do documento")
    return tabelas

# Uso
tabelas = extrair_tabelas_documento("laudo_pericial.pdf")
for i, tab in enumerate(tabelas):
    print(f"\n=== Tabela {i+1}: {tab['caption'] or 'Sem legenda'} ===")
    print(f"Dimensões: {tab['linhas']}x{tab['colunas']}")
    print(tab["markdown"])
```

### 4.3 Tratamento de Tabelas em Markdown para RAG

```
Tabela extraída pelo Docling (formato Markdown):

| Amostra | Massa (g) | Substância           | Pureza |
|---------|-----------|----------------------|--------|
| LAB-001 | 87,3      | Cloridrato de Cocaína| 72,4%  |
| LAB-002 | 143,7     | Cloridrato de Cocaína| 68,1%  |
| LAB-003 | 52,8      | Cloridrato de Cocaína| 75,2%  |

↓ Chunking especial para tabelas:

Chunk gerado com contexto da tabela:
"TABELA - Resultados de exame pericial:
Amostra LAB-001: 87,3g de Cloridrato de Cocaína (72,4% de pureza).
Amostra LAB-002: 143,7g de Cloridrato de Cocaína (68,1% de pureza).
Amostra LAB-003: 52,8g de Cloridrato de Cocaína (75,2% de pureza).
Total identificado: 283,8g de substância entorpecente."
```

**Regra de ouro:** Tabelas nunca devem ser divididas em meio — um chunk deve sempre conter a tabela inteira ou pelo menos suas colunas de cabeçalho mais as linhas correspondentes.

---

## 5. Estratégias de Chunking com Docling

### 5.1 HybridChunker — O Chunker Nativo do Docling

O Docling oferece um chunker nativo chamado `HybridChunker` que aproveita a estrutura semântica do documento para fazer divisões inteligentes:

```python
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

# Converter documento
converter = DocumentConverter()
resultado = converter.convert("acordao_stj.pdf")
doc = resultado.document

# HybridChunker: combina chunking hierárquico com tamanho máximo
chunker = HybridChunker(
    tokenizer="BAAI/bge-m3",  # Usa tokenizer do modelo de embedding
    max_tokens=512,            # Máximo de tokens por chunk
    merge_peers=True,          # Une chunks pequenos vizinhos
)

# Gerar chunks
chunks = list(chunker.chunk(doc))

print(f"📦 Total de chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Texto: {chunk.text[:200]}...")
    print(f"Metadados: {chunk.meta.headings}")  # Títulos da hierarquia
```

### 5.2 Comparação de Estratégias de Chunking para Documentos Jurídicos

| Estratégia | Docling HybridChunker | RecursiveCharacterTextSplitter | MarkdownHeaderSplitter | Artigo/Parágrafo |
|---|---|---|---|---|
| **Preserva tabelas** | ✅ Sim | ❌ Pode dividir | ⚠️ Parcial | ✅ Sim |
| **Respeita hierarquia** | ✅ Sim (headings) | ❌ Não | ✅ Sim | ✅ Sim |
| **OCR integrado** | ✅ Sim | ❌ Não | ❌ Não | ❌ Não |
| **Metadados por chunk** | ✅ Ricos | ⚠️ Básicos | ✅ Headers | ✅ Artigo/§ |
| **Configuração** | Moderada | Simples | Simples | Complexa |
| **Ideal para** | Relatórios, laudos | Texto corrido | Legislação em MD | Legislação em PDF |

### 5.3 Chunker Personalizado para Legislação Brasileira

```python
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChunkerLegislacao:
    """
    Chunker especializado para legislação brasileira.
    Respeita a hierarquia: Título > Capítulo > Seção > Artigo > Parágrafo > Inciso
    """
    
    # Padrões de separadores jurídicos, em ordem de prioridade
    SEPARADORES = [
        r"\n(?=TÍTULO\s+[IVXLC]+)",           # Títulos
        r"\n(?=CAPÍTULO\s+[IVXLC]+)",         # Capítulos
        r"\n(?=Seção\s+[IVXLC]+)",            # Seções
        r"\n(?=Art\.\s+\d+[º°])",             # Artigos
        r"\n(?=§\s*\d+[º°])",                 # Parágrafos
        r"\n(?=Parágrafo único\.)",            # Parágrafo único
        r"\n(?=[IVXLC]+\s*[-–])",             # Incisos romanos
        r"\n(?=[a-z]\))",                      # Alíneas
        r"\n\n",                               # Parágrafos em branco
        r"\n",                                 # Quebras de linha
    ]
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.SEPARADORES,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
        )
    
    def chunk(self, texto: str, metadados: dict = None) -> list[dict]:
        """Divide texto legislativo em chunks com metadados."""
        chunks_texto = self.splitter.split_text(texto)
        
        chunks = []
        for i, texto_chunk in enumerate(chunks_texto):
            # Detectar artigo de referência do chunk
            artigo_ref = self._extrair_artigo(texto_chunk)
            
            chunk = {
                "texto": texto_chunk,
                "metadados": {
                    **(metadados or {}),
                    "chunk_index": i,
                    "artigo_referencia": artigo_ref,
                    "num_chars": len(texto_chunk),
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _extrair_artigo(self, texto: str) -> str:
        """Extrai o primeiro número de artigo encontrado no texto."""
        match = re.search(r"Art\.\s+(\d+[º°]?)", texto)
        return match.group(1) if match else "N/A"
```

---

## 6. Pipelines de Ingestão em Escala

### 6.1 Arquitetura de Pipeline para Múltiplos Documentos

```
┌─────────────────────────────────────────────────────────────────────┐
│                PIPELINE DE INGESTÃO EM ESCALA                        │
│                                                                      │
│  ┌───────────┐   ┌──────────────┐   ┌─────────────────────────────┐ │
│  │  Fila de  │   │  Docling     │   │  Pós-processamento          │ │
│  │ Documentos│──▶│  Converter   │──▶│  - Limpeza de texto         │ │
│  │ (JSON/Dir)│   │  (Paralelo)  │   │  - Normalização             │ │
│  └───────────┘   └──────────────┘   │  - Extração de metadados    │ │
│                                     └─────────────┬───────────────┘ │
│                                                   │                  │
│  ┌──────────────────────────────────────────────▼────────────────┐  │
│  │  Chunking Inteligente                                          │  │
│  │  HybridChunker (relatórios) | LegislacaoChunker (leis)       │  │
│  └──────────────────────────────┬─────────────────────────────── ┘  │
│                                 │                                    │
│  ┌──────────────────────────────▼─────────────────────────────────┐ │
│  │  Embeddings BGE-M3 (batch)                                     │ │
│  │  + Indexação OpenSearch (kNN + BM25)                          │ │
│  └──────────────────────────────────────────────────────────────── │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Processamento em Lote com ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from docling.document_converter import DocumentConverter
import time

def processar_documento(caminho: Path, converter: DocumentConverter) -> dict:
    """Processa um único documento e retorna chunks + metadados."""
    try:
        resultado = converter.convert(str(caminho))
        doc = resultado.document
        markdown = doc.export_to_markdown()
        
        return {
            "arquivo": caminho.name,
            "status": "sucesso",
            "paginas": len(doc.pages),
            "chars": len(markdown),
            "markdown": markdown
        }
    except Exception as e:
        return {
            "arquivo": caminho.name,
            "status": "erro",
            "erro": str(e)
        }

def pipeline_ingestao_paralela(
    diretorio: str, 
    max_workers: int = 4,
    extensoes: list = [".pdf", ".docx"]
) -> list[dict]:
    """
    Processa múltiplos documentos em paralelo.
    
    Args:
        diretorio: Pasta com os documentos
        max_workers: Número de threads paralelas
        extensoes: Extensões de arquivo aceitas
    """
    pasta = Path(diretorio)
    arquivos = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]
    
    print(f"📂 {len(arquivos)} documentos encontrados em {diretorio}")
    
    # Um converter por thread (thread-safe no Docling 2.0+)
    converter = DocumentConverter()
    
    resultados = []
    inicio = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todas as tarefas
        futures = {
            executor.submit(processar_documento, arq, converter): arq 
            for arq in arquivos
        }
        
        # Coletar resultados conforme completam
        for future in as_completed(futures):
            resultado = future.result()
            resultados.append(resultado)
            
            status_emoji = "✅" if resultado["status"] == "sucesso" else "❌"
            print(f"{status_emoji} {resultado['arquivo']} — {resultado.get('chars', 0)} chars")
    
    tempo_total = time.time() - inicio
    sucessos = sum(1 for r in resultados if r["status"] == "sucesso")
    
    print(f"\n📊 Processamento concluído em {tempo_total:.1f}s")
    print(f"✅ Sucesso: {sucessos}/{len(arquivos)} documentos")
    
    return resultados
```

### 6.3 Cache de Documentos Processados

```python
import hashlib
import json
import pickle
from pathlib import Path

class CacheDocling:
    """
    Cache em disco para evitar reprocessamento de documentos já convertidos.
    Invalida automaticamente quando o arquivo fonte é modificado.
    """
    
    def __init__(self, diretorio_cache: str = "/tmp/docling_cache"):
        self.cache_dir = Path(diretorio_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _hash_arquivo(self, caminho: Path) -> str:
        """Gera hash MD5 do arquivo para detecção de mudanças."""
        md5 = hashlib.md5()
        with open(caminho, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()
    
    def obter(self, caminho: Path) -> dict | None:
        """Retorna resultado em cache, ou None se não existir/expirado."""
        hash_arquivo = self._hash_arquivo(caminho)
        cache_path = self.cache_dir / f"{hash_arquivo}.pkl"
        
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None
    
    def salvar(self, caminho: Path, resultado: dict):
        """Salva resultado no cache."""
        hash_arquivo = self._hash_arquivo(caminho)
        cache_path = self.cache_dir / f"{hash_arquivo}.pkl"
        
        with open(cache_path, "wb") as f:
            pickle.dump(resultado, f)
```

**Aplicação jurídica:** Em sistemas de segurança pública, o cache é essencial para evitar o reprocessamento de toda a base legislativa (Código Penal, CPP, legislações específicas) a cada atualização do sistema RAG.

---

## 7. Metadados e Enriquecimento de Chunks

### 7.1 Metadados Jurídicos Essenciais

Cada chunk deve carregar metadados suficientes para que o sistema RAG possa:
1. Apresentar a fonte exata para o usuário
2. Filtrar por tipo de documento ou tribunal
3. Ranquear por data (preferindo jurisprudência mais recente)
4. Respeitar controles de acesso (relatórios sigilosos)

```python
from datetime import datetime
import re

def enriquecer_metadados_juridicos(chunk_texto: str, metadados_base: dict) -> dict:
    """
    Extrai e enriquece metadados jurídicos a partir do texto do chunk.
    
    Args:
        chunk_texto: Texto do chunk
        metadados_base: Metadados já conhecidos (fonte, tipo, etc.)
    
    Returns:
        Dicionário de metadados enriquecido
    """
    meta = metadados_base.copy()
    
    # Detectar número de processo CNJ
    padrao_cnj = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
    numeros_processo = re.findall(padrao_cnj, chunk_texto)
    if numeros_processo:
        meta["numero_processo"] = numeros_processo[0]
    
    # Detectar artigos de lei mencionados
    artigos = re.findall(r'art(?:igo)?\.\s*(\d+[º°]?)', chunk_texto, re.IGNORECASE)
    if artigos:
        meta["artigos_mencionados"] = list(set(artigos))
    
    # Detectar leis citadas
    leis = re.findall(r'Lei\s+(?:n[º°]\s*)?(\d+[\./]\d+)', chunk_texto)
    if leis:
        meta["leis_citadas"] = list(set(leis))
    
    # Detectar tribunal
    tribunais = {
        "STF": "Supremo Tribunal Federal",
        "STJ": "Superior Tribunal de Justiça",
        "TJ": "Tribunal de Justiça",
        "TRF": "Tribunal Regional Federal",
        "TST": "Tribunal Superior do Trabalho"
    }
    for sigla, nome in tribunais.items():
        if sigla in chunk_texto:
            meta["tribunal"] = sigla
            meta["tribunal_nome"] = nome
            break
    
    # Timestamp de ingestão
    meta["data_ingestao"] = datetime.now().isoformat()
    meta["num_chars"] = len(chunk_texto)
    meta["num_tokens_estimado"] = len(chunk_texto.split()) * 1.3  # Estimativa
    
    return meta
```

### 7.2 Esquema Completo de Metadados para Indexação

```python
# Esquema de metadados para documentos jurídicos no OpenSearch
SCHEMA_METADADOS = {
    # Identificação
    "id_chunk": "str",          # UUID único do chunk
    "id_documento": "str",       # ID do documento de origem
    "indice_chunk": "int",       # Posição do chunk no documento
    
    # Fonte e tipo
    "tipo_documento": "str",     # acordao, legislacao, laudo, relatorio, bo
    "tribunal": "str",           # STF, STJ, TJ-SP, etc.
    "numero_processo": "str",    # Formato CNJ
    "data_documento": "date",    # Data do documento original
    
    # Conteúdo
    "texto": "str",              # Texto do chunk (para BM25)
    "embedding": "knn_vector",   # Vetor BGE-M3 1024 dims
    "titulo_secao": "str",       # Heading pai do chunk
    
    # Jurídico
    "artigos_mencionados": "keyword[]",  # Art. 33, Art. 7º...
    "leis_citadas": "keyword[]",          # 11.343/2006, LGPD...
    "assuntos": "keyword[]",              # Tags temáticas
    "classificacao": "keyword",           # publico, restrito, sigiloso
    
    # Técnico
    "data_ingestao": "date",
    "versao_docling": "str",
    "metodo_ocr": "str",         # none, easyocr, tesseract
    "qualidade_ocr": "float",    # Score de confiança do OCR
}
```

---

## 8. Limpeza e Normalização de Texto Jurídico

### 8.1 Ruídos Comuns em Documentos Jurídicos Extraídos

```python
import re
import unicodedata

def limpar_texto_juridico(texto: str) -> str:
    """
    Remove ruídos comuns em textos jurídicos extraídos por OCR ou Docling.
    
    Ruídos tratados:
    - Cabeçalhos/rodapés repetitivos (números de página, nome do tribunal)
    - Hifenização de palavras (quebras de linha em PDFs)
    - Caracteres especiais do OCR (l→1, O→0, etc.)
    - Espaços duplos e quebras de linha excessivas
    - Numeração automática de páginas
    """
    # 1. Normalizar unicode (remove acentos problemáticos do OCR)
    texto = unicodedata.normalize('NFKC', texto)
    
    # 2. Corrigir hifenização (palavras quebradas no final da linha)
    # "fun-\ndamento" → "fundamento"
    texto = re.sub(r'(\w+)-\n(\w+)', r'\1\2', texto)
    
    # 3. Remover cabeçalhos de página (ex: "STJ - SUPERIOR TRIBUNAL DE JUSTIÇA\n1\n")
    texto = re.sub(r'^.{0,80}TRIBUNAL.{0,80}\n\d+\n', '', texto, flags=re.MULTILINE|re.IGNORECASE)
    
    # 4. Remover números de página isolados
    texto = re.sub(r'^\s*\d+\s*$', '', texto, flags=re.MULTILINE)
    
    # 5. Normalizar espaços múltiplos
    texto = re.sub(r' {2,}', ' ', texto)
    
    # 6. Normalizar quebras de linha (máximo 2 consecutivas)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    # 7. Remover linhas que são apenas separadores visuais
    texto = re.sub(r'^[─━═\-_\.]{3,}\s*$', '', texto, flags=re.MULTILINE)
    
    # 8. Corrigir erros comuns de OCR em termos jurídicos
    CORRECOES_OCR = {
        'arl.': 'art.',
        'Arl.': 'Art.',
        'I3rasil': 'Brasil',
        'Iegis': 'legis',
        'judiciál': 'judicial',
    }
    for erro, correto in CORRECOES_OCR.items():
        texto = texto.replace(erro, correto)
    
    # 9. Strip final
    texto = texto.strip()
    
    return texto
```

---

## 9. Limitações e Armadilhas do Docling

### 9.1 Cinco Armadilhas Comuns

**Armadilha 1: PDFs com proteção de cópia**
```python
# ❌ Falha silenciosa — Docling retorna texto vazio
resultado = converter.convert("processo_protegido.pdf")
markdown = resultado.document.export_to_markdown()
if len(markdown.strip()) < 100:
    print("⚠️ ATENÇÃO: PDF pode ter proteção de cópia")
    print("Solução: Usar qpdf para remover proteção ou solicitar versão sem restrição")
```

**Armadilha 2: OCR lento em documentos longos**
```python
# ❌ Timeout em documentos de 500+ páginas com OCR
# ✅ Processar por intervalo de páginas
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Limitar páginas processadas por batch
pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    # Docling 2.x: use page_range para subsets
)
```

**Armadilha 3: Tabelas multi-página perdidas**
```python
# Tabelas que continuam em páginas seguintes podem ser
# detectadas como tabelas separadas. Verificação:
def verificar_tabelas_continuacao(tabelas: list) -> list:
    """Detecta e une tabelas que são continuação uma da outra."""
    if len(tabelas) < 2:
        return tabelas
    
    tabelas_unidas = [tabelas[0]]
    for tab_atual in tabelas[1:]:
        tab_anterior = tabelas_unidas[-1]
        # Heurística: mesma estrutura de colunas e sem caption → continuação
        if (tab_atual["colunas"] == tab_anterior["colunas"] 
                and not tab_atual["caption"]):
            # Une os dados da tabela
            tab_anterior["markdown"] += "\n" + tab_atual["markdown"]
            tab_anterior["linhas"] += tab_atual["linhas"]
        else:
            tabelas_unidas.append(tab_atual)
    
    return tabelas_unidas
```

**Armadilha 4: Figuras e gráficos incluídos como texto vazio**
```python
# Figuras geram chunks com pouco ou nenhum texto — poluem o índice
def filtrar_chunks_validos(chunks: list, min_chars: int = 50) -> list:
    """Remove chunks muito curtos (provavelmente de figuras/imagens)."""
    validos = [c for c in chunks if len(c.get("texto", "")) >= min_chars]
    filtrados = len(chunks) - len(validos)
    print(f"🗑️ {filtrados} chunks filtrados (texto insuficiente)")
    return validos
```

**Armadilha 5: Encoding incorreto em PDFs antigos**
```python
# PDFs gerados antes de 2010 podem ter encoding Windows-1252
# Docling lida bem, mas verifique:
def verificar_encoding_pdf(texto: str) -> bool:
    """Verifica se há caracteres corrompidos no texto extraído."""
    caracteres_suspeitos = ['â€™', 'Ã©', 'Ã£', '\\x00']
    return not any(c in texto for c in caracteres_suspeitos)
```

---

## 10. Síntese e Próximos Passos

### 10.1 Mapa de Decisão para Ingestão de Documentos Jurídicos

| Tipo de Documento | Formato | OCR Necessário | Chunker Recomendado | Metadados Prioritários |
|---|---|---|---|---|
| Acórdão STJ/STF | PDF nativo | Não | HybridChunker | tribunal, número_processo, data |
| Laudo pericial | PDF escaneado | Sim (EasyOCR) | HybridChunker | tipo_laudo, número_ic, data_exame |
| Legislação | PDF nativo | Não | ChunkerLegislacao | lei, artigo, data_vigência |
| Formulário BO | PDF escaneado | Sim | HybridChunker | número_bo, delegacia, data |
| Relatório DENARC | PDF nativo c/ tabelas | Não | HybridChunker | classificação, operação, data |
| Sumula | PDF nativo | Não | Por súmula individual | número_súmula, tribunal |

### 10.2 Pipeline Completo Integrado

```
PDF/DOCX → Docling → HybridChunker → Metadados → BGE-M3 → OpenSearch
   │                      │               │              │
   ▼                      ▼               ▼              ▼
Nativo vs.         Tabelas +          Tipo doc,      kNN +
OCR detect.        hierarquia         tribunal,      BM25
                   respeitada         data           híbrido
```

A próxima aula (Aula 6) expandirá este pipeline incorporando o **Model Context Protocol (MCP)** para conectar sistemas RAG a ferramentas externas de forma padronizada — como bancos de dados de processos judiciais, APIs de consulta de CPF/CNPJ e repositórios de legislação online.

---

## Referências (ABNT)

AUER, Peter et al. **Docling Technical Report**. arXiv:2408.09869, 2024. IBM Research Europe. Disponível em: <https://arxiv.org/abs/2408.09869>. Acesso em: abr. 2026.

CHEN, J. et al. **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**. arXiv:2309.07597, 2024.

IBM RESEARCH. **Docling Documentation**. Disponível em: <https://ds4sd.github.io/docling/>. Acesso em: abr. 2026.

LANGCHAIN. **Document Loaders — Docling**. Disponível em: <https://python.langchain.com/docs/integrations/document_loaders/docling/>. Acesso em: abr. 2026.

LEWIS, Patrick et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems*, v. 33, p. 9459-9474, 2020.

NAKAYAMA, Henrique. **Processamento de Linguagem Natural com Python**. São Paulo: Novatec, 2023. (Adaptado para contexto jurídico brasileiro)

SMITH, Ray; TESSERACT OCR Team. **An Overview of the Tesseract OCR Engine**. In: *Ninth International Conference on Document Analysis and Recognition*. IEEE, 2007.

BRASIL. Lei nº 13.709, de 14 de agosto de 2018. **Lei Geral de Proteção de Dados Pessoais (LGPD)**. Brasília, DF: Presidência da República, 2018. Disponível em: <http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709.htm>. Acesso em: abr. 2026.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_graph_dataset_advanced_graphrag.py

STRICT + CLEAN VERSION

Features:
- original dataset generator
- mandatory graph-source RAG setup
- graph_source_rag returns ONLY str
- no MindMap / formulate / dict RAG wrapper
- if keywords are extracted but no graph nodes match, graph_source_rag returns ""
- verbose mode prints what happened
- RAG section is included in prompt only when RAG text is non-empty
- rejected answers are medium-quality direct answers
- resume is tolerant: local -> hub -> fresh start
- critical pipeline failures raise immediately
"""

import argparse
import glob
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfFolder
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


# ============================================================
# Sentinels
# ============================================================

THINK_START = "<think>"
THINK_END = "</think>"
BRAINSTORM_START = "<brainstorm>"
BRAINSTORM_END = "</brainstorm>"
GRAPH_START = "<graph>"
GRAPH_END = "</graph>"
GRAPH_JSON_START = "<graph_json>"
GRAPH_JSON_END = "</graph_json>"
PATTERNS_START = "<patterns>"
PATTERNS_END = "</patterns>"
SYNTHESIS_START = "<synthesis>"
SYNTHESIS_END = "</synthesis>"

TRUNC_LEN = 32000
MAX_KEYWORDS = 5
LOWER_WORDS = {
    "in", "to", "on", "for", "and", "or", "of", "with", "at", "by",
    "from", "as", "the", "a", "an"
}
DEBUG = False


# ============================================================
# Dataset graph schema
# ============================================================

class Node(BaseModel):
    id: str = Field(...)
    type: Literal[
        "entity",
        "attribute",
        "process",
        "event",
        "outcome",
        "law",
        "claim",
    ]
    level: Optional[Literal["micro", "meso", "macro"]] = None


class Edge(BaseModel):
    source: str
    relation: Literal[
        "causes",
        "enables",
        "inhibits",
        "modulates",
        "part_of",
        "instance_of",
        "supports",
        "challenges",
        "represents",
        "promotes",
        "violates",
        "constrains",
    ]
    target: str


class GraphJSON(BaseModel):
    nodes: List[Node]
    edges: List[Edge] = Field(default_factory=list)


class QuestionResponse(BaseModel):
    question: str = Field(..., description="A challenging, standalone question based on the context")


class RejectedAnswerResponse(BaseModel):
    answer: str = Field(
        ...,
        description="A medium-quality direct answer without chain-of-thought or graph markup"
    )


class GraphRepairResponse(BaseModel):
    nodes: List[Node]
    edges: List[Edge] = Field(default_factory=list)
    repair_notes: Optional[str] = Field(None, description="Notes about any repairs made")


def validate_graph_semantics(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        graph = GraphJSON(**obj)
    except Exception:
        return None

    ids = [n.id for n in graph.nodes]
    if len(ids) != len(set(ids)):
        return None

    for _id in ids:
        if not isinstance(_id, str) or not _id or " " in _id:
            return None

    id_set = set(ids)
    for e in graph.edges:
        if e.source not in id_set or e.target not in id_set:
            return None

    return obj


# ============================================================
# Graph-maker schema
# ============================================================

class GraphMakerNode(BaseModel):
    id: str
    type: str


class GraphMakerNodes(BaseModel):
    nodes: List[GraphMakerNode]


SYSTEM_PROMPT_GRAPHMAKER = (
    "You are a network ontology graph maker. Given a context (including chunked text, extracted paths, citations, and references), "
    "you must extract entities and relations to build a scientific knowledge graph using clear, widely used technical names "
    "(materials, systems, devices, methods, processes). Use category-theoretic thinking to keep relations well-typed and meaningful. "

    "INPUT SCOPE "
    "You will receive context that may include: (i) plain text, (ii) existing graph paths, (iii) RAG snippets with titles/DOIs/chunk_id/figure labels, "
    "and (iv) chat artifacts such as images or screenshots with a location/identifier. You MUST process ALL provided content. "

    "REFERENCE-AWARE GRAPHING (CRITICAL) "
    "You must treat references and provenance as FIRST-CLASS graph components. "
    "Whenever you see ANY of the following, you MUST represent it in the graph and connect it to the scientific claims it supports: "
    "reference titles, paper/book/report names, dataset/tool names, DOIs, figure/table identifiers, chunk_id, section headers, and image locations. "

    "NODES "
    "Every node MUST have fields <id> and <type>. "
    "Use stable, human-readable IDs when available (e.g., the exact paper title string as shown, the exact chunk_id string, the exact figure label, "
    "or the exact image location string). Do NOT invent bibliographic details. Do NOT normalize or rename reference text; keep it exactly as given. "
    "Required reference node types: "
    'type="reference_title" for an explicitly shown title/name of an external source, '
    'type="doi" for DOI strings, '
    'type="chunk" for chunk_id nodes, '
    'type="figure" for figure/table identifiers (e.g., "Figure 2", "Fig. S1"), '
    'type="image" for any image/screenshot location/identifier, '
    'type="tool" or "dataset" if a named tool/dataset is explicitly present, '
    'type="term" for scientific/technical entities (materials, devices, methods, phenomena). '

    "EDGES "
    "Every edge MUST have <source>, <target>, and <relation>. "
    "<relation> must be a concise, information-carrying predicate that reflects a real statement in the context, not a generic link. "
    "Prefer relations that capture scientific meaning (e.g., 'enables', 'causes', 'measured_by', 'limits', 'depends_on') and provenance meaning "
    "(e.g., 'supported_by', 'reported_in', 'appears_in', 'extracted_from', 'has_doi', 'has_chunk', 'described_in_figure'). "

    "PROVENANCE LINKS (MANDATORY) "
    "For every non-trivial technical relation you add between term nodes, you MUST also add provenance edges to at least one reference artifact "
    "(reference_title / doi / chunk / figure / image) that is present in the provided context. "
    "If multiple provenance artifacts are present, prefer the most specific: figure/image > chunk_id > DOI > title. "

    "EDGE METADATA "
    "Each edge MAY include <metadata> only when the context explicitly provides metadata. "
    "Use <metadata> to store any explicitly seen fields such as title, DOI, chunk_id, figure label, page number, or source name. "
    "If no metadata is explicitly present for that edge, set <metadata> to an empty object {}. "
    "NEVER fabricate metadata. "

    "OUTPUT "
    "Return a JSON object with exactly two top-level fields: <nodes> and <edges>. "
    "<nodes> is a list of node objects, each with <id> and <type>. "
    "<edges> is a list of edge objects, each with <source>, <target>, <relation>, and optional <metadata>."
)


# ============================================================
# OpenAI helpers
# ============================================================

def get_llm_client(api_key: str, base_url: Optional[str] = None, timeout: float = 120.0) -> OpenAI:
    params: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        params["base_url"] = base_url
    if timeout:
        params["timeout"] = timeout
    return OpenAI(**params)


def llm_call(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Optional[str]:
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )
        return (resp.output_text or "").strip()
    except Exception:
        pass

    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (chat.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[llm_call] both APIs failed: {e}")
        return None


def llm_parse(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format: type,
) -> Optional[Any]:
    try:
        resp = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=response_format,
        )
        return resp.output_parsed
    except Exception:
        pass

    try:
        chat = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        return chat.choices[0].message.parsed
    except Exception:
        pass

    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = chat.choices[0].message.content or ""
        obj = json.loads(content)
        return response_format(**obj)
    except Exception as e:
        print(f"[llm_parse] all methods failed: {e}")
        return None


# ============================================================
# Dataset parsing/sampling
# ============================================================

def parse_dataset_spec(spec: str) -> Tuple[str, Optional[int]]:
    spec = spec.strip()
    if "[::" in spec or spec.count("[") > 1:
        raise ValueError(f"Invalid dataset spec: {spec}")

    if "[:" in spec and spec.endswith("]"):
        bracket_idx = spec.rfind("[:")
        dataset_name = spec[:bracket_idx]
        num_str = spec[bracket_idx + 2:-1]
        try:
            num_samples = int(num_str)
        except ValueError:
            raise ValueError(f"Invalid sample count in spec: {spec}")
        return dataset_name, num_samples

    return spec, None


def parse_datasets_string(datasets_str: str) -> List[Tuple[str, Optional[int]]]:
    if not datasets_str or not datasets_str.strip():
        raise ValueError("--datasets string cannot be empty")

    specs = datasets_str.split("|")
    parsed = []
    for spec in specs:
        spec = spec.strip()
        if spec:
            parsed.append(parse_dataset_spec(spec))

    if not parsed:
        raise ValueError("No valid datasets found in --datasets string")

    return parsed


def sample_streamed_dataset(
    dataset_name: str,
    split: str = "train",
    sample_size: Optional[int] = None,
    streaming: bool = True,
    field: str = "text",
) -> Dataset:
    limit_str = f"n={sample_size}" if sample_size else "all"
    print(f"Streaming dataset: {dataset_name} (split={split}, {limit_str})")
    ds_stream = load_dataset(dataset_name, split=split, streaming=streaming)
    it = iter(ds_stream)
    rows = []
    count = 0
    for row in it:
        if sample_size is not None and count >= sample_size:
            break
        if field in row and isinstance(row[field], str):
            rows.append(row)
            count += 1
    ds = Dataset.from_list(rows)
    print(f"  -> Collected {len(ds)} rows")
    return ds


def build_corpus(
    dataset_specs: List[Tuple[str, Optional[int]]],
    text_field: str = "text",
) -> Dataset:
    datasets_list = []
    for dataset_name, num_samples in dataset_specs:
        ds = sample_streamed_dataset(dataset_name, sample_size=num_samples, field=text_field)
        if len(ds) > 0:
            datasets_list.append(ds)

    if not datasets_list:
        raise ValueError("No data collected from any dataset")

    combined = datasets_list[0] if len(datasets_list) == 1 else concatenate_datasets(datasets_list)
    combined = combined.shuffle(seed=42)
    print(f"Final combined corpus size: {len(combined)}")
    return combined


# ============================================================
# Graph-source RAG resources and pipeline
# ============================================================

@dataclass
class GraphRAGResources:
    client: OpenAI
    model: str
    graph: nx.DiGraph
    node_embeddings: Dict[str, np.ndarray]
    embedding_tokenizer: Any
    embedding_model: Any
    embedding_device: str
    chunk_docs_by_id: Dict[str, Dict[str, Any]]
    verbose: bool = False


GRAPH_RAG_RESOURCES: Optional[GraphRAGResources] = None


@dataclass
class Context:
    knowledgebase: Optional["KnowledgeBase"] = None
    chunk_docs_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    verbose: bool = False


shared_context = Context()


def _is_verbose() -> bool:
    return bool(shared_context.verbose or (GRAPH_RAG_RESOURCES and GRAPH_RAG_RESOURCES.verbose))


def clean_edge_metadata_title(G: nx.DiGraph, meta_key: str = "metadata", title_key: str = "title") -> int:
    changed = 0
    for _, _, d in G.edges(data=True):
        meta = d.get(meta_key)
        if isinstance(meta, dict):
            t = meta.get(title_key)
            if isinstance(t, str) and "- PDF Free Download" in t:
                meta[title_key] = t.replace("- PDF Free Download", "").replace("-PDF Free Download", "").strip()
                changed += 1
    return changed


def dedup_by_chunk_ids(
    chunk_ids: List[str],
    chunks: List[str],
    titles: List[str],
    keep: str = "first",
) -> Tuple[List[str], List[str], List[str]]:
    if not (len(chunk_ids) == len(chunks) == len(titles)):
        raise ValueError("Lists must be the same length.")

    seen = set()
    keep_idx = []
    rng = range(len(chunk_ids)) if keep == "first" else range(len(chunk_ids) - 1, -1, -1)

    for i in rng:
        cid = chunk_ids[i]
        if cid in seen:
            continue
        seen.add(cid)
        keep_idx.append(i)

    keep_idx.sort()
    return (
        [chunk_ids[i] for i in keep_idx],
        [chunks[i] for i in keep_idx],
        [titles[i] for i in keep_idx],
    )


def load_chunk_documents(chunk_dir: str, chunk_glob: str = "*chunks_clean.csv") -> Dict[str, Dict[str, Any]]:
    csv_files = sorted(glob.glob(os.path.join(chunk_dir, chunk_glob)))
    if not csv_files:
        raise FileNotFoundError(f"No chunk CSV files found in {chunk_dir!r} with pattern {chunk_glob!r}")

    chunk_ids: List[str] = []
    chunks: List[str] = []
    titles: List[str] = []

    for path in csv_files:
        df = pd.read_csv(path)
        base = os.path.basename(path).replace("_chunks_clean.csv", "")
        parts = base.replace("_", " ").split(" ")
        parts = parts[1:]

        parts_cap = []
        for idx, w in enumerate(parts):
            if not w:
                parts_cap.append("")
                continue
            if idx == 0:
                parts_cap.append(w[:1].upper() + w[1:])
            elif w.lower() in LOWER_WORDS:
                parts_cap.append(w.lower())
            else:
                parts_cap.append(w[:1].upper() + w[1:])

        title = " ".join(parts_cap)
        df["title"] = title

        chunk_ids += list(df["chunk_id"])
        chunks += list(df["text"])
        titles += list(df["title"])

    chunk_ids, chunks, titles = dedup_by_chunk_ids(chunk_ids, chunks, titles)

    chunk_docs_by_id: Dict[str, Dict[str, Any]] = {}
    for cid, content, title in zip(chunk_ids, chunks, titles):
        chunk_docs_by_id[str(cid)] = {
            "id": str(cid),
            "content": content,
            "metadata": {"title": title},
        }

    return chunk_docs_by_id


def embedding_function(
    input: Union[str, List[str]],
    batch_size: int = 8,
    is_query: bool = False,
    **kwargs,
) -> List[List[float]]:
    if GRAPH_RAG_RESOURCES is None:
        raise RuntimeError("GRAPH_RAG_RESOURCES is not initialized.")

    from GraphReasoning import generate_node_embeddings

    if isinstance(input, (list, tuple)):
        texts = ["" if x is None else str(x).strip() for x in input]
    else:
        texts = ["" if input is None else str(input).strip()]

    texts = [t for t in texts if t]
    if not texts:
        raise RuntimeError("embedding_function received empty text input")

    embs = generate_node_embeddings(
        items=texts,
        batch_size=batch_size,
        tokenizer=GRAPH_RAG_RESOURCES.embedding_tokenizer,
        model=GRAPH_RAG_RESOURCES.embedding_model,
        device=GRAPH_RAG_RESOURCES.embedding_device,
        embedding_function=None,
        pool="mean",
        normalize=True,
    )

    if isinstance(embs, dict):
        return [np.asarray(embs[t], dtype=np.float32).tolist() for t in texts]

    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.tolist()


def _as_nd(v: Union[List[float], np.ndarray]) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def _l2_normalize(v: Union[List[float], np.ndarray], eps: float = 1e-12) -> np.ndarray:
    x = _as_nd(v)
    if x.ndim == 1:
        return x / (float(np.linalg.norm(x)) + eps)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def graph_rag_generate(
    system_prompt: str = "",
    prompt: str = "",
    temperature: float = 0,
    response_model=None,
):
    if GRAPH_RAG_RESOURCES is None:
        raise RuntimeError("GRAPH_RAG_RESOURCES is not initialized.")

    full_system = system_prompt.strip()
    if response_model is not None:
        full_system = (full_system + "\n\n" + SYSTEM_PROMPT_GRAPHMAKER).strip()

    if response_model is not None:
        return llm_parse(
            client=GRAPH_RAG_RESOURCES.client,
            model=GRAPH_RAG_RESOURCES.model,
            system_prompt=full_system,
            user_prompt=prompt,
            response_format=response_model,
        )

    return llm_call(
        client=GRAPH_RAG_RESOURCES.client,
        model=GRAPH_RAG_RESOURCES.model,
        system_prompt=full_system,
        user_prompt=prompt,
    )


def find_shortest_path_subgraph_between_nodes(
    graph: nx.DiGraph,
    nodes: List[Any],
) -> nx.DiGraph:
    terminals: List[Any] = []
    seen = set()
    for x in nodes:
        if x not in seen:
            terminals.append(x)
            seen.add(x)

    if not terminals:
        raise RuntimeError("find_shortest_path_subgraph_between_nodes received no terminal nodes")

    if len(terminals) == 1:
        return graph.subgraph([terminals[0]]).copy()

    pair_path: Dict[Tuple[Any, Any], List[Any]] = {}
    pair_len: Dict[Tuple[Any, Any], int] = {}

    for i, a in enumerate(terminals):
        for j, b in enumerate(terminals):
            if i == j:
                continue
            try:
                p = nx.shortest_path(graph, a, b)
                pair_path[(a, b)] = p
                pair_len[(a, b)] = len(p) - 1
            except nx.NetworkXNoPath:
                pass

    succs: Dict[Any, List[Any]] = {u: [] for u in terminals}
    indeg: Dict[Any, int] = {u: 0 for u in terminals}
    for (u, v), _ in pair_len.items():
        succs[u].append(v)
    for u in terminals:
        for v in succs[u]:
            indeg[v] += 1

    uncovered = set(terminals)
    terminal_chains: List[List[Any]] = []

    while uncovered:
        starts = sorted(uncovered, key=lambda x: (indeg[x], terminals.index(x)))
        cur = starts[0]
        chain = [cur]
        uncovered.remove(cur)

        while True:
            candidates = [v for v in succs.get(cur, []) if v in uncovered]
            if not candidates:
                break
            nxt = min(candidates, key=lambda v: pair_len[(cur, v)])
            chain.append(nxt)
            uncovered.remove(nxt)
            cur = nxt

        terminal_chains.append(chain)

    keep_nodes = set()
    for chain in terminal_chains:
        if len(chain) == 1:
            keep_nodes.add(chain[0])
            continue
        for k in range(len(chain) - 1):
            u, v = chain[k], chain[k + 1]
            path = pair_path[(u, v)]
            if k == 0:
                keep_nodes.update(path)
            else:
                keep_nodes.update(path[1:])

    if not keep_nodes:
        raise RuntimeError(f"find_shortest_path_subgraph_between_nodes produced empty subgraph for nodes: {nodes}")

    return graph.subgraph(list(keep_nodes)).copy()


def collect_entities(graph: nx.DiGraph, chunk_ids: Optional[List[str]] = None) -> List[str]:
    want = set(chunk_ids) if chunk_ids is not None else None
    lines: List[str] = []
    for u, v, data in graph.out_edges(data=True):
        if want is not None:
            cid = data.get("chunk_id", "")
            if cid not in want:
                continue

        relation = data.get("relation")
        chunk_id = data.get("chunk_id")
        doi = data.get("DOI")
        rel_txt = f"-[{relation}]->" if relation else "-->"

        line = f"{u} {rel_txt} {v}."
        if doi:
            line += f" | title (DOI): {doi}"
        if chunk_id:
            line += f" | chunk_id: {chunk_id}"
        lines.append(line)

    return lines


class KnowledgeBase:
    def __init__(
        self,
        graph: Optional[nx.DiGraph],
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        generate: Optional[Callable[..., Any]] = None,
        node_embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 3,
        node_text: Optional[Callable[[str, Mapping], str]] = None,
    ):
        self.G: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self.embed_fn = embed_fn
        self.generate = generate
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.top_k = int(top_k)
        self.node_text = node_text or (lambda nid, data: str(nid))

        if len(self.G.nodes) == 0:
            raise RuntimeError("KnowledgeBase initialized with empty graph")

        if node_embeddings:
            for k, v in node_embeddings.items():
                self.node_embeddings[k] = _l2_normalize(v)

        missing = [n for n in self.G.nodes if n not in self.node_embeddings]
        if missing:
            self._embed_nodes(missing)

        if not self.node_embeddings:
            raise RuntimeError("KnowledgeBase has no node embeddings after initialization")

    def _embed_nodes(self, node_ids: List[str]) -> None:
        texts = [self.node_text(nid, self.G.nodes[nid]) for nid in node_ids]
        vecs = self.embed_fn(texts)
        for nid, v in zip(node_ids, vecs):
            self.node_embeddings[nid] = _l2_normalize(v)

    def similar_nodes(
        self,
        keyword: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        if not self.node_embeddings:
            raise RuntimeError("similar_nodes called with empty node embeddings")

        k = int(top_k or self.top_k)
        q = _l2_normalize(self.embed_fn([keyword])[0])

        scored: List[Tuple[str, float]] = []
        for node, emb in self.node_embeddings.items():
            sim = float(np.dot(q.ravel(), _l2_normalize(emb).ravel()))
            scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:k]

        if threshold is not None:
            scored = [(n, s) for (n, s) in scored if s >= threshold]

        return scored

    def extract_keywords(self, query: str) -> List[str]:
        if not self.generate:
            raise RuntimeError("generate is not set on KnowledgeBase")

        resp = self.generate(
            system_prompt=(
                f"Identify the keywords, each of which is in one to three words, and can partially overlap "
                f"with each other, in the context. Never give zero, one, or more than {MAX_KEYWORDS} keywords. "
                f"Expand the context if the question is too generic."
            ),
            prompt=f"Context: ```{query}```",
            response_model=GraphMakerNodes,
        )
        if resp is None:
            raise RuntimeError(f"extract_keywords returned None for query: {query}")

        keywords = [n.id for n in resp.nodes if getattr(n, "id", None)]
        if not keywords:
            raise RuntimeError(f"extract_keywords returned no keywords for query: {query}")

        if _is_verbose():
            print(f"[graph_rag] keywords={keywords}")

        return keywords

    def keywords_to_subgraph(
        self,
        keywords: List[str],
        max_n_samples: int = 3,
        similarity_threshold: float = 0.9,
    ) -> nx.DiGraph:
        chosen: Set[str] = set()

        for kw in keywords:
            cands = self.similar_nodes(
                kw,
                top_k=max_n_samples,
                threshold=similarity_threshold,
            )
            for cand, _score in cands:
                chosen.add(cand)

        # 唯一放寬：有 keywords 但 pair 不到 graph node -> 回空 subgraph
        if not chosen:
            if _is_verbose():
                print(f"[graph_rag] keywords matched no graph nodes: {keywords}; returning empty subgraph")
            return self.G.subgraph([]).copy()

        if len(chosen) == 1:
            node = list(chosen)[0]
            graph_nodes = {node}
            graph_nodes.update(self.G.successors(node))
            graph_nodes.update(self.G.predecessors(node))
            subgraph = self.G.subgraph(graph_nodes).copy()
        else:
            subgraph = find_shortest_path_subgraph_between_nodes(self.G, list(chosen))

        if len(subgraph.nodes) == 0:
            raise RuntimeError(f"keywords_to_subgraph produced empty subgraph for keywords: {keywords}")

        if _is_verbose():
            print(f"[graph_rag] chosen_nodes={len(chosen)} subgraph_nodes={len(subgraph.nodes)} subgraph_edges={len(subgraph.edges)}")

        return subgraph

    def extract_keywords_to_subgraph(
        self,
        query: str,
        max_n_samples: int,
        similarity_threshold: float,
    ) -> nx.DiGraph:
        keywords = self.extract_keywords(query)
        return self.keywords_to_subgraph(keywords, max_n_samples, similarity_threshold)


def init_context(
    *,
    graph: nx.DiGraph,
    embed_fn: Callable[[List[str]], List[List[float]]],
    generate: Callable[..., Any],
    node_embeddings: Dict[str, np.ndarray],
    chunk_docs_by_id: Dict[str, Dict[str, Any]],
    verbose: bool = False,
) -> None:
    shared_context.knowledgebase = KnowledgeBase(
        graph,
        embed_fn=embed_fn,
        generate=generate,
        node_embeddings=node_embeddings,
    )
    shared_context.chunk_docs_by_id = chunk_docs_by_id
    shared_context.verbose = bool(verbose)


def graph_source_rag(
    query: str = "",
    similarity_threshold: Union[float, str] = 0.95,
) -> str:
    if not shared_context.knowledgebase:
        raise RuntimeError("graph_source_rag: knowledgebase is not initialized")

    if _is_verbose():
        print(f"[graph_rag] query={query}")

    t0 = time.time()

    max_n = 1
    sim_thr = float(similarity_threshold)

    subgraph = shared_context.knowledgebase.extract_keywords_to_subgraph(query, max_n, sim_thr)

    # 唯一允許不 raise 的情況
    if len(subgraph.nodes) == 0:
        if _is_verbose():
            print(
                f"[graph_rag] no graph nodes matched extracted keywords; "
                f"returning empty RAG result | elapsed={time.time() - t0:.2f}s"
            )
        return ""

    paths_list = collect_entities(subgraph)

    if not paths_list:
        raise RuntimeError(f"graph_source_rag: no relations found for query: {query}")

    seen_chunk_ids: Set[str] = set()
    chunk_lines: List[str] = []

    for _, _, data in subgraph.out_edges(data=True):
        cid = data.get("chunk_id")
        if not cid:
            continue

        cid = str(cid)
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)

        if not shared_context.chunk_docs_by_id:
            raise RuntimeError("graph_source_rag: chunk_docs_by_id is not initialized")

        doc = shared_context.chunk_docs_by_id.get(cid)
        if not doc:
            continue

        content = (doc.get("content", "") or "").strip()
        if not content:
            continue

        doi = (data.get("DOI", "") or "").strip()

        if doi:
            chunk_lines.append(
                f"Source chunk: {content} | title (DOI): {doi} | chunk_id: {cid}"
            )
        else:
            chunk_lines.append(
                f"Source chunk: {content} | chunk_id: {cid}"
            )

    if not chunk_lines:
        raise RuntimeError(f"graph_source_rag: no usable source chunk contents found for query: {query}")

    paths_text = "\n".join(paths_list)
    chunks_text = "\n".join(chunk_lines)

    out = (
        f"Retrieved PATH:\n{paths_text}\n\n"
        f"Retrieved Source Text:\n{chunks_text}"
    ).strip()

    if not out:
        raise RuntimeError(f"graph_source_rag: empty output for query: {query}")

    if _is_verbose():
        print(
            f"[graph_rag] ok | paths={len(paths_list)} | chunks={len(chunk_lines)} | "
            f"elapsed={time.time() - t0:.2f}s"
        )

    return out


def initialize_graph_rag(
    *,
    graphml_path: str,
    chunk_dir: str,
    embedding_model_path: str,
    embedding_cache_path: str,
    graph_rag_client: OpenAI,
    graph_rag_model: str,
    chunk_glob: str = "*chunks_clean.csv",
    verbose: bool = False,
) -> None:
    global GRAPH_RAG_RESOURCES

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from GraphReasoning import load_embeddings, save_embeddings, generate_node_embeddings

    device_n = torch.cuda.device_count()
    embedding_device = f"cuda:{max(device_n - 1, 0)}" if torch.cuda.is_available() else "cpu"

    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, use_fast=False)
    embedding_model = AutoModelForCausalLM.from_pretrained(
        embedding_model_path,
        device_map=embedding_device if embedding_device.startswith("cuda") else None,
        torch_dtype="auto",
    )

    G = nx.read_graphml(graphml_path)
    relation = nx.get_edge_attributes(G, "title")
    nx.set_edge_attributes(G, relation, "relation")
    nx.set_node_attributes(G, nx.pagerank(G), "pr")
    clean_edge_metadata_title(G)

    if len(G.nodes) == 0 or len(G.edges) == 0:
        raise RuntimeError(f"initialize_graph_rag: loaded graph is empty or invalid: {graphml_path}")

    os.makedirs(os.path.dirname(embedding_cache_path) or ".", exist_ok=True)

    regenerate_embeddings = not os.path.exists(embedding_cache_path)
    with torch.no_grad():
        if regenerate_embeddings:
            print("Regenerating graph node embeddings...")
            node_embeddings = generate_node_embeddings(
                items=list(G.nodes),
                batch_size=32,
                tokenizer=embedding_tokenizer,
                model=embedding_model,
                device=embedding_device,
                pool="mean",
                normalize=False,
            )
            save_embeddings(node_embeddings, embedding_cache_path)
        else:
            print(f"Loading graph node embeddings from: {embedding_cache_path}")
            node_embeddings = load_embeddings(embedding_cache_path)

    if not node_embeddings:
        raise RuntimeError("initialize_graph_rag: node embeddings are empty")

    chunk_docs_by_id = load_chunk_documents(chunk_dir, chunk_glob=chunk_glob)
    if not chunk_docs_by_id:
        raise RuntimeError(f"initialize_graph_rag: no chunk documents found in {chunk_dir}")

    GRAPH_RAG_RESOURCES = GraphRAGResources(
        client=graph_rag_client,
        model=graph_rag_model,
        graph=G,
        node_embeddings=node_embeddings,
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        chunk_docs_by_id=chunk_docs_by_id,
        verbose=verbose,
    )

    init_context(
        graph=G,
        embed_fn=embedding_function,
        generate=graph_rag_generate,
        node_embeddings=node_embeddings,
        chunk_docs_by_id=chunk_docs_by_id,
        verbose=verbose,
    )

    print(f"Graph RAG initialized: {len(G.nodes)} nodes, {len(G.edges)} edges, {len(chunk_docs_by_id)} chunks")


# ============================================================
# Teacher calls
# ============================================================

def teacher_generate_question(
    client: OpenAI,
    model: str,
    context: str,
    system_prompt: str = "You are a scientist who designs deep, self-contained questions.",
) -> Optional[str]:
    user_prompt = (
        "Using ONLY the information and style implicit in the following context, write a single challenging, "
        "standalone question. The question should:\n"
        "- not mention the word 'context', 'paper', 'authors', or references,\n"
        "- be answerable by a highly trained expert (in that domain),\n"
        "- be specific, not vague.\n\n"
        f"Context:\n{context}\n\n"
        "Return a JSON object with a 'question' field containing your question."
    )

    result = llm_parse(client, model, system_prompt, user_prompt, QuestionResponse)
    if result is not None:
        q = result.question.strip()
        if not q.endswith("?"):
            q += "?"
        return q

    txt = llm_call(client, model, system_prompt, user_prompt)
    if not txt:
        return None

    q = txt.split("\n")[0].strip()
    if not q.endswith("?"):
        q += "?"
    return q


def teacher_generate_structured_answer(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
) -> Optional[str]:
    sys = (
        "You are an expert in mechanistic, graph-based reasoning across ALL domains: "
        "science, engineering, software, art, humanities, social sciences, etc. "
        "You always reason using a typed, multi-level graph as a latent world model."
    )

    rag_text = graph_source_rag(question)

    if not isinstance(rag_text, str):
        raise TypeError(f"graph_source_rag must return str, got {type(rag_text)}")

    has_rag = bool(rag_text.strip())

    if _is_verbose():
        print(f"[graph_rag] has_rag={has_rag}")

    if has_rag:
        user = f"""
Given the following question, context, and graph-source RAG result, answer in TWO phases.

You MUST first inspect the graph-source RAG result carefully.
Use the retrieved graph paths / graph evidence as primary grounding whenever they are relevant.
If the graph-source RAG result is incomplete, use the context only as supplemental support.
Do not ignore the graph-source RAG result.

QUESTION:
{question}

GRAPH-SOURCE RAG RESULT:
{rag_text}

CONTEXT:
{context}

FIRST, produce a structured internal reasoning trace using EXACTLY this template:

{THINK_START}
{BRAINSTORM_START}
Freely explore the problem space. Start from the graph-source RAG result first.
Identify the most relevant graph paths, entities, relations, mechanisms, actors, events, themes, and outcomes.
If the graph-source RAG result contains paths, evidence chains, source snippets, or node/edge structure, prioritize those.
Then organize your thinking across three scales:
- micro: concrete components, individuals, local actions or details (e.g., cells, functions, characters, transactions),
- meso: interactions, subsystems, scenes, workflows, social groups (e.g., pathways, services, rituals, committees),
- macro: large-scale systems, institutions, themes, goals, long-term outcomes (e.g., ecosystems, markets, historical trends, narrative themes).
Think broadly and creatively before structuring anything. Note key variables and how they might interact, in any domain.
{BRAINSTORM_END}

{GRAPH_START}
Now, create a domain-agnostic conceptual graph *in words* (NOT JSON yet).

Important:
- Base this graph primarily on the graph-source RAG result.
- If graph paths were retrieved, preserve their key entities and relations as much as possible.
- You may add missing bridge nodes only when necessary for coherence.

1. List 8–20 core nodes. For each node, specify:
   - a short CamelCase id with no spaces (e.g., SilkFiber, LaserPulse, ThemeAlienation, UserSignupFlow),
   - a type chosen from: entity, attribute, process, event, outcome, law, claim,
   - an optional level chosen from: micro, meso, macro.

2. Then, describe the directed relationships among these nodes using ONLY the following relation verbs:
   causes, enables, inhibits, modulates, part_of, instance_of, supports, challenges, represents, promotes, violates, constrains.

These relationships should be fundamentally causal, structural, argumentative, or symbolic – not just loose associations.
{GRAPH_END}

{GRAPH_JSON_START}
Now provide a STRICT JSON object with keys "nodes" and "edges" ONLY, following this schema:

- Each node must have:
  - "id": a short CamelCase string without spaces.
  - "type": one of ["entity", "attribute", "process", "event", "outcome", "law", "claim"].
  - "level": optional, one of ["micro", "meso", "macro"].

- Each edge must have:
  - "source": a node id (string),
  - "relation": one of ["causes", "enables", "inhibits", "modulates", "part_of", "instance_of",
                       "supports", "challenges", "represents", "promotes", "violates", "constrains"],
  - "target": a node id (string).

- Node ids must be unique. All edges must refer only to existing node ids.
- Do NOT include any extra keys or comments.
- Prefer to preserve the core graph structure implied by the graph-source RAG result.
{GRAPH_JSON_END}

{PATTERNS_START}
Based on the graph, describe 2–4 abstract patterns or laws that summarize its structure.

Guidelines:
- Use node ids and the allowed relation verbs only.
- Prefer multi-step chains like:
  - A causes B causes C
  - A enables B, which inhibits C
  - MicroProcessX modulates MacroOutcomeY
- When relevant, express simple quantitative or comparative relations, but keep them tied to the graph nodes.
- When graph-source RAG paths are available, summarize the dominant path patterns explicitly.

Avoid storytelling here; treat this as compressed graph structure and causal/argument patterns.
{PATTERNS_END}

{SYNTHESIS_START}
Integrate everything above into a coherent picture that answers the question.

Explain:
- What the graph-source RAG result contributes,
- How the retrieved graph paths or evidence chains support the reasoning,
- How micro-level elements interact at the meso level to produce macro-level outcomes,
- How the key relationships in the graph jointly explain the phenomenon or support an answer to the question.
{SYNTHESIS_END}
{THINK_END}

SECOND, on a NEW line after {THINK_END}, write a comprehensive, detailed final answer to the question.

Your final answer should:
- Be thorough and well-structured,
- Answer the question using the graph-source RAG result as primary grounding when relevant,
- Explain the underlying mechanisms, structures, or arguments,
- Reference concepts from your graph but ALWAYS use natural language,
- Provide specific details, examples, or quantitative insights where relevant,
- Be suitable for an expert audience seeking deep understanding.
- If helpful, you may include a brief subsection title related to the retrieved graph evidence.
- Only include such a subsection if the graph-source RAG result materially contributes to the answer.

IMPORTANT: The final answer must be written in proper English prose. Do NOT use CamelCase identifiers, graph notation, or any technical markup from the thinking section.
"""
    else:
        user = f"""
Given the following question and context, answer in TWO phases.

Use the context only.
Do NOT mention graph-source RAG, retrieval, retrieved evidence, or the absence of retrieval in the final answer.

QUESTION:
{question}

CONTEXT:
{context}

FIRST, produce a structured internal reasoning trace using EXACTLY this template:

{THINK_START}
{BRAINSTORM_START}
Freely explore the problem space from the provided context.
Identify the most relevant entities, relations, mechanisms, actors, events, themes, and outcomes.
Then organize your thinking across three scales:
- micro: concrete components, individuals, local actions or details (e.g., cells, functions, characters, transactions),
- meso: interactions, subsystems, scenes, workflows, social groups (e.g., pathways, services, rituals, committees),
- macro: large-scale systems, institutions, themes, goals, long-term outcomes (e.g., ecosystems, markets, historical trends, narrative themes).
Think broadly and creatively before structuring anything. Note key variables and how they might interact, in any domain.
{BRAINSTORM_END}

{GRAPH_START}
Now, create a domain-agnostic conceptual graph *in words* (NOT JSON yet) based on the context.

1. List 8–20 core nodes. For each node, specify:
   - a short CamelCase id with no spaces,
   - a type chosen from: entity, attribute, process, event, outcome, law, claim,
   - an optional level chosen from: micro, meso, macro.

2. Then, describe the directed relationships among these nodes using ONLY the following relation verbs:
   causes, enables, inhibits, modulates, part_of, instance_of, supports, challenges, represents, promotes, violates, constrains.

These relationships should be fundamentally causal, structural, argumentative, or symbolic – not just loose associations.
{GRAPH_END}

{GRAPH_JSON_START}
Now provide a STRICT JSON object with keys "nodes" and "edges" ONLY, following this schema:

- Each node must have:
  - "id": a short CamelCase string without spaces.
  - "type": one of ["entity", "attribute", "process", "event", "outcome", "law", "claim"].
  - "level": optional, one of ["micro", "meso", "macro"].

- Each edge must have:
  - "source": a node id (string),
  - "relation": one of ["causes", "enables", "inhibits", "modulates", "part_of", "instance_of",
                       "supports", "challenges", "represents", "promotes", "violates", "constrains"],
  - "target": a node id (string).

- Node ids must be unique. All edges must refer only to existing node ids.
- Do NOT include any extra keys or comments.
{GRAPH_JSON_END}

{PATTERNS_START}
Based on the graph, describe 2–4 abstract patterns or laws that summarize its structure.

Guidelines:
- Use node ids and the allowed relation verbs only.
- Prefer multi-step chains like:
  - A causes B causes C
  - A enables B, which inhibits C
  - MicroProcessX modulates MacroOutcomeY
- When relevant, express simple quantitative or comparative relations, but keep them tied to the graph nodes.

Avoid storytelling here; treat this as compressed graph structure and causal/argument patterns.
{PATTERNS_END}

{SYNTHESIS_START}
Integrate everything above into a coherent picture that answers the question.

Explain:
- How the key relationships in the graph support the answer,
- How micro-level elements interact at the meso level to produce macro-level outcomes.
{SYNTHESIS_END}
{THINK_END}

SECOND, on a NEW line after {THINK_END}, write a comprehensive, detailed final answer to the question.

Your final answer should:
- Be thorough and well-structured,
- Answer the question from the provided context,
- Explain the underlying mechanisms, structures, or arguments,
- Reference concepts from your graph but ALWAYS use natural language,
- Provide specific details, examples, or quantitative insights where relevant,
- Be suitable for an expert audience seeking deep understanding.
- Do NOT add any subsection or sentence about graph-source RAG, retrieval, or retrieved evidence.

IMPORTANT: The final answer must be written in proper English prose. Do NOT use CamelCase identifiers, graph notation, or any technical markup from the thinking section.
"""

    return llm_call(client, model, sys, user)


def teacher_generate_rejected(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
) -> Optional[str]:
    """
    Medium-quality direct answer:
    - no <think>
    - no graph_json
    - no hidden reasoning
    - weaker than chosen because it is shorter, less systematic,
      and does not explicitly build graph-structured reasoning
    """
    sys = (
        "You are a reasonably competent expert giving a direct answer without showing chain-of-thought. "
        "Your answer should be useful and readable, but clearly less complete and less rigorous than a full research-grade answer."
    )

    user = f"""
Answer the following question directly, WITHOUT showing reasoning, and WITHOUT any special tokens.

Requirements:
- Write a decent answer, not a throwaway short response.
- Aim for about 1 to 3 short paragraphs.
- Be informative and technically plausible.
- Do NOT use <think>, <graph>, <graph_json>, or any hidden reasoning format.
- Do NOT explicitly enumerate a graph or reasoning trace.
- Do NOT be as comprehensive as a full expert report.
- Do NOT be extremely brief.
- Prefer a clear direct explanation over exhaustive detail.

Question:
{question}

Context:
{context}

Return a JSON object with an 'answer' field containing your answer.
"""

    result = llm_parse(client, model, sys, user, RejectedAnswerResponse)
    if result is not None:
        ans = result.answer.strip()
        if ans:
            return ans

    return llm_call(client, model, sys, user)


# ============================================================
# Graph JSON parsing helpers
# ============================================================

def validate_and_repair_graph(
    client: OpenAI,
    model: str,
    raw_json_str: str,
) -> Optional[GraphJSON]:
    try:
        obj = json.loads(raw_json_str)
        graph = GraphJSON(**obj)

        ids = [n.id for n in graph.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs")

        for _id in ids:
            if " " in _id:
                raise ValueError(f"Space in node ID: {_id}")

        id_set = set(ids)
        for e in graph.edges:
            if e.source not in id_set or e.target not in id_set:
                raise ValueError(f"Edge references non-existent node: {e.source} -> {e.target}")

        return graph
    except Exception:
        pass

    sys = (
        "You are a graph schema validator. Fix the provided graph JSON to comply with the schema. "
        "Node types must be one of: entity, attribute, process, event, outcome, law, claim. "
        "Edge relations must be one of: causes, enables, inhibits, modulates, part_of, instance_of, "
        "supports, challenges, represents, promotes, violates, constrains. "
        "Node IDs must be CamelCase with no spaces. All edge endpoints must reference existing nodes."
    )
    user = f"""
Fix this graph JSON to comply with the schema. Preserve the semantic content but correct any schema violations.

Invalid JSON:
{raw_json_str}

Return a valid graph with 'nodes' and 'edges' arrays following the schema.
"""

    result = llm_parse(client, model, sys, user, GraphRepairResponse)
    if result is not None:
        try:
            graph = GraphJSON(nodes=result.nodes, edges=result.edges)
            ids = [n.id for n in graph.nodes]
            if len(ids) != len(set(ids)):
                return None
            id_set = set(ids)
            for e in graph.edges:
                if e.source not in id_set or e.target not in id_set:
                    return None
            return graph
        except Exception:
            return None

    return None


def find_tag_block(s: str, start_tag: str, end_tag: str) -> Optional[str]:
    i = s.find(start_tag)
    if i == -1:
        return None
    j = s.find(end_tag, i + len(start_tag))
    if j == -1:
        return None
    return s[i + len(start_tag):j].strip()


def extract_graph_json_block(
    chosen: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    inner = find_tag_block(chosen, GRAPH_JSON_START, GRAPH_JSON_END)
    if not inner:
        return None

    if client is not None and model is not None:
        graph = validate_and_repair_graph(client, model, inner)
        if graph is not None:
            return graph.model_dump_json()
        return None

    try:
        obj = json.loads(inner)
    except Exception:
        return None

    validated = validate_graph_semantics(obj)
    if not validated:
        return None

    try:
        return json.dumps(validated, ensure_ascii=False)
    except Exception:
        return None


def extract_final_answer(chosen: str) -> str:
    idx = chosen.find(THINK_END)
    if idx == -1:
        return chosen.strip()
    return chosen[idx + len(THINK_END):].strip()


# ============================================================
# Dataset builder
# ============================================================

def build_graph_reasoning_dataset(
    corpus: Dataset,
    teacher_client: OpenAI,
    reject_client: OpenAI,
    teacher_model: str,
    reject_model: str,
    num_examples: int,
    output_path: str,
    save_steps: int = 100,
    resume: bool = False,
    push_to_hub: bool = False,
    output_repo: Optional[str] = None,
    hub_public: bool = False,
) -> Dataset:
    rows: List[Dict[str, Any]] = []

    if resume:
        loaded_resume = False
        resume_errors: List[str] = []

        if os.path.exists(output_path):
            try:
                existing_ds = load_dataset("json", data_files=output_path, split="train")
                rows = [dict(row) for row in existing_ds]
                print(f"Resuming from local file {output_path} with {len(rows)} existing examples")
                loaded_resume = True
            except Exception as e:
                resume_errors.append(f"local resume failed: {e}")
                print(f"[resume] Could not load local file {output_path}: {e}")
                print("[resume] Will try Hub next if output_repo is set.")

        if (not loaded_resume) and output_repo:
            try:
                print(f"[resume] Checking Hub for {output_repo} ...")
                existing_ds = load_dataset(output_repo, split="train")
                rows = [dict(row) for row in existing_ds]
                print(f"Downloaded {len(rows)} existing examples from Hub: {output_repo}")
                if len(rows) > 0:
                    Dataset.from_list(rows).to_json(output_path, lines=True)
                    print(f"Saved Hub data to local file: {output_path}")
                loaded_resume = True
            except Exception as e:
                resume_errors.append(f"hub resume failed: {e}")
                print(f"[resume] Could not load resume data from Hub {output_repo}: {e}")

        if not loaded_resume:
            print("[resume] No valid resume source found. Starting fresh...")
            for err in resume_errors:
                print(f"[resume] {err}")

    if len(rows) >= num_examples:
        print(f"Already have {len(rows)} examples (target: {num_examples}). Nothing to generate.")
        return Dataset.from_list(rows)

    remaining = num_examples - len(rows)
    print(f"Building up to {num_examples} examples (have {len(rows)}, need {remaining} more)...")
    print(f"  Teacher model (questions + chosen): {teacher_model}")
    print(f"  Reject model (rejected answers): {reject_model}")
    print(f"  Saving every {save_steps} examples to {output_path}")
    if push_to_hub and output_repo:
        print(f"  Pushing to Hub: {output_repo}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    indices = list(range(len(corpus)))
    random.shuffle(indices)

    pbar = tqdm(total=num_examples, initial=len(rows), desc="Building examples", unit="ex")
    last_save_count = len(rows)

    for idx in indices:
        if len(rows) >= num_examples:
            break

        row = corpus[idx]
        context = row.get("text", "")
        if not isinstance(context, str) or len(context.strip()) < 200:
            continue

        if len(context) > TRUNC_LEN:
            context = context[:TRUNC_LEN]

        q = teacher_generate_question(teacher_client, teacher_model, context)
        if not q:
            raise RuntimeError(f"teacher_generate_question failed at corpus idx={idx}")

        structured = teacher_generate_structured_answer(teacher_client, teacher_model, q, context)
        if not structured:
            raise RuntimeError(f"teacher_generate_structured_answer failed at corpus idx={idx}, question={q}")

        graph_json_str = extract_graph_json_block(
            structured,
            client=teacher_client,
            model=teacher_model,
        )
        if not graph_json_str:
            raise RuntimeError(f"extract_graph_json_block failed at corpus idx={idx}, question={q}")

        final_answer = extract_final_answer(structured)
        if not final_answer:
            raise RuntimeError(f"extract_final_answer failed at corpus idx={idx}, question={q}")

        rejected = teacher_generate_rejected(reject_client, reject_model, q, context)
        if not rejected:
            raise RuntimeError(f"teacher_generate_rejected failed at corpus idx={idx}, question={q}")

        rows.append(
            {
                "prompt": q.strip(),
                "answer": final_answer.strip(),
                "chosen": structured.strip(),
                "rejected": rejected.strip(),
                "teacher_graph_json": graph_json_str,
            }
        )
        pbar.update(1)

        if len(rows) - last_save_count >= save_steps:
            last_save_count = len(rows)
            ds_checkpoint = Dataset.from_list(rows)
            ds_checkpoint.to_json(output_path, lines=True)
            tqdm.write(f"  [checkpoint] Saved {len(rows)} examples to {output_path}")
            if push_to_hub and output_repo:
                try:
                    ds_checkpoint.push_to_hub(output_repo, private=(not hub_public))
                    tqdm.write(f"  [checkpoint] Pushed {len(rows)} examples to {output_repo}")
                except Exception as e:
                    raise RuntimeError(f"Checkpoint push to Hub failed: {e}") from e

    pbar.close()
    print(f"Done. Created {len(rows)} examples.")
    return Dataset.from_list(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build advanced graph-native reasoning dataset with mandatory graph-source RAG.")

    # dataset generation
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Dataset specs: "dataset_a[:1000]|dataset_b[:500]|dataset_c"',
    )
    parser.add_argument("--num_examples", type=int, default=500)

    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--reject_model", type=str, default=None)
    parser.add_argument("--teacher_api_key", type=str, required=True)
    parser.add_argument("--teacher_base_url", type=str, default=None)
    parser.add_argument("--reject_api_key", type=str, default=None)
    parser.add_argument("--reject_base_url", type=str, default=None)

    parser.add_argument("--output_path", type=str, default="./graph_reasoning_advanced.jsonl")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--output_repo", type=str, default=None)
    parser.add_argument("--hub_public", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    # graph-source rag (mandatory setup)
    parser.add_argument("--graph_rag_graphml_path", type=str, required=True)
    parser.add_argument("--graph_rag_chunk_dir", type=str, required=True)
    parser.add_argument("--graph_rag_embedding_model_path", type=str, required=True)
    parser.add_argument("--graph_rag_embedding_cache_path", type=str, required=True)
    parser.add_argument("--graph_rag_model", type=str, default=None)
    parser.add_argument("--graph_rag_api_key", type=str, default=None)
    parser.add_argument("--graph_rag_base_url", type=str, default=None)
    parser.add_argument("--graph_rag_chunk_glob", type=str, default="*chunks_clean.csv")
    parser.add_argument("--graph_rag_verbose", action="store_true")

    args = parser.parse_args()

    if args.hf_token:
        HfFolder.save_token(args.hf_token)

    dataset_specs = parse_datasets_string(args.datasets)
    print(f"Parsed {len(dataset_specs)} dataset(s):")
    for name, n in dataset_specs:
        print(f"  - {name} [{n if n else 'all'}]")

    corpus = build_corpus(dataset_specs=dataset_specs)

    teacher_client = get_llm_client(api_key=args.teacher_api_key, base_url=args.teacher_base_url)

    reject_model = args.reject_model if args.reject_model else args.teacher_model
    reject_api_key = args.reject_api_key if args.reject_api_key else args.teacher_api_key
    reject_base_url = args.reject_base_url if args.reject_base_url else args.teacher_base_url
    reject_client = (
        get_llm_client(api_key=reject_api_key, base_url=reject_base_url)
        if (reject_api_key != args.teacher_api_key or reject_base_url != args.teacher_base_url)
        else teacher_client
    )

    if args.push_to_hub and not args.output_repo:
        raise ValueError("--output_repo must be set if --push_to_hub is used")

    graph_rag_model = args.graph_rag_model if args.graph_rag_model else args.teacher_model
    graph_rag_api_key = args.graph_rag_api_key if args.graph_rag_api_key else args.teacher_api_key
    graph_rag_base_url = args.graph_rag_base_url if args.graph_rag_base_url else args.teacher_base_url
    graph_rag_client = get_llm_client(api_key=graph_rag_api_key, base_url=graph_rag_base_url)

    initialize_graph_rag(
        graphml_path=args.graph_rag_graphml_path,
        chunk_dir=args.graph_rag_chunk_dir,
        embedding_model_path=args.graph_rag_embedding_model_path,
        embedding_cache_path=args.graph_rag_embedding_cache_path,
        graph_rag_client=graph_rag_client,
        graph_rag_model=graph_rag_model,
        chunk_glob=args.graph_rag_chunk_glob,
        verbose=args.graph_rag_verbose,
    )

    ds = build_graph_reasoning_dataset(
        corpus=corpus,
        teacher_client=teacher_client,
        reject_client=reject_client,
        teacher_model=args.teacher_model,
        reject_model=reject_model,
        num_examples=args.num_examples,
        output_path=args.output_path,
        save_steps=args.save_steps,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
        output_repo=args.output_repo,
        hub_public=args.hub_public,
    )

    ds.to_json(args.output_path, lines=True)
    print(f"Final dataset saved to {args.output_path}")

    if args.push_to_hub:
        print(f"Final push to Hub: {args.output_repo}")
        ds.push_to_hub(args.output_repo, private=(not args.hub_public))


if __name__ == "__main__":
    main()
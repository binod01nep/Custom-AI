import json
import math
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request, send_file

DIMS = 16

app = Flask(__name__)


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


DistFn = Callable[[List[float], List[float]], float]


def euclidean(a: List[float], b: List[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return math.sqrt(total)


def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - (dot / (math.sqrt(na) * math.sqrt(nb)))


def manhattan(a: List[float], b: List[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += abs(x - y)
    return total


def get_dist_fn(metric: str) -> DistFn:
    m = (metric or "").lower()
    if m == "cosine":
        return cosine
    if m == "manhattan":
        return manhattan
    return euclidean


class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        result = [(dist(q, item.emb), item.id) for item in self.items]
        result.sort(key=lambda x: (x[0], x[1]))
        return result[:k]

    def remove(self, item_id: int) -> None:
        self.items = [item for item in self.items if item.id != item_id]


@dataclass
class KDNode:
    item: VectorItem
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.root: Optional[KDNode] = None
        self.dims = dims

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, 0)

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def _knn(self, node: Optional[KDNode], q: List[float], k: int, depth: int, dist: DistFn, out: List[Tuple[float, int]]) -> None:
        if node is None:
            return
        out.append((dist(q, node.item.emb), node.item.id))
        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer, q, k, depth + 1, dist, out)
        if len(out) < k or abs(diff) < sorted(out, key=lambda x: x[0])[-1][0]:
            self._knn(farther, q, k, depth + 1, dist, out)
        out.sort(key=lambda x: (x[0], x[1]))
        if len(out) > k:
            del out[k:]

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        out: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, out)
        out.sort(key=lambda x: (x[0], x[1]))
        return out[:k]

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for item in items:
            self.insert(item)


@dataclass
class HNSWNode:
    item: VectorItem
    max_layer: int
    neighbors: List[List[int]] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.m_l = 1.0 / math.log(m)
        self.rng = random.Random(42)
        self.G: Dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt: Optional[int] = None

    def _rand_level(self) -> int:
        u = self.rng.random()
        return int(math.floor(-math.log(u) * self.m_l))

    def _search_layer(self, q: List[float], ep: int, ef: int, layer: int, dist: DistFn) -> List[Tuple[float, int]]:
        if ep not in self.G:
            return []

        vis = {ep}
        cands = [(dist(q, self.G[ep].item.emb), ep)]
        found = [(dist(q, self.G[ep].item.emb), ep)]

        while cands:
            cands.sort(key=lambda x: (x[0], x[1]))
            cd, cid = cands.pop(0)
            if len(found) >= ef and cd > found[-1][0]:
                break
            if layer >= len(self.G[cid].neighbors):
                continue
            for nid in self.G[cid].neighbors[layer]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < found[-1][0]:
                    cands.append((nd, nid))
                    found.append((nd, nid))
                    if len(found) > ef:
                        found.sort(key=lambda x: (x[0], x[1]))
                        found.pop()
        found.sort(key=lambda x: (x[0], x[1]))
        return found

    def _select_neighbors(self, candidates: List[Tuple[float, int]], max_m: int) -> List[int]:
        candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
        return [item_id for _, item_id in candidates[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item=item, max_layer=lvl, neighbors=[[] for _ in range(lvl + 1)])

        if self.entry_pt is None:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for layer in range(self.top_layer, lvl, -1):
            if layer < len(self.G[ep].neighbors):
                candidates = self._search_layer(item.emb, ep, 1, layer, dist)
                if candidates:
                    ep = candidates[0][1]

        for layer in range(min(self.top_layer, lvl), -1, -1):
            candidates = self._search_layer(item.emb, ep, self.ef_build, layer, dist)
            max_m = self.M0 if layer == 0 else self.M
            selected = self._select_neighbors(candidates, max_m)
            self.G[item_id].neighbors[layer] = selected

            for nid in selected:
                if nid not in self.G:
                    continue
                if layer >= len(self.G[nid].neighbors):
                    self.G[nid].neighbors.extend([[] for _ in range(layer - len(self.G[nid].neighbors) + 1)])
                conn = self.G[nid].neighbors[layer]
                conn.append(item_id)
                if len(conn) > max_m:
                    distances = [(dist(self.G[nid].item.emb, self.G[cid].item.emb), cid) for cid in conn if cid in self.G]
                    distances.sort(key=lambda x: (x[0], x[1]))
                    self.G[nid].neighbors[layer] = [cid for _, cid in distances[:max_m]]

            if candidates:
                ep = candidates[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt is None:
            return []
        ep = self.entry_pt
        for layer in range(self.top_layer, 0, -1):
            if layer < len(self.G[ep].neighbors):
                candidates = self._search_layer(q, ep, 1, layer, dist)
                if candidates:
                    ep = candidates[0][1]
        candidates = self._search_layer(q, ep, max(ef, k), 0, dist)
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return
        for node in self.G.values():
            for layer in node.neighbors:
                if item_id in layer:
                    layer[:] = [x for x in layer if x != item_id]
        if self.entry_pt == item_id:
            self.entry_pt = None
            for nid, node in self.G.items():
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.G[item_id]

    def get_info(self) -> Dict[str, Any]:
        nodes_per_layer = [0] * max(self.top_layer + 2, 1)
        edges_per_layer = [0] * max(self.top_layer + 2, 1)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, int]] = []

        for item_id, node in self.G.items():
            nodes.append({
                "id": item_id,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_layer,
            })
            for layer in range(min(node.max_layer, len(node.neighbors) - 1), -1, -1):
                nodes_per_layer[layer] += 1
                for nid in node.neighbors[layer]:
                    if item_id < nid:
                        edges_per_layer[layer] += 1
                        edges.append({"src": item_id, "dst": nid, "lyr": layer})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.G)


class VectorDB:
    def __init__(self, dims: int):
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = dims

    def insert(self, metadata: str, category: str, emb: List[float], dist: DistFn) -> int:
        with self.mu:
            item = VectorItem(self.next_id, metadata, category, emb)
            self.next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            remaining = list(self.store.values())
            self.kdt.rebuild(remaining)
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> Dict[str, Any]:
        with self.mu:
            dist_fn = get_dist_fn(metric)
            t0 = __import__("time").perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, 50, dist_fn)
            elapsed_us = int((__import__("time").perf_counter() - t0) * 1_000_000)
            hits = []
            for distance, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": distance,
                        "embedding": item.emb,
                    })
            return {"results": hits, "latencyUs": elapsed_us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> Dict[str, Any]:
        with self.mu:
            dfn = get_dist_fn(metric)
            def time_it(fn):
                start = __import__("time").perf_counter()
                fn()
                return int((__import__("time").perf_counter() - start) * 1_000_000)
            return {
                "bruteforceUs": time_it(lambda: self.bf.knn(q, k, dfn)),
                "kdtreeUs": time_it(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": time_it(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "itemCount": len(self.store),
            }

    def all(self) -> List[Dict[str, Any]]:
        with self.mu:
            return [{
                "id": item.id,
                "metadata": item.metadata,
                "category": item.category,
                "embedding": item.emb,
            } for item in self.store.values()]

    def hnsw_info(self) -> Dict[str, Any]:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_available(self) -> bool:
        try:
            res = requests.get(f"{self._base_url()}/api/tags", timeout=2)
            return res.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            res = requests.post(
                f"{self._base_url()}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=(3, 30),
            )
            if res.status_code != 200:
                return []
            payload = res.json()
            return payload.get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            res = requests.post(
                f"{self._base_url()}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=(3, 180),
            )
            if res.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            payload = res.json()
            return payload.get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            self.hnsw.insert(VectorItem(item.id, title, "doc", emb), cosine)
            self.bf.insert(VectorItem(item.id, title, "doc", emb))
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            hits = []
            for distance, item_id in raw:
                if item_id in self.store and distance <= max_dist:
                    hits.append((distance, self.store[item_id]))
            return hits

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap_words
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks


def parse_vec(value: str) -> List[float]:
    if not value:
        return []
    parts = []
    for piece in value.split(','):
        piece = piece.strip()
        if not piece:
            continue
        try:
            parts.append(float(piece))
        except ValueError:
            continue
    return parts


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    items = [
        ("Linked List: nodes connected by pointers", "cs", [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs", [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs", [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs", [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs", [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math", [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math", [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math", [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math", [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math", [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food", [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food", [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food", [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports", [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports", [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports", [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for metadata, category, emb in items:
        db.insert(metadata, category, emb, dist)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/search", methods=["GET", "OPTIONS"])
def route_search():
    if request.method == "OPTIONS":
        return "", 204
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5) or 5)
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")
    payload = db.search(q, k, metric, algo)
    return jsonify(payload)


@app.route("/insert", methods=["POST", "OPTIONS"])
def route_insert():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    metadata = data.get("metadata", "")
    category = data.get("category", "")
    emb = data.get("embedding", [])
    if not metadata or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    item_id = db.insert(metadata, category, emb, get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE", "OPTIONS"])
def route_delete(item_id: int):
    if request.method == "OPTIONS":
        return "", 204
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/items", methods=["GET", "OPTIONS"])
def route_items():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify(db.all())


@app.route("/benchmark", methods=["GET", "OPTIONS"])
def route_benchmark():
    if request.method == "OPTIONS":
        return "", 204
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5) or 5)
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(q, k, metric))


@app.route("/hnsw-info", methods=["GET", "OPTIONS"])
def route_hnsw_info():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify(db.hnsw_info())


@app.route("/doc/insert", methods=["POST", "OPTIONS"])
def route_doc_insert():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    text = data.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids: List[int] = []
    for index, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2"
            }), 400
        if len(chunks) > 1:
            chunk_title = f"{title} [{index + 1}/{len(chunks)}]"
        else:
            chunk_title = title
        ids.append(doc_db.insert(chunk_title, chunk, emb))
    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE", "OPTIONS"])
def route_doc_delete(item_id: int):
    if request.method == "OPTIONS":
        return "", 204
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/doc/list", methods=["GET", "OPTIONS"])
def route_doc_list():
    if request.method == "OPTIONS":
        return "", 204
    docs = []
    for item in doc_db.all():
        preview = item.text[:120]
        if len(item.text) > 120:
            preview += "…"
        docs.append({
            "id": item.id,
            "title": item.title,
            "preview": preview,
            "words": max(1, len(item.text.split())),
        })
    return jsonify(docs)


@app.route("/doc/search", methods=["POST", "OPTIONS"])
def route_doc_search():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    k = int(data.get("k", 3) or 3)
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 400
    hits = doc_db.search(q_emb, k)
    return jsonify({"contexts": [{
        "id": item.id,
        "title": item.title,
        "distance": distance,
    } for distance, item in hits]})


@app.route("/doc/ask", methods=["POST", "OPTIONS"])
def route_doc_ask():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    k = int(data.get("k", 3) or 3)
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 400
    hits = doc_db.search(q_emb, k)

    context = ""
    for index, (_, item) in enumerate(hits, start=1):
        context += f"[{index}] {item.title}:\n{item.text}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{context}"
        f"Question: {question}\n\nAnswer:"
    )
    answer = ollama.generate(prompt)
    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [{
            "id": item.id,
            "title": item.title,
            "text": item.text,
            "distance": distance,
        } for distance, item in hits],
        "docCount": doc_db.size(),
    })


@app.route("/status", methods=["GET", "OPTIONS"])
def route_status():
    if request.method == "OPTIONS":
        return "", 204
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.route("/stats", methods=["GET", "OPTIONS"])
def route_stats():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


@app.route("/", methods=["GET", "OPTIONS"])
def serve_index():
    if request.method == "OPTIONS":
        return "", 204
    index_path = Path(__file__).resolve().parent / "index.html"
    if not index_path.exists():
        return "Not Found", 404
    return send_file(index_path, mimetype="text/html")


if __name__ == "__main__":
    db = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()
    load_demo(db)
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama.is_available() else 'OFFLINE (install from ollama.com)'}")
    if ollama.is_available():
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    app.run(host="0.0.0.0", port=8080, debug=False)

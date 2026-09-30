# -*- coding: utf-8 -*-
"""
Literature Novelty Check Service (Item 8).

Searches Crossref and Semantic Scholar APIs for publications matching a
candidate crystal formula, returning whether the material has prior
synthesis/characterization literature and citing the top matching papers.

API endpoints used:
  Crossref:         https://api.crossref.org/works?query=<formula>&filter=type:journal-article
  Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/search?query=<formula>
                    &fields=title,year,externalIds,publicationTypes

No API keys required — both APIs are free-tier accessible.
Rate limits: Crossref = 50 req/s with polite pool, SemanticScholar = 100 req/min.
"""

import re
import time
import logging
import hashlib
import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

# Cache results for 24 hours to avoid hammering APIs on repeated queries
_CACHE: Dict[str, Dict] = {}
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "lit_check_cache.json")

# Polite pool contact email (required by Crossref for polite pool access — faster rate limits)
CROSSREF_MAILTO = "matscreen-ai@research.edu"

# Number of top references to surface per candidate
TOP_N_REFS = 5


def _load_cache():
    global _CACHE
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r") as f:
                raw = json.load(f)
            # Only use cached entries less than 24h old
            now = time.time()
            _CACHE = {k: v for k, v in raw.items() if now - v.get("_cached_at", 0) < 86400}
        except Exception:
            _CACHE = {}


def _save_cache():
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(_CACHE, f, indent=2)
    except Exception as e:
        logger.debug(f"[LiteratureCheck] Cache save failed: {e}")


_load_cache()


def _cache_key(formula: str) -> str:
    return hashlib.md5(formula.lower().strip().encode()).hexdigest()[:16]


def _normalize_formula(formula: str) -> str:
    """
    Produce multiple query variants for a formula.
    E.g. "KZrCl3" → ["KZrCl3", "K Zr Cl3", "KZrCl₃"]
    """
    # Insert spaces between element-number groups for broader matching
    spaced = re.sub(r"([A-Z][a-z]?)(\d*)", lambda m: m.group(1) + " " + m.group(2), formula).strip()
    return spaced


class LiteratureCheckService:
    """
    Queries Crossref and Semantic Scholar for papers mentioning a crystal formula.
    Returns structured LiteratureHit with known_in_literature flag and top references.
    """

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                "User-Agent": f"MatScreenAI/1.0 (mailto:{CROSSREF_MAILTO})"
            })

    def check_formula(self, formula: str) -> Dict[str, Any]:
        """
        Search for literature references for the given crystal formula.
        
        Returns:
            known_in_literature: bool — True if ≥1 relevant paper found
            confidence: str — high/medium/low (based on # hits and title match quality)
            references: List[{doi, title, year, source, relevance_score}]
            query_formula: str — what was searched
            sources_queried: List[str]
        """
        if not HAS_REQUESTS:
            return self._no_requests_fallback(formula)

        cache_key = _cache_key(formula)
        if cache_key in _CACHE:
            logger.debug(f"[LiteratureCheck] Cache hit for {formula}")
            result = _CACHE[cache_key].copy()
            result["from_cache"] = True
            return result

        refs = []
        sources_queried = []

        # 1. Query Crossref
        try:
            crossref_refs = self._query_crossref(formula)
            refs.extend(crossref_refs)
            sources_queried.append("crossref")
        except Exception as e:
            logger.warning(f"[LiteratureCheck] Crossref failed for {formula}: {e}")

        # 2. Query Semantic Scholar
        try:
            ss_refs = self._query_semantic_scholar(formula)
            refs.extend(ss_refs)
            sources_queried.append("semantic_scholar")
        except Exception as e:
            logger.warning(f"[LiteratureCheck] Semantic Scholar failed for {formula}: {e}")

        # Deduplicate by DOI
        seen_dois = set()
        deduped = []
        for ref in refs:
            doi = ref.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            deduped.append(ref)

        # Score and sort by relevance
        scored = self._score_and_sort(formula, deduped)
        top_refs = scored[:TOP_N_REFS]

        # Determine known_in_literature and confidence
        n_high_relevance = sum(1 for r in top_refs if r.get("relevance_score", 0) >= 0.7)
        n_total = len(top_refs)

        if n_high_relevance >= 2:
            known = True
            confidence = "high"
        elif n_high_relevance == 1 or n_total >= 3:
            known = True
            confidence = "medium"
        elif n_total >= 1:
            known = True
            confidence = "low"
        else:
            known = False
            confidence = "none"

        result = {
            "known_in_literature": known,
            "confidence": confidence,
            "n_references_found": n_total,
            "n_high_relevance": n_high_relevance,
            "references": top_refs,
            "query_formula": formula,
            "sources_queried": sources_queried,
            "from_cache": False,
            "_cached_at": time.time(),
        }

        _CACHE[cache_key] = result
        _save_cache()
        return result

    def _query_crossref(self, formula: str) -> List[Dict]:
        """Query Crossref works API for papers mentioning the formula."""
        normalized = _normalize_formula(formula)
        url = (
            f"https://api.crossref.org/works"
            f"?query={quote(normalized)}"
            f"&filter=type:journal-article"
            f"&rows=20"
            f"&select=DOI,title,published,container-title,author,abstract"
            f"&mailto={CROSSREF_MAILTO}"
        )

        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            return []

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        refs = []
        for item in items:
            doi = item.get("DOI", "")
            title_list = item.get("title", [])
            title = title_list[0] if title_list else ""
            year = None
            pub = item.get("published", {}).get("date-parts", [[None]])
            if pub and pub[0]:
                year = pub[0][0]
            journal_list = item.get("container-title", [])
            journal = journal_list[0] if journal_list else ""
            abstract = item.get("abstract", "")
            authors = item.get("author", [])
            author_str = ", ".join(
                f"{a.get('family', '')}, {a.get('given', '')[:1]}." 
                for a in authors[:3]
                if a.get("family")
            )
            refs.append({
                "doi": doi,
                "title": title,
                "year": year,
                "journal": journal,
                "authors": author_str,
                "abstract_snippet": abstract[:300] if abstract else "",
                "source": "crossref",
                "url": f"https://doi.org/{doi}" if doi else "",
                "relevance_score": 0.0,  # set by _score_and_sort
            })
        return refs

    def _query_semantic_scholar(self, formula: str) -> List[Dict]:
        """Query Semantic Scholar graph API for papers."""
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={quote(formula)}"
            f"&fields=title,year,externalIds,publicationTypes,authors,abstract,venue"
            f"&limit=20"
        )

        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            return []

        data = resp.json()
        papers = data.get("data", [])
        refs = []
        for p in papers:
            doi = p.get("externalIds", {}).get("DOI", "")
            year = p.get("year")
            title = p.get("title", "")
            venue = p.get("venue", "")
            abstract = p.get("abstract", "")
            authors = p.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            pub_types = p.get("publicationTypes", [])
            # Skip non-journal items (book chapters, patents)
            if pub_types and not any(t in ("JournalArticle", "Review") for t in pub_types):
                continue
            refs.append({
                "doi": doi,
                "title": title,
                "year": year,
                "journal": venue,
                "authors": author_str,
                "abstract_snippet": abstract[:300] if abstract else "",
                "source": "semantic_scholar",
                "url": f"https://doi.org/{doi}" if doi else f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                "relevance_score": 0.0,
            })
        return refs

    def _score_and_sort(self, formula: str, refs: List[Dict]) -> List[Dict]:
        """
        Score each reference by relevance to the queried formula.
        
        Scoring heuristic:
          +0.5 if formula appears verbatim in title
          +0.3 if element symbols from formula appear in title
          +0.2 if formula appears in abstract snippet
          +0.1 if published 2015 or later (recency bonus)
          
        Returns sorted list (highest relevance first).
        """
        formula_lower = formula.lower()
        # Extract element symbols from formula
        elements = re.findall(r"[A-Z][a-z]?", formula)
        elements_lower = [e.lower() for e in elements]

        for ref in refs:
            title_lower = (ref.get("title", "") or "").lower()
            abstract_lower = (ref.get("abstract_snippet", "") or "").lower()
            year = ref.get("year") or 0
            score = 0.0

            if formula_lower in title_lower:
                score += 0.5
            elif all(e in title_lower for e in elements_lower):
                score += 0.3

            if formula_lower in abstract_lower:
                score += 0.2
            elif all(e in abstract_lower for e in elements_lower):
                score += 0.1

            if year and int(year) >= 2015:
                score += 0.1

            ref["relevance_score"] = round(min(score, 1.0), 3)

        refs.sort(key=lambda r: r["relevance_score"], reverse=True)
        return refs

    def _no_requests_fallback(self, formula: str) -> Dict[str, Any]:
        return {
            "known_in_literature": None,
            "confidence": "unavailable",
            "n_references_found": 0,
            "n_high_relevance": 0,
            "references": [],
            "query_formula": formula,
            "sources_queried": [],
            "error": "requests library not available",
            "from_cache": False,
        }


# Module-level singleton
_literature_check_instance: Optional[LiteratureCheckService] = None


def get_literature_checker() -> LiteratureCheckService:
    global _literature_check_instance
    if _literature_check_instance is None:
        _literature_check_instance = LiteratureCheckService()
    return _literature_check_instance

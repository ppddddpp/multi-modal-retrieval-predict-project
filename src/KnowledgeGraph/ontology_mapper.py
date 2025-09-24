import difflib
import re
import time
import random
import os
from pathlib import Path
import json
import requests
from google import genai
from typing import Dict, List, Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONTO_DIR = BASE_DIR / "data" / "ontologies"
ONTO_DIR.mkdir(parents=True, exist_ok=True)

DOID_URL = "http://purl.obolibrary.org/obo/doid.obo"
DOID_PATH = ONTO_DIR / "doid.obo"

RADLEX_URL = "https://bioportal.bioontology.org/ontologies/RADLEX"  # metadata page
RADLEX_PATH = ONTO_DIR / "RadLex.owl"
load_dotenv()
API_KEY = os.getenv("BIOPORTAL_API_KEY")

class OntologyMapper:
    def __init__(self, use_doid: bool = True, use_radlex: bool = False,
                    use_bioportal: bool = True, bioportal_key: Optional[str] = None,
                    api_keys: Optional[List[str]] = None):
        self.name2id: Dict[str, str] = {}
        self.use_bioportal = use_bioportal
        self.bioportal_key = bioportal_key

        # Load Gemini keys
        raw = os.getenv("GEMINI_KEYS", "")
        env_keys = [k.strip() for k in raw.split(",") if k.strip()]
        self.gemini_keys = api_keys if api_keys else env_keys
        if not self.gemini_keys:
            print("[Gemini] Warning: no API keys provided")

        # Track current key index for round-robin usage
        self._key_index = 0

        # BioPortal
        self.bioportal_key = bioportal_key if bioportal_key is not None else os.getenv("BIOPORTAL_API_KEY")

        # Load ontologies
        if use_doid:
            if not DOID_PATH.exists():
                print(f"[OntologyMapper] Downloading Disease Ontology to {DOID_PATH}")
                self._download_file(DOID_URL, DOID_PATH)
            self._load_doid(DOID_PATH)

        if use_radlex:
            if not RADLEX_PATH.exists():
                print(f"[OntologyMapper] Please manually download RadLex OWL from {RADLEX_URL} into {RADLEX_PATH}")
            else:
                self._load_radlex(RADLEX_PATH)

    def _next_gemini_key(self) -> Optional[str]:
        """Rotate through Gemini API keys (round-robin)."""
        if not self.gemini_keys:
            return None
        key = self.gemini_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.gemini_keys)
        return key

    # ------------------------- Loaders -------------------------
    def _download_file(self, url: str, out_path: Path):
        r = requests.get(url)
        r.raise_for_status()
        out_path.write_text(r.text, encoding="utf8")

    def _load_doid(self, path: Path):
        cur_id, cur_syns = None, []
        with open(path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("id: DOID:"):
                    cur_id = line.split("id: ")[1]
                    cur_syns = []
                elif line.startswith("name:"):
                    name = line.split("name: ")[1].lower()
                    ont_id = cur_id if cur_id and cur_id.startswith("DOID:") else (f"DOID:{cur_id}" if cur_id else None)
                    if ont_id:
                        self.name2id[name] = ont_id
                elif line.startswith("synonym:"):
                    parts = line.split("\"")
                    if len(parts) > 1:
                        cur_syns.append(parts[1].lower())
                elif line == "" and cur_id:
                    ont_id = cur_id if cur_id.startswith("DOID:") else f"DOID:{cur_id}"
                    for syn in cur_syns:
                        self.name2id[syn] = ont_id
                    cur_id, cur_syns = None, []

    def _load_radlex(self, path: Path):
        # Minimal OWL parser: extract "RIDxxx" IDs and labels
        with open(path, encoding="utf8") as f:
            content = f.read()
            entries = re.findall(
                r'<Class rdf:about=".*?(RID\d+)".*?>.*?<rdfs:label>(.*?)</rdfs:label>',
                content,
                re.S
            )
            for rid, label in entries:
                self.name2id[label.lower()] = f"RADLEX:{rid}"

    def _search_bioportal(self, term: str, ontology: str = "SNOMEDCT") -> Optional[str]:
        """Query BioPortal for a term in SNOMED CT (or other ontologies), with local caching."""
        if not self.bioportal_key:
            return None

        cache_file = ONTO_DIR / "bioportal_cache.json"
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())
        else:
            cache = {}

        # Return cached result if available (even None)
        if term in cache:
            return cache[term]

        url = "https://data.bioontology.org/search"
        params = {
            "q": term,
            "ontologies": ontology,
            "apikey": self.bioportal_key
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            bio_id = None
            if data.get("collection"):
                best = data["collection"][0]
                bio_id = best.get("@id")  # BioPortal URI (e.g. SNOMED concept)

            # Save result (even None to avoid re-querying)
            cache[term] = bio_id
            cache_file.write_text(json.dumps(cache, indent=2))

            return bio_id
        except Exception as e:
            print(f"[BioPortal] Failed for '{term}': {e}")
            cache[term] = None
            cache_file.write_text(json.dumps(cache, indent=2))
            return None
    
    def _sleep_backoff(self, base: float, attempt: int, cap: float = 60.0):
        """Exponential backoff with jitter, capped at 60s."""
        wait = min(cap, base * (2 ** attempt) + random.uniform(0, base))
        print(f"[Backoff] Sleeping {wait:.1f}s before retry…")
        time.sleep(wait)

    def normalize_with_gemini(self, term: str, max_retries: int = None, base_sleep: float = 2.0) -> Optional[str]:
        """
        Normalize a free-text clinical label using Gemini.
        - Caches results
        - Retries with exponential backoff
        - Disables invalid/expired keys automatically
        """
        if not self.gemini_keys:
            return None

        cache_file = ONTO_DIR / "gemini_cache.json"
        if cache_file.exists():
            cache = json.loads(cache_file.read_text())
        else:
            cache = {}

        if term in cache:
            return cache[term]

        if max_retries is None:
            max_retries = len(self.gemini_keys)  # try each key once

        tried = 0
        result = None
        while tried < max_retries and self.gemini_keys:
            api_key = self._next_gemini_key()
            try:
                client = genai.Client(api_key=api_key)
                prompt = (
                    "You are a clinical terminology assistant. "
                    "Given the following free-text label, return the closest "
                    "canonical disease/finding name from standard ontologies "
                    "(SNOMED CT, DOID, or RadLex). "
                    "Respond ONLY with the cleaned term, no explanation.\n\n"
                    "Return ONLY the exact SNOMED CT preferred term for the following clinical finding.\n\n"
                    f"Label: {term}"
                )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                cleaned = resp.text.strip()
                if cleaned:
                    result = cleaned
                    break
            except Exception as e:
                err = str(e)
                print(f"[Gemini] Key {api_key[:4]}… failed for '{term}': {err}")

                # Handle specific error types
                if "API_KEY_INVALID" in err or "expired" in err:
                    print(f"[Gemini] Removing invalid key {api_key[:4]}…")
                    self.gemini_keys.remove(api_key)
                    if not self.gemini_keys:
                        break
                elif "RESOURCE_EXHAUSTED" in err or "RATE_LIMIT_EXCEEDED" in err:
                    self._sleep_backoff(base=base_sleep, attempt=tried)
                else:
                    time.sleep(random.uniform(1, 3))

            tried += 1

        cache[term] = result
        cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

        if result:
            return result

        print(f"[Gemini] All {len(self.gemini_keys)} keys failed for '{term}'")
        return None
    
    # ------------------------- Core mapping -------------------------
    def expand_label(self, lbl: str) -> List[str]:
        """Expand compound group labels into atomic pieces."""
        parts = re.split(r"[\/&\-\(\)]", lbl)
        return [p.strip().lower() for p in parts if p.strip()]

    def map_labels(self, labels: List[str]) -> Dict[str, Optional[str]]:
        mapping = {}
        for lbl in labels:
            candidates = self.expand_label(lbl)
            matched_ids = []

            for cand in candidates:
                # 1. Dictionary lookup
                if cand in self.name2id:
                    matched_ids.append(self.name2id[cand])
                    continue

                # 2. Fuzzy match
                norm2 = re.sub(r"[^a-z0-9 ]", "", cand)
                if norm2 in self.name2id:
                    matched_ids.append(self.name2id[norm2])
                    continue

                close = difflib.get_close_matches(cand, self.name2id.keys(), n=1, cutoff=0.8)
                if close:
                    matched_ids.append(self.name2id[close[0]])
                    continue

                # 3. BioPortal search
                bio_id = None
                if self.use_bioportal:
                    print(f"[BioPortal] Searching for '{cand}'")
                    bio_id = self._search_bioportal(cand, ontology="SNOMEDCT")
                    if bio_id:
                        matched_ids.append(bio_id)
                        continue

                # 4. Gemini normalization + retry BioPortal
                normalized = self.normalize_with_gemini(cand)
                if normalized and normalized.lower() != cand.lower():
                    print(f"[Gemini] Normalized '{cand}' -> '{normalized}'")
                    # try dictionary again
                    if normalized.lower() in self.name2id:
                        matched_ids.append(self.name2id[normalized.lower()])
                    else:
                        bio_id = self._search_bioportal(normalized, ontology="SNOMEDCT")
                        if bio_id:
                            matched_ids.append(bio_id)

            # assign
            if not matched_ids:
                mapping[lbl] = None
            elif len(matched_ids) == 1:
                mapping[lbl] = matched_ids[0]
            else:
                mapping[lbl] = matched_ids
        return mapping

    # ------------------------- Group-level mapping -------------------------
    def map_grouped_labels(self, groups: Dict[str, List[str]], 
                        auto_save: bool = True, 
                        out_path: Path = ONTO_DIR / "nested_label2ontology.json"
    ) -> Dict[str, Dict[str, str]]:
        """
        Map each element inside groups (like disease_groups).
        Returns nested dict: { group_name: { element: ontology_id or LOCAL:... } }
        """
        nested_mapping = {}
        for group_name, elements in groups.items():
            nested_mapping[group_name] = {}
            for el in elements:
                result = self.map_labels([el])
                ont_id = result[el]
                if ont_id is None:  # fallback to pseudo-ID
                    ont_id = f"LOCAL:{el.replace(' ', '_').upper()}"
                nested_mapping[group_name][el] = ont_id

        if auto_save:
            with open(out_path, "w", encoding="utf8") as f:
                json.dump(nested_mapping, f, indent=2, ensure_ascii=False)
            print(f"[OntologyMapper] Nested mapping saved to {out_path}")

        return nested_mapping
    
    # ------------------------- Reporting -------------------------
    def report_group_coverage(self, nested_mapping: Dict[str, Dict[str, str]]):
        """ Print coverage stats per group. """
        print("\n=== Group Coverage Report ===")
        for group_name, mapping in nested_mapping.items():
            total = len(mapping)
            matched = sum(1 for v in mapping.values() if v and not str(v).startswith("LOCAL:"))
            coverage = matched / total if total > 0 else 0.0
            print(f"{group_name}: {matched}/{total} mapped ({coverage:.1%})")

    # ------------------------- Save -------------------------
    def save_mapping(self, nested_mapping: Dict[str, Dict[str, str]], out_path: Path = ONTO_DIR / "nested_label2ontology.json"):
        with open(out_path, "w", encoding="utf8") as f:
            json.dump(nested_mapping, f, indent=2, ensure_ascii=False)
        print(f"[OntologyMapper] Nested mapping saved to {out_path}")
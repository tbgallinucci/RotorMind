import os
import re
from pathlib import Path
from typing import Optional
import tiktoken
from openai import OpenAI

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
WIKI_DIR = BASE_DIR / "wiki"
RAW_DIR = BASE_DIR / "raw"
INDEX_FILE = WIKI_DIR / "index.md"

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────

# Defaults target a local LM Studio server. Override with environment variables.
# For Gemini, set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL_NAME (never commit real keys).
LLM_BASE_URL   = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY    = os.getenv("LLM_API_KEY", "lm-studio")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen2.5-14b-instruct-1m")

# Token budget
MAX_TOTAL_TOKENS   = 8_000
SYSTEM_TOKENS      = 1_000
HISTORY_TOKENS     = 1_000
CONTEXT_BUDGET     = MAX_TOTAL_TOKENS - SYSTEM_TOKENS - HISTORY_TOKENS
MIN_SECTION_TOKENS = 200

# tiktoken downloads the encoding on first use; guard so imports and tests
# never fail offline. Fallback: crude ~4 chars/token estimate.
try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKENIZER = None

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", 
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", 
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

SYSTEM_PROMPT = """\
You are RotorMind, a precise rotordynamics engineering copilot that answers strictly \
from the project's knowledge library — theory notes, standards summaries, glossaries, \
and simulation run reports — supplied to you as the wiki context below. You never rely \
on outside knowledge.

RULES:
1. STRICT GROUNDING: Answer ONLY using the wiki context provided below. NEVER use pre-training data or outside knowledge to fill gaps. If a specific value, step, rule, ingredient, or detail is not in the context, do not guess — state that it is missing.
2. MISSING CONTEXT: If the answer is not in the provided context, state: "This information is not available in the provided context." Then suggest which wiki page(s) might contain it.
3. CITATION — MANDATORY: Append the source to EVERY factual claim using this EXACT format: (wiki: [slug], [Section]).
   - The [slug] MUST be taken from the "### WIKI PAGE: [slug]" header in the context.
   - Cite the section, table, or heading the fact came from.
   - Example: "The film cavitates in the divergent region (wiki: reynolds-equation, Film thickness)."
   - The bracketed table prefix inside the data (e.g. [Table 2.1]) is for your internal matching ONLY. Do NOT print it as the citation.
4. TABLE & LIST READING:
   Tabular data is flattened into explicit key=value lines, each prefixed with its source table in brackets.
   a) Find the line that matches the queried parameters EXACTLY.
   b) Quote that exact line verbatim on its own line before giving the answer.
   c) If the exact row is not present, DO NOT invent one — state that it is missing.
   d) NEVER read a value from a row whose bracket prefix does not match the table you identified.
5. EXACT VALUES: Report the exact value, quantity, time, temperature, or wording from the source. Never approximate or round unless the user asks.
6. NOTES, CONDITIONS & LIMITS: If a table or section carries a note, condition, exclusion, or limit (e.g. TABLE_NOTES), check it first and quote it whenever it affects the answer.
7. INTERPOLATION / SCALING: Only when the document supports it (e.g. scaling a recipe up or down, reading between two listed values), you may interpolate or scale. Quote the source rows you used, show the arithmetic, and state clearly that the result is calculated, not quoted. Do not extrapolate beyond the listed range.
8. UNITS: Use the source's native units. If the user asks for a different unit, look up the native value first, then convert and show both.
9. TONE: Clear, friendly, and precise. Get to the answer without filler or invented detail. Use LaTeX ($ ... $ inline, $$ ... $$ block) only when showing a calculation.
"""

def count_tokens(text: str) -> int:
    if TOKENIZER is None:
        return max(1, len(text) // 4)
    return len(TOKENIZER.encode(text, disallowed_special=()))

def trim_to_token_budget(text: str, budget: int) -> str:
    if TOKENIZER is None:
        if max(1, len(text) // 4) <= budget:
            return text
        return text[: budget * 4] + "\n\n[... content truncated to fit context window ...]"
    tokens = TOKENIZER.encode(text, disallowed_special=())
    if len(tokens) <= budget:
        return text
    return TOKENIZER.decode(tokens[:budget]) + "\n\n[... content truncated to fit context window ...]"

def flatten_tables_generic(text: str) -> str:
    lines   = text.splitlines(keepends=True)
    output  = []
    i       = 0
    n       = len(lines)
    current_heading = ""

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if re.match(r"^#{1,3} ", stripped):
            current_heading = stripped.lstrip("#").strip()

        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3:
            header_cells = [c.strip() for c in stripped.split("|")[1:-1]]
            sep_idx = i + 1
            if sep_idx < n and re.match(r"^\|[-| :]+\|$", lines[sep_idx].strip()):
                i = sep_idx + 1
                row_prefix = f"[{current_heading}] " if current_heading else ""
                
                # Dynamic header cleaning for key-value mapping
                clean_headers = []
                for h in header_cells:
                    h_clean = re.sub(r"[\(\[].*?[\)\]]", "", h) # remove units in brackets
                    h_clean = re.sub(r"[^a-zA-Z0-9\s]", "", h_clean).strip()
                    h_clean = re.sub(r"\s+", "_", h_clean)
                    clean_headers.append(h_clean)

                while i < n:
                    data_line = lines[i].strip()
                    if not (data_line.startswith("|") and data_line.endswith("|")):
                        break
                    data_cells = [c.strip() for c in data_line.split("|")[1:-1]]
                    if len(data_cells) != len(header_cells):
                        output.append(lines[i])
                        i += 1
                        continue

                    parts = []
                    for hdr, val in zip(clean_headers, data_cells):
                        parts.append(f"{hdr}={val}")

                    output.append(row_prefix + " | ".join(parts) + "\n")
                    i += 1

                j = i
                while j < n and lines[j].strip() == "":
                    j += 1
                note_lines = []
                while j < n and re.match(r"^NOTES?\s*:", lines[j].strip(), re.IGNORECASE):
                    note_lines.append(lines[j].strip())
                    j += 1

                if note_lines:
                    notes_text = " | ".join(note_lines)
                    output.append(f"{row_prefix}TABLE_NOTES: {notes_text}\n")
                    i = j
                else:
                    output.append(f"{row_prefix}TABLE_NOTES: none\n")
                continue

        output.append(line)
        i += 1

    return "".join(output)

def load_wiki_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not INDEX_FILE.exists():
        return index
    try:
        content = INDEX_FILE.read_text(encoding="utf-8")
        for line in content.splitlines():
            # OKF format: * [Title](slug.md) - description
            m = re.match(r"^\s*\*\s*\[[^\]]*\]\(([^)]+?)\)\s*-\s*(.+?)\s*$", line)
            if m:
                slug = m.group(1).rsplit("/", 1)[-1]
                if slug.endswith(".md"):
                    slug = slug[:-3]
                index[slug] = m.group(2).strip()
                continue
            # Legacy format: | [[page-name]] | description |
            m = re.search(r"\|\s*\[\[([^\]]+)\]\]\s*\|\s*(.+?)\s*(?:\||$)", line)
            if m:
                index[m.group(1)] = m.group(2).strip()
    except Exception:
        pass
    return index

def load_page(slug: str) -> Optional[str]:
    path = WIKI_DIR / f"{slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # Fallback: pages that live in subfolders (e.g. simulation runs in wiki/runs/)
    # are indexed by bare slug; find them recursively.
    matches = [m for m in WIKI_DIR.rglob(f"{slug}.md")]
    if matches:
        return matches[0].read_text(encoding="utf-8")
    return None

def split_into_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    parts = re.split(r"(?m)^(#{1,3} .+)$", content)

    if parts[0].strip():
        sections.append(("__preamble__", parts[0]))

    it = iter(parts[1:])
    for heading in it:
        body = next(it, "")
        sections.append((heading.strip(), body))

    return sections

def extract_query_numbers(query_lower: str) -> list[float]:
    """Extracts standalone numbers or decimals, excluding those attached to letters."""
    # Find numbers that are NOT preceded or followed by letters (identifiers)
    return [float(n) for n in re.findall(r"(?<![a-zA-Z])(\d+(?:\.\d+)?)(?![a-zA-Z])", query_lower)]

def score_section_generic(heading: str, body: str, query_lower: str, target_numbers: list[float] = None) -> float:
    # 1. Word matching
    query_words = set(re.findall(r"\b\w+\b", query_lower)) - STOPWORDS
    heading_lower = heading.lower()
    body_lower = body.lower()
    
    # Alphanumeric identifiers (e.g., CMFS010) deserve huge boosts
    identifiers = {w for w in query_words if any(c.isdigit() for c in w) and any(c.isalpha() for c in w)}
    
    # Base score from word hits (with simple fuzzy matching for plurals)
    score = 0.0
    for qw in query_words:
        # Check heading
        if qw in heading_lower or (qw.endswith('s') and qw[:-1] in heading_lower) or (not qw.endswith('s') and qw+'s' in heading_lower):
            score += 10.0
        # Check body
        if qw in body_lower or (qw.endswith('s') and qw[:-1] in body_lower) or (not qw.endswith('s') and qw+'s' in body_lower):
            score += 2.0
    
    # 2. Identifier Partial Match (CMFS010 should match CMFS010H/P)
    for ident in identifiers:
        if ident in heading_lower:
            score += 500.0 # Massive boost for identifier in heading
        elif ident in body_lower:
            score += 250.0 # Huge boost for identifier in body

    # 3. Numerical boost (exact standalone numbers like 600 for Class 600)
    if target_numbers:
        for num in target_numbers:
            num_str = str(int(num)) if num.is_integer() else str(num)
            if re.search(fr"(?<![a-zA-Z0-9]){re.escape(num_str)}(?![a-zA-Z0-9])", body_lower):
                score += 100.0

    return score

def extract_relevant_sections(content: str, query: str, budget: int) -> str:
    content = flatten_tables_generic(content)

    if count_tokens(content) <= budget:
        return content

    query_lower  = query.lower()
    target_numbers = extract_query_numbers(query_lower)
    sections     = split_into_sections(content)

    scored = []
    for heading, body in sections:
        section_text = (heading + "\n" + body).strip()
        if count_tokens(section_text) < MIN_SECTION_TOKENS and heading == "__preamble__":
            scored.append((999.0, heading, section_text))
        else:
            score = score_section_generic(heading, body, query_lower, target_numbers)
            scored.append((score, heading, section_text))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected_parts: list[str] = []
    used_tokens = 0

    for score, heading, text in scored:
        tok = count_tokens(text)
        if used_tokens + tok <= budget:
            selected_parts.append(text)
            used_tokens += tok
        else:
            remaining = budget - used_tokens
            if remaining > MIN_SECTION_TOKENS:
                selected_parts.append(trim_to_token_budget(text, remaining))
            break

    if not selected_parts:
        return trim_to_token_budget(content, budget)

    banner = (
        f"\n\n> ⚠️ **Large page — only the most relevant sections are shown "
        f"({used_tokens}/{count_tokens(content)} tokens).**\n\n"
    )
    return banner + "\n\n---\n\n".join(selected_parts)

def rank_pages(query: str, index: dict[str, str]) -> list[tuple[str, float]]:
    query_lower = query.lower()
    query_words = set(re.findall(r"\b\w+\b", query_lower)) - STOPWORDS

    results: list[tuple[str, float]] = []
    for slug, description in index.items():
        combined = (slug + " " + description).lower()
        combined_words = set(re.findall(r"\b\w+\b", combined)) - STOPWORDS
        if not query_words:
            score = 0.0
        else:
            score = len(query_words & combined_words) / max(len(query_words), 1)
        results.append((slug, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def build_context(query: str, index: dict[str, str], budget: int) -> str:
    ranked_pages = rank_pages(query, index)
    query_lower = query.lower()
    target_numbers = extract_query_numbers(query_lower)
    
    all_scored_sections = []
    
    # 1. Pool sections from all relevant pages
    for slug, page_score in ranked_pages:
        if page_score == 0.0:
            continue
        
        content = load_page(slug)
        if not content:
            continue
        
        # Flatten tables and split
        flattened = flatten_tables_generic(content)
        sections = split_into_sections(flattened)
        
        for heading, body in sections:
            section_text = f"### WIKI PAGE: {slug}\n## {heading}\n{body}".strip()
            # If it's a preamble, give it a baseline boost if the page matched
            if heading == "__preamble__":
                score = 50.0 + page_score * 10.0
            else:
                score = score_section_generic(heading, body, query_lower, target_numbers)
            
            if score > 0:
                all_scored_sections.append((score, section_text))

    # 2. Sort all sections GLOBALLY by score
    all_scored_sections.sort(key=lambda x: x[0], reverse=True)

    # 3. Fill budget
    context_parts: list[str] = []
    used = 0
    for score, text in all_scored_sections:
        tok = count_tokens(text)
        if used + tok <= budget:
            context_parts.append(text)
            used += tok
        else:
            remaining = budget - used
            if remaining > MIN_SECTION_TOKENS:
                context_parts.append(trim_to_token_budget(text, remaining))
            break

    if not context_parts:
        idx_text = INDEX_FILE.read_text(encoding="utf-8") if INDEX_FILE.exists() else ""
        return "### WIKI INDEX (no specific section matched)\n\n" + trim_to_token_budget(idx_text, budget)

    return "\n\n" + ("=" * 60) + "\n\n" + "\n\n---\n\n".join(context_parts)

def trim_history(history: list[dict], budget: int) -> list[dict]:
    total = 0
    kept: list[dict] = []
    for msg in reversed(history):
        t = count_tokens(msg["content"])
        if total + t > budget and kept:
            break
        kept.insert(0, msg)
        total += t
    return kept

def build_index_tree():
    """
    Parses wiki/index.md and builds a hierarchical tree based on headings.
    Example heading: ## Manuals / Flow Meter and Prover / Datasheet.pdf
    """
    if not INDEX_FILE.exists():
        return []

    content = INDEX_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    root = []
    current_pdf_path = ""
    slug_to_pdf = {}

    def add_to_tree(tree, path_parts, slug, description):
        if not path_parts:
            return
        
        part = path_parts[0].strip()
        
        # Check if it's the PDF file (last part of path)
        if len(path_parts) == 1:
            # Find if this PDF node already exists
            node = next((n for n in tree if n['name'] == part), None)
            if not node:
                node = {
                    "type": "pdf",
                    "name": part,
                    "children": []
                }
                tree.append(node)
            
            # Add the wiki page under this PDF
            node['children'].append({
                "type": "file",
                "name": slug,
                "slug": slug,
                "description": description
            })
            return

        # It's a directory
        node = next((n for n in tree if n['name'] == part), None)
        if not node:
            node = {
                "type": "directory",
                "name": part,
                "children": []
            }
            tree.append(node)
        
        add_to_tree(node['children'], path_parts[1:], slug, description)

    for line in lines:
        # Detect Heading
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            current_pdf_path = heading_match.group(1).strip()
            continue
        
        # Detect page row — OKF format: * [Title](slug.md) - description
        row_match = re.match(r"^\s*\*\s*\[[^\]]*\]\(([^)]+?)\)\s*-\s*(.+?)\s*$", line)
        if row_match and current_pdf_path:
            slug = row_match.group(1).rsplit("/", 1)[-1]
            if slug.endswith(".md"):
                slug = slug[:-3]
            desc = row_match.group(2).strip()
            add_to_tree(root, current_pdf_path.split("/"), slug, desc)
            slug_to_pdf[slug] = current_pdf_path
            continue

        # Legacy format: | [[slug]] | description |
        table_match = re.match(r"\|\s*\[\[([^\]]+)\]\]\s*\|\s*(.+?)\s*\|", line)
        if table_match and current_pdf_path:
            slug = table_match.group(1).strip()
            desc = table_match.group(2).strip()
            add_to_tree(root, current_pdf_path.split("/"), slug, desc)
            slug_to_pdf[slug] = current_pdf_path

    return root, slug_to_pdf

def full_text_search(query: str) -> list[dict]:
    """
    Search for query string in all wiki pages content.
    Returns list of {slug, snippet}.
    """
    results = []
    query_lower = query.lower()

    # Load index to get descriptions
    index = load_wiki_index()

    for file in WIKI_DIR.rglob("*.md"):
        if file.name in ("index.md", "log.md"):
            continue

        content = file.read_text(encoding="utf-8")
        if query_lower in content.lower():
            slug = file.stem
            idx = content.lower().find(query_lower)
            start = max(0, idx - 50)
            end = min(len(content), idx + 100)
            snippet = content[start:end].replace("\n", " ")
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."

            results.append({
                "slug": slug,
                "name": slug,
                "description": index.get(slug, ""),
                "snippet": snippet,
            })

    return results

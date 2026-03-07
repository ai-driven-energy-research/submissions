"""
AIDER Deep Review Agent

Runs after AI pre-screening passes. Reads the full manuscript and:
1. Extracts bold/priority claims ("first", "novel", "unprecedented", etc.)
2. Checks each claim's verifiability and flags those needing editor attention
3. Extracts references and checks DOI accessibility (open vs paywalled)
4. Posts a structured report as a GitHub issue comment

Usage:
    python deep_review.py --repo-url <url> --issue-number <n>

Requires:
    GROQ_API_KEY - Groq API key
    GITHUB_TOKEN - GitHub token with issues:write permission
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Patterns that signal a "big claim" in academic writing
CLAIM_PATTERNS = [
    r"(?i)\b(?:we are |this is |this paper is |this work is |this study is )?the first\b",
    r"(?i)\bfirst(?:\s+\w+){0,3}\s+(?:to |that |which )",
    r"(?i)\bnovel\b",
    r"(?i)\bunprecedented\b",
    r"(?i)\bstate[\s-]of[\s-]the[\s-]art\b",
    r"(?i)\boutperform(?:s|ed|ing)?\b",
    r"(?i)\bsuperior(?:\s+to)?\b",
    r"(?i)\bsignificantly\s+(?:better|improve|outperform|exceed|surpass)",
    r"(?i)\bno\s+(?:prior|previous|existing)\s+(?:work|study|research|paper)\b",
    r"(?i)\bto\s+(?:the|our)\s+(?:best\s+of\s+)?(?:our\s+)?knowledge\b",
    r"(?i)\bfirst[\s-]of[\s-]its[\s-]kind\b",
    r"(?i)\bground[\s-]?breaking\b",
    r"(?i)\bparadigm[\s-]shift\b",
    r"(?i)\brevolutionary\b",
    r"(?i)\btransformative\b",
    r"(?i)\bsurpass(?:es|ed|ing)?\b",
    r"(?i)\bdominate(?:s|d)?\b",
    r"(?i)\bonly\s+(?:work|study|approach|method|framework)\b",
]


def extract_claims(text: str) -> list[dict]:
    """Find sentences containing bold claims and return with context."""
    # Split into sentences (rough but good enough for LaTeX)
    # Remove LaTeX commands for matching but keep originals for context
    clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    clean = re.sub(r"\\[a-zA-Z]+", "", clean)
    clean = re.sub(r"[{}]", "", clean)

    sentences = re.split(r"(?<=[.!?])\s+", clean)

    claims = []
    seen = set()
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        for pattern in CLAIM_PATTERNS:
            if re.search(pattern, sent):
                # Deduplicate by first 60 chars
                key = sent[:60].lower()
                if key not in seen:
                    seen.add(key)
                    matched = re.search(pattern, sent).group(0)
                    claims.append({
                        "sentence": sent[:500],
                        "trigger": matched,
                    })
                break  # one match per sentence is enough

    return claims


# ---------------------------------------------------------------------------
# Reference extraction & accessibility check
# ---------------------------------------------------------------------------

def extract_references_from_bib(repo_dir: str) -> list[dict]:
    """Parse .bib files for references with DOIs/URLs."""
    root = Path(repo_dir)
    bib_files = list(root.rglob("*.bib"))
    refs = []

    for bib_path in bib_files:
        try:
            content = bib_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Split into entries
        entries = re.findall(
            r"@\w+\{([^,]+),\s*(.*?)\n\}",
            content,
            re.DOTALL,
        )

        for cite_key, body in entries:
            ref = {"key": cite_key.strip()}

            # Extract title
            title_m = re.search(r"title\s*=\s*[{\"](.*?)[}\"]", body, re.DOTALL)
            if title_m:
                ref["title"] = re.sub(r"\s+", " ", title_m.group(1)).strip()

            # Extract DOI
            doi_m = re.search(r"doi\s*=\s*[{\"](.*?)[}\"]", body, re.IGNORECASE)
            if doi_m:
                ref["doi"] = doi_m.group(1).strip()

            # Extract URL
            url_m = re.search(r"url\s*=\s*[{\"](.*?)[}\"]", body, re.IGNORECASE)
            if url_m:
                ref["url"] = url_m.group(1).strip()

            # Extract year
            year_m = re.search(r"year\s*=\s*[{\"]*(\d{4})[}\"]*", body)
            if year_m:
                ref["year"] = year_m.group(1)

            if ref.get("title"):
                refs.append(ref)

    return refs


def check_reference_accessibility(refs: list[dict], max_checks: int = 30) -> list[dict]:
    """Check if references are openly accessible or paywalled."""
    results = []

    for ref in refs[:max_checks]:
        result = {
            "key": ref["key"],
            "title": ref.get("title", "Unknown"),
            "year": ref.get("year", ""),
        }

        doi = ref.get("doi", "")
        url = ref.get("url", "")

        if doi:
            result["doi"] = doi
            # Check via DOI
            doi_url = f"https://doi.org/{doi}"
            try:
                resp = requests.head(
                    doi_url,
                    allow_redirects=True,
                    timeout=10,
                    headers={"User-Agent": "AIDER-Bot/1.0 (academic review)"},
                )
                final_url = resp.url

                # Heuristics for paywall detection
                paywall_domains = [
                    "sciencedirect.com", "springer.com", "link.springer",
                    "wiley.com", "tandfonline.com", "ieee.org/document",
                    "sagepub.com", "nature.com", "science.org",
                    "oup.com", "cambridge.org", "jstor.org",
                    "elsevier.com",
                ]
                open_domains = [
                    "arxiv.org", "openaccess", "mdpi.com", "plos.org",
                    "frontiersin.org", "biorxiv.org", "medrxiv.org",
                    "hal.science", "zenodo.org", "github.com",
                    "ssrn.com", "preprints.org",
                ]

                is_paywalled = any(d in final_url.lower() for d in paywall_domains)
                is_open = any(d in final_url.lower() for d in open_domains)

                if is_open:
                    result["access"] = "open"
                elif is_paywalled:
                    result["access"] = "likely-paywalled"
                else:
                    result["access"] = "unknown"

                result["resolved_url"] = final_url

            except Exception:
                result["access"] = "unreachable"

        elif url:
            result["url"] = url
            if "arxiv" in url.lower():
                result["access"] = "open"
            else:
                result["access"] = "unknown"
        else:
            result["access"] = "no-doi-or-url"

        results.append(result)
        time.sleep(0.3)  # Be polite to DOI servers

    return results


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

def analyze_claims_with_llm(claims: list[dict], abstract: str) -> str:
    """Send claims to Groq for analysis."""
    from groq import Groq

    if not claims:
        return "No bold claims detected in the manuscript."

    claims_text = ""
    for i, c in enumerate(claims[:20], 1):  # Cap at 20 claims
        claims_text += f"\n{i}. **Trigger:** \"{c['trigger']}\"\n   **Sentence:** \"{c['sentence']}\"\n"

    prompt = f"""You are a claim verification assistant for an academic journal (AIDER — AI-Driven Energy Research).

Below is the paper's abstract and a list of bold claims extracted from the manuscript.

For EACH claim, provide:
1. **Severity**: HIGH / MEDIUM / LOW
   - HIGH = "first ever", "no prior work", "unprecedented" — needs strong evidence or literature check
   - MEDIUM = "outperforms", "state-of-the-art", "significantly better" — needs baseline comparison
   - LOW = "novel approach", "transformative" — common academic language, minor concern
2. **What editor should verify**: one sentence on what specifically the editor should check
3. **Likely verifiable from paper alone?**: Yes / No / Partially

IMPORTANT RULES:
- Do NOT judge whether claims are true or false — you don't have the full context
- Focus on what the EDITOR needs to check, not on your own assessment
- Be specific: "check Table 3 for baseline comparison" not "verify results"
- If a claim is standard academic language with no factual assertion, mark it LOW

## Abstract
{abstract[:2000]}

## Extracted Claims
{claims_text}

Respond in this exact markdown format:

### Claim Verification Summary

| # | Trigger | Severity | What to Verify |
|---|---------|----------|----------------|
| 1 | "..." | HIGH/MED/LOW | ... |
...

### HIGH-Severity Claims (Editor Must Check)
[For each HIGH claim, explain in 1-2 sentences what evidence is needed and where to look]

### Notes
[Any overall observations about the claim patterns — e.g., "Paper makes 3 first-ever claims but cites no prior surveys in Section 2"]
"""

    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(
    claims: list[dict],
    claim_analysis: str,
    ref_results: list[dict],
    total_refs: int,
) -> str:
    """Build the final deep review report."""
    sections = []

    sections.append("## AI Deep Review — Claim & Reference Audit\n")
    sections.append(
        "> This automated review checks for bold claims that need editor verification "
        "and flags references that may be behind paywalls. It is NOT a peer review.\n"
    )

    # --- Claim analysis ---
    sections.append(f"### Claims Analysis ({len(claims)} bold claims detected)\n")
    sections.append(claim_analysis)
    sections.append("")

    # --- Reference accessibility ---
    paywalled = [r for r in ref_results if r["access"] == "likely-paywalled"]
    open_refs = [r for r in ref_results if r["access"] == "open"]
    no_doi = [r for r in ref_results if r["access"] == "no-doi-or-url"]
    unreachable = [r for r in ref_results if r["access"] == "unreachable"]
    unknown = [r for r in ref_results if r["access"] == "unknown"]

    sections.append("---\n")
    sections.append(f"### Reference Accessibility ({len(ref_results)} of {total_refs} checked)\n")
    sections.append(
        f"| Status | Count |\n|---|---|\n"
        f"| Open access | {len(open_refs)} |\n"
        f"| Likely paywalled | {len(paywalled)} |\n"
        f"| No DOI/URL | {len(no_doi)} |\n"
        f"| Unreachable | {len(unreachable)} |\n"
        f"| Unknown | {len(unknown)} |\n"
    )

    if paywalled:
        sections.append(
            "\n#### Paywalled References — Editor Must Download to Verify\n"
            "These references resolved to publisher sites that typically require "
            "institutional access. The editor should verify the cited content.\n"
        )
        for r in paywalled:
            doi_str = f"DOI: `{r.get('doi', 'N/A')}`" if r.get("doi") else ""
            sections.append(
                f"- **[{r['key']}]** {r['title']} ({r.get('year', '?')}) "
                f"— {doi_str} → {r.get('resolved_url', 'N/A')}"
            )
        sections.append("")

    if no_doi:
        sections.append(
            "\n#### References Without DOI or URL\n"
            "These entries have no DOI or URL in the .bib file. "
            "Editor may need to locate them manually.\n"
        )
        for r in no_doi[:10]:  # Cap display
            sections.append(f"- **[{r['key']}]** {r['title']} ({r.get('year', '?')})")
        if len(no_doi) > 10:
            sections.append(f"- ... and {len(no_doi) - 10} more")
        sections.append("")

    # --- Footer ---
    sections.append("---\n")
    sections.append(
        "*This is an automated claim & reference audit. "
        "The editor should use this to prioritise their review, not as a final judgement. "
        "HIGH-severity claims and paywalled references need human verification.*"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# GitHub posting
# ---------------------------------------------------------------------------

def post_github_comment(issue_number: int, body: str):
    """Post a comment on the GitHub issue."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "ai-driven-energy-research/submissions")

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={"body": body},
    )
    resp.raise_for_status()
    print(f"Posted deep review to issue #{issue_number}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_manuscript(repo_dir: str) -> str:
    """Read the main manuscript file."""
    root = Path(repo_dir)
    for pattern in ["paper/main.tex", "paper/main.md", "paper/*.tex", "paper/*.md"]:
        matches = list(root.glob(pattern))
        if matches:
            try:
                return matches[0].read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return ""


def extract_abstract(text: str) -> str:
    """Extract abstract from LaTeX or markdown."""
    # LaTeX
    m = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        text,
        re.DOTALL,
    )
    if m:
        abstract = m.group(1).strip()
        abstract = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", abstract)
        abstract = re.sub(r"[{}\\]", "", abstract)
        return abstract

    # Markdown
    m = re.search(r"#+\s*Abstract\s*\n(.*?)(?=\n#+|\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    return text[:2000]


def main():
    parser = argparse.ArgumentParser(description="AIDER Deep Review Agent")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Clone
    clone_dir = "/tmp/aider-deep-review"
    subprocess.run(["rm", "-rf", clone_dir], check=False)
    print(f"Cloning {args.repo_url}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", args.repo_url, clone_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to clone: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Read manuscript
    print("Reading manuscript...")
    manuscript = read_manuscript(clone_dir)
    if not manuscript:
        print("ERROR: No manuscript found", file=sys.stderr)
        sys.exit(1)

    # Extract claims
    print("Extracting claims...")
    claims = extract_claims(manuscript)
    print(f"  Found {len(claims)} bold claims")

    # Extract abstract for LLM context
    abstract = extract_abstract(manuscript)

    # Analyze claims with LLM
    print("Analyzing claims with LLM...")
    claim_analysis = analyze_claims_with_llm(claims, abstract)

    # Extract and check references
    print("Extracting references...")
    refs = extract_references_from_bib(clone_dir)
    print(f"  Found {len(refs)} references")

    print("Checking reference accessibility...")
    ref_results = check_reference_accessibility(refs)

    # Build report
    print("Building report...")
    report = build_report(claims, claim_analysis, ref_results, len(refs))

    if args.dry_run:
        print("\n" + report)
    else:
        post_github_comment(args.issue_number, report)

    # Cleanup
    subprocess.run(["rm", "-rf", clone_dir], check=False)
    print("Done.")


if __name__ == "__main__":
    main()

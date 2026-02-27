"""
AIDER AI Pre-Screening Agent

Reads a submission repository (paper, code, data, process log) and generates
a pre-screening report highlighting key points for human editors to focus on.

This is NOT a replacement for peer review. It performs basic checks, flags
potential issues, and saves editors time by surfacing what matters.

Usage:
    python review.py --repo-url <github-repo-url> --issue-number <n>

Requires:
    GROQ_API_KEY - Groq API key (free tier)
    GITHUB_TOKEN - GitHub token with issues:write permission
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def clone_repo(url: str, dest: str) -> bool:
    """Clone the submission repository."""
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, dest],
        capture_output=True, text=True
    )
    return result.returncode == 0


def read_file(path: Path, max_chars: int = 15000) -> str:
    """Read a file, truncating if too long."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[... truncated at {max_chars} chars ...]"
        return text
    except Exception:
        return ""


def collect_submission_content(repo_dir: str) -> dict:
    """Read the key files from a submission repository."""
    root = Path(repo_dir)
    content = {}

    # Paper - find main manuscript
    for pattern in ["paper/main.tex", "paper/main.md", "paper/*.tex", "paper/*.md"]:
        matches = list(root.glob(pattern))
        if matches:
            content["manuscript"] = read_file(matches[0])
            content["manuscript_file"] = str(matches[0].relative_to(root))
            break

    # Code - README and key source files
    code_readme = root / "code" / "README.md"
    if code_readme.exists():
        content["code_readme"] = read_file(code_readme)

    # Collect source files (up to 10, prioritise .py files)
    source_files = []
    code_dir = root / "code"
    if code_dir.exists():
        py_files = sorted(code_dir.rglob("*.py"))[:10]
        for f in py_files:
            source_files.append({
                "path": str(f.relative_to(root)),
                "content": read_file(f, max_chars=5000)
            })
    content["source_files"] = source_files

    # Dependencies
    for dep_file in ["code/requirements.txt", "code/pyproject.toml", "code/environment.yml"]:
        dep_path = root / dep_file
        if dep_path.exists():
            content["dependencies"] = read_file(dep_path, max_chars=3000)
            content["dependencies_file"] = dep_file
            break

    # Process log
    process_readme = root / "process-log" / "README.md"
    if process_readme.exists():
        content["process_log"] = read_file(process_readme)

    # AI session files
    ai_sessions_dir = root / "process-log" / "ai-sessions"
    if ai_sessions_dir.exists():
        ai_files = [f for f in ai_sessions_dir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        content["ai_session_count"] = len(ai_files)
        if ai_files:
            content["ai_session_sample"] = read_file(ai_files[0], max_chars=3000)

    # Human decisions files
    human_dir = root / "process-log" / "human-decisions"
    if human_dir.exists():
        human_files = [f for f in human_dir.rglob("*") if f.is_file() and f.name != ".gitkeep"]
        content["human_decision_count"] = len(human_files)

    # Reproducibility checklist
    repro = root / "REPRODUCIBILITY.md"
    if repro.exists():
        content["reproducibility_checklist"] = read_file(repro)

    # reproduce.sh
    reproduce_sh = root / "results" / "reproduce.sh"
    if reproduce_sh.exists():
        content["reproduce_script"] = read_file(reproduce_sh, max_chars=3000)

    # Data README
    data_readme = root / "data" / "README.md"
    if data_readme.exists():
        content["data_readme"] = read_file(data_readme)

    # Top-level README
    readme = root / "README.md"
    if readme.exists():
        content["readme"] = read_file(readme)

    # Count files for stats
    content["total_py_files"] = len(list((root / "code").rglob("*.py"))) if (root / "code").exists() else 0
    content["total_data_files"] = len(list((root / "data").rglob("*"))) if (root / "data").exists() else 0

    return content


def build_review_prompt(content: dict) -> str:
    """Build the prompt for the pre-screening report."""
    sections = []

    sections.append("""You are an AI pre-screening assistant for AIDER (AI-Driven Energy Research), an open-access academic journal for AI + energy research.

Your job is NOT to be the reviewer. Your job is to help the human editor by:
1. Performing basic checks on completeness and quality
2. Flagging potential issues or red flags the editor should investigate
3. Highlighting the key claims and contributions so the editor can quickly understand the paper
4. Checking code for obvious issues (hardcoded paths, missing error handling, suspicious patterns)
5. Assessing whether the process log is substantive or just boilerplate

Generate a pre-screening report in this exact format:

## AI Pre-Screening Report

### Paper Summary
[2-3 sentences: what is this paper about, what is the main contribution]

### Key Claims
[Bulleted list of the paper's main claims that the editor should verify during review]

### Completeness Check
| Item | Status | Notes |
|---|---|---|
| Manuscript | Present/Missing | [any issues] |
| Source code | Present/Missing | [file count, languages] |
| Data | Present/Missing | [size, format, accessibility] |
| Process log | Present/Missing | [substantive or boilerplate?] |
| AI session logs | Present/Missing | [number of files, appears genuine?] |
| Human decision log | Present/Missing | [number of entries] |
| reproduce.sh | Present/Missing | [does it look complete?] |
| Reproducibility checklist | Present/Missing | [all items checked?] |

### Code Observations
[Bulleted list of observations about the code — NOT a full code review, just things the editor should look at:]
- Any hardcoded paths or credentials?
- Are random seeds fixed?
- Does the code structure match the methodology described in the paper?
- Any obvious bugs or suspicious patterns?
- Are there tests?

### Process Log Assessment
[Is the process log substantive or just template boilerplate? Does it document real AI sessions and real human decisions, or is it generic filler? Be specific about what you found.]

### Attention Points for Editor
[Bulleted list of specific things the human editor should pay close attention to during their review. These are NOT judgements — they are flags:]
- e.g., "Section 3.2 claims 15% improvement over baseline but the comparison code is not included"
- e.g., "reproduce.sh only generates 2 of the 5 figures in the paper"
- e.g., "Process log mentions GPT-4 was used but no session logs are provided"
- e.g., "The dataset appears to be synthetic but this is not stated in the paper"

### Scope Check
[Does this paper fall within AIDER's scope (AI + energy systems)? Brief assessment.]

---
*This is an automated pre-screening report, not a peer review. It is intended to help the editor prioritise their attention. The editor will conduct the full review and make all editorial decisions.*

IMPORTANT RULES:
- You are a screening tool, NOT a reviewer. Do not make accept/reject recommendations.
- Do not fabricate details. If material is missing or truncated, say so.
- Be specific: reference file names, line numbers, or section titles.
- Focus on factual observations, not opinions on quality.
- If something looks suspicious, flag it neutrally — let the editor decide.""")

    if content.get("readme"):
        sections.append(f"\n---\n## README.md\n```\n{content['readme']}\n```")

    if content.get("manuscript"):
        sections.append(f"\n---\n## Manuscript ({content.get('manuscript_file', 'paper')})\n```\n{content['manuscript']}\n```")

    if content.get("code_readme"):
        sections.append(f"\n---\n## Code README\n```\n{content['code_readme']}\n```")

    if content.get("source_files"):
        sections.append(f"\n---\n## Source Code Files ({content.get('total_py_files', '?')} .py files total, showing up to 10)")
        for sf in content["source_files"]:
            sections.append(f"\n### {sf['path']}\n```python\n{sf['content']}\n```")

    if content.get("dependencies"):
        sections.append(f"\n---\n## Dependencies ({content.get('dependencies_file', '')})\n```\n{content['dependencies']}\n```")

    if content.get("reproduce_script"):
        sections.append(f"\n---\n## reproduce.sh\n```bash\n{content['reproduce_script']}\n```")

    if content.get("data_readme"):
        sections.append(f"\n---\n## Data README\n```\n{content['data_readme']}\n```")

    if content.get("process_log"):
        sections.append(f"\n---\n## Process Log README\n```\n{content['process_log']}\n```")

    sections.append(f"\n---\n## Process Log Stats")
    sections.append(f"- AI session files: {content.get('ai_session_count', 0)}")
    sections.append(f"- Human decision files: {content.get('human_decision_count', 0)}")
    if content.get("ai_session_sample"):
        sections.append(f"\n### Sample AI session (first file)\n```\n{content['ai_session_sample']}\n```")

    if content.get("reproducibility_checklist"):
        sections.append(f"\n---\n## Reproducibility Checklist\n```\n{content['reproducibility_checklist']}\n```")

    return "\n".join(sections)


def call_llm(prompt: str) -> str:
    """Send prompt to Groq and get the pre-screening report."""
    from groq import Groq

    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content


def post_github_comment(issue_number: int, body: str):
    """Post a comment on the GitHub issue."""
    import requests

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
    print(f"Posted pre-screening report to issue #{issue_number}")


def main():
    parser = argparse.ArgumentParser(description="AIDER AI Pre-Screening Agent")
    parser.add_argument("--repo-url", required=True, help="GitHub repo URL to review")
    parser.add_argument("--issue-number", type=int, required=True, help="Submission issue number")
    parser.add_argument("--dry-run", action="store_true", help="Print report to stdout instead of posting")
    args = parser.parse_args()

    # Clone
    clone_dir = "/tmp/aider-review-submission"
    subprocess.run(["rm", "-rf", clone_dir], check=False)
    print(f"Cloning {args.repo_url}...")
    if not clone_repo(args.repo_url, clone_dir):
        print("ERROR: Failed to clone repository", file=sys.stderr)
        sys.exit(1)

    # Collect content
    print("Reading submission content...")
    content = collect_submission_content(clone_dir)

    if not content.get("manuscript"):
        print("WARNING: No manuscript found in paper/", file=sys.stderr)

    # Build prompt and call LLM
    print("Generating pre-screening report...")
    prompt = build_review_prompt(content)
    report = call_llm(prompt)

    if args.dry_run:
        print("\n" + report)
    else:
        post_github_comment(args.issue_number, report)

    # Cleanup
    subprocess.run(["rm", "-rf", clone_dir], check=False)
    print("Done.")


if __name__ == "__main__":
    main()

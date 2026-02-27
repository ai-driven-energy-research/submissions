"""
AIDER AI Reviewer Agent

Reads a submission repository (paper, code, data, process log) and generates
a structured peer review using Claude. Posts the review as a GitHub Issue comment.

Usage:
    python review.py --repo-url <github-repo-url> --issue-number <n>

Requires:
    ANTHROPIC_API_KEY - Claude API key
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

    return content


def build_review_prompt(content: dict) -> str:
    """Build the prompt for Claude to generate a review."""
    sections = []

    sections.append("""You are an AI peer reviewer for AIDER (AI-Driven Energy Research), an open-access academic journal focused on AI applications in energy systems.

You are reviewing a submission. Based on the materials provided below, generate a structured peer review. Be specific, cite exact sections/lines where possible, and be constructive.

Your review must follow this exact format:

## AI Reviewer Report

### Summary
[2-3 sentence summary of the paper's contribution and your overall assessment]

### Methodology Assessment
- **Approach:** [Is the methodology appropriate for the research question? Are assumptions clearly stated?]
- **Technical correctness:** [Any errors, questionable assumptions, or missing justifications?]
- **Novelty:** [Does this make a new contribution, or is it incremental/derivative?]

### Code Quality Assessment
- **Readability:** [Is the code well-structured and documented?]
- **Completeness:** [Does the code cover all claims in the paper?]
- **Dependencies:** [Are dependencies reasonable and well-specified?]
- **Reproducibility script:** [Does reproduce.sh appear to cover all key results?]

### Data Assessment
- **Availability:** [Is data provided or linked?]
- **Documentation:** [Is the data format and source described?]
- **Sufficiency:** [Is there enough data to support the claims?]

### Process Log Assessment
- **AI tool documentation:** [Are AI tools and their usage documented?]
- **Human decision documentation:** [Are key human decisions recorded?]
- **Completeness:** [Could another researcher understand the full workflow?]

### Results Consistency
- **Claims vs evidence:** [Do the results in the paper match what the code appears to produce?]
- **Statistical validity:** [Are statistical claims supported? Are error bars / confidence intervals provided where needed?]
- **Figures and tables:** [Do they appear to be generated by the provided code?]

### Strengths
[Bulleted list of what the paper does well]

### Weaknesses
[Bulleted list of issues that need addressing]

### Recommendations
[Specific, actionable items for the authors]

### Overall Recommendation
[One of: **Accept**, **Minor Revision**, **Major Revision**, **Reject**]
[Brief justification for the recommendation]

---
*This review was generated by the AIDER AI Reviewer Agent (Claude). Human editors will verify this assessment and make the final editorial decision. Authors may respond to any point raised.*

IMPORTANT:
- Be fair and constructive. Point out strengths as well as weaknesses.
- If you cannot assess something because the material is missing or truncated, say so explicitly.
- Do not fabricate details. Only comment on what you can actually see in the provided materials.
- Be specific: reference file names, line numbers, or section titles where relevant.""")

    if content.get("readme"):
        sections.append(f"\n---\n## README.md\n```\n{content['readme']}\n```")

    if content.get("manuscript"):
        sections.append(f"\n---\n## Manuscript ({content.get('manuscript_file', 'paper')})\n```\n{content['manuscript']}\n```")

    if content.get("code_readme"):
        sections.append(f"\n---\n## Code README\n```\n{content['code_readme']}\n```")

    if content.get("source_files"):
        sections.append("\n---\n## Source Code Files")
        for sf in content["source_files"]:
            sections.append(f"\n### {sf['path']}\n```python\n{sf['content']}\n```")

    if content.get("dependencies"):
        sections.append(f"\n---\n## Dependencies ({content.get('dependencies_file', '')})\n```\n{content['dependencies']}\n```")

    if content.get("reproduce_script"):
        sections.append(f"\n---\n## reproduce.sh\n```bash\n{content['reproduce_script']}\n```")

    if content.get("data_readme"):
        sections.append(f"\n---\n## Data README\n```\n{content['data_readme']}\n```")

    if content.get("process_log"):
        sections.append(f"\n---\n## Process Log\n```\n{content['process_log']}\n```")

    if content.get("reproducibility_checklist"):
        sections.append(f"\n---\n## Reproducibility Checklist\n```\n{content['reproducibility_checklist']}\n```")

    return "\n".join(sections)


def call_claude(prompt: str) -> str:
    """Send prompt to Claude and get the review."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


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
    print(f"Posted review to issue #{issue_number}")


def main():
    parser = argparse.ArgumentParser(description="AIDER AI Reviewer Agent")
    parser.add_argument("--repo-url", required=True, help="GitHub repo URL to review")
    parser.add_argument("--issue-number", type=int, required=True, help="Submission issue number")
    parser.add_argument("--dry-run", action="store_true", help="Print review to stdout instead of posting")
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

    # Build prompt and call Claude
    print("Generating review with Claude...")
    prompt = build_review_prompt(content)
    review = call_claude(prompt)

    if args.dry_run:
        print("\n" + review)
    else:
        post_github_comment(args.issue_number, review)

    # Cleanup
    subprocess.run(["rm", "-rf", clone_dir], check=False)
    print("Done.")


if __name__ == "__main__":
    main()

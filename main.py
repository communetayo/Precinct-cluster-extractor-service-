"""
FastAPI service for extracting election-day clusters from COMELEC
"Project of Precinct" PDFs at batch scale - hundreds of files across
a province, not a one-off. Mirrors the existing voter-list extractor's
architecture (FastAPI + BackgroundTasks + direct Supabase REST calls)
for consistency with what's already deployed and understood.

Verified extraction logic - see extract_precinct_clusters.py for the
original single-file version this was built from, and its own notes
on the one known minor limitation (occasional stray text in the
voting_center display field, never in the core cluster/precinct/
voter-count data).

Deploy on Render, not Railway - this is occasional, per-need batch
work rather than an always-on service, and Render's free tier doesn't
expire the way Railway's trial credit does.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pdfplumber
import requests
import tempfile
import os
import re
import secrets
import string
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Precinct-Cluster-Extractor", version="1.0.0")

PRECINCT_RE = re.compile(r'^\d{4}[A-Z]$')
CLUSTER_NUM_RE = re.compile(r'^\d{1,3}$')

HEADER_LABELS = {
    'BARANGAY', 'NAME', 'AND', 'ADDRESS', 'OF', 'VOTING', 'CENTER',
    'ESTABLISHED', 'PRECINCT', 'NUMBER', 'TOTAL', 'NO.', 'REGISTERED',
    'VOTERS', 'CLUSTER/', 'GROUP', 'CLUSTERED/', 'GROUPED', 'NO.OF',
    'REG', 'AFTER', 'CLUSTERING/', 'GROUPING',
}
NOISE_WORDS = {
    'PROVINCE', 'MAGUINDANAO', 'DEL', 'NORTE', 'CITY', 'MUNICIPALITY',
    'Republic', 'of', 'the', 'Philippines', 'COMMISSION', 'ON',
    'ELECTIONS', 'SEPTEMBER', 'BARMM', 'PARLIAMENTARY', 'AM', 'PM',
}


def generate_access_code(length=6):
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def is_noise(text):
    return (
        (len(text) == 1 and text.isalpha() and text.isupper())
        or text in NOISE_WORDS
        or re.match(r'^\d{1,2}-\w{3}-\d{2}$', text)
        or re.match(r'^\d{2}:\d{2}:\d{2}', text)
    )


def extract_clusters(pdf_path):
    clusters = []
    current_barangay = None
    voting_center_words = []
    pending_cluster = None
    subtotal_y = None
    stop_extraction = False
    city_name = None
    province_name = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if stop_extraction:
                break
            page_done = False
            words = page.extract_words(keep_blank_chars=False)
            words.sort(key=lambda w: (w['top'], w['x0']))

            page_text = page.extract_text() or ''
            if city_name is None:
                m = re.search(r'CITY\s*/\s*MUNICIPALITY\s*:\s*(.+)', page_text)
                if m:
                    city_name = m.group(1).strip().split('\n')[0]
            if province_name is None:
                m = re.search(r'PROVINCE\s*:\s*(.+)', page_text)
                if m:
                    province_name = m.group(1).strip().split('\n')[0]

            for w in words:
                if page_done:
                    continue
                x, y, text = w['x0'], w['top'], w['text']

                if text == 'Date:':
                    page_done = True
                    continue
                if text == 'Total':
                    stop_extraction = True
                    break
                if text == 'SUBTOTAL':
                    subtotal_y = y
                    if pending_cluster:
                        clusters.append(pending_cluster)
                        pending_cluster = None
                    voting_center_words.clear()
                    continue
                if subtotal_y is not None and abs(y - subtotal_y) < 5:
                    continue
                if text in HEADER_LABELS:
                    continue

                if x < 45 and not any(c.isdigit() for c in text):
                    if pending_cluster:
                        clusters.append(pending_cluster)
                        pending_cluster = None
                    current_barangay = text
                    voting_center_words.clear()
                    continue

                if 48 <= x < 230:
                    if is_noise(text):
                        continue
                    voting_center_words.append(text)
                    current_name = ' '.join(voting_center_words).strip()
                    if pending_cluster:
                        pending_cluster['voting_center'] = current_name
                    for c in reversed(clusters):
                        if c['barangay'] != current_barangay:
                            break
                        c['voting_center'] = current_name
                    continue

                if 225 <= x < 290 and PRECINCT_RE.match(text):
                    continue
                if 295 <= x < 370:
                    continue

                if 372 <= x < 400 and CLUSTER_NUM_RE.match(text):
                    if pending_cluster:
                        clusters.append(pending_cluster)
                    pending_cluster = {
                        'cluster_number': text,
                        'barangay': current_barangay,
                        'voting_center': ' '.join(voting_center_words).strip(),
                        'grouped_words': [],
                        'total': None,
                    }
                    continue

                if 405 <= x < 485:
                    if pending_cluster:
                        pending_cluster['grouped_words'].append(text)
                    continue
                if 505 <= x < 545:
                    if pending_cluster:
                        pending_cluster['total'] = text
                    continue

        if pending_cluster:
            clusters.append(pending_cluster)

    results = []
    for c in clusters:
        codes = ' '.join(c['grouped_words']).replace(' ', '').split(',')
        codes = [x for x in codes if x]
        results.append({
            'cluster_number': c['cluster_number'],
            'established_precinct_codes': codes,
            'voting_center_name': c['voting_center'],
            'barangay_name': c['barangay'],
            'city_name': city_name,
            'province_name': province_name,
            'total_registered_voters':
                int(c['total'].replace(',', '')) if c['total'] else None,
            'access_code': generate_access_code(),
        })
    return results, city_name, province_name


def process_in_background(pdf_url, import_id, supabase_url, supabase_key,
                            source_filename):
    logger.info(f"Import started: {import_id} ({source_filename})")
    tmp_path = None
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(pdf_url, timeout=120)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        clusters, city_name, province_name = extract_clusters(tmp_path)

        # A prior COMPLETED import for the same city - surfaced as a
        # warning inside the final status update, never a hard block,
        # matching the same philosophy as the duplicate-submission
        # handling elsewhere in this module. Someone re-running the
        # same city's PDF a second time is a real, ordinary case (a
        # corrected file, a retry) - it shouldn't be prevented, just
        # visible.
        dup_check = requests.get(
            f"{supabase_url}/rest/v1/election_day_pdf_imports",
            params={
                "city_name": f"eq.{city_name}",
                "status": "eq.completed",
                "select": "id,created_at",
            },
            headers=headers,
            timeout=30,
        )
        prior_imports = dup_check.json() if dup_check.status_code == 200 else []

        insert_res = requests.post(
            f"{supabase_url}/rest/v1/election_day_clusters",
            json=clusters,
            headers={**headers, "Prefer": "return=minimal"},
            timeout=60,
        )
        if insert_res.status_code not in (200, 201):
            raise RuntimeError(f"Cluster insert failed: {insert_res.text[:300]}")

        total_rv = sum(c['total_registered_voters'] for c in clusters
                        if c['total_registered_voters'])
        total_precincts = sum(len(c['established_precinct_codes']) for c in clusters)

        warning = None
        if prior_imports:
            warning = (f"{len(prior_imports)} prior completed import(s) already "
                       f"exist for '{city_name}' - check for duplicates before "
                       f"generating access codes for real use.")

        requests.patch(
            f"{supabase_url}/rest/v1/election_day_pdf_imports?id=eq.{import_id}",
            json={
                "status": "completed",
                "city_name": city_name,
                "province_name": province_name,
                "clusters_extracted": len(clusters),
                "total_registered_voters": total_rv,
                "total_established_precincts": total_precincts,
                "error_message": warning,
                "completed_at": "now()",
            },
            headers=headers,
            timeout=30,
        )
        logger.info(f"Import completed: {import_id} - {len(clusters)} clusters")

    except Exception as e:
        logger.error(f"Import failed: {import_id} - {e}")
        try:
            requests.patch(
                f"{supabase_url}/rest/v1/election_day_pdf_imports?id=eq.{import_id}",
                json={"status": "error", "error_message": str(e)},
                headers=headers,
                timeout=10,
            )
        except Exception:
            pass
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


class ExtractRequest(BaseModel):
    pdf_url: str
    supabase_url: str
    supabase_key: str
    source_filename: str | None = None


@app.get("/")
def health_check():
    return {"status": "ok", "service": "precinct-cluster-extractor"}


@app.post("/extract")
async def extract(req: ExtractRequest, background_tasks: BackgroundTasks):
    """One call per PDF - for hundreds of files, loop this call from
    your own script or admin tool, one pdf_url per city/municipality."""
    headers = {
        "apikey": req.supabase_key,
        "Authorization": f"Bearer {req.supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    create_res = requests.post(
        f"{req.supabase_url}/rest/v1/election_day_pdf_imports",
        json={"source_filename": req.source_filename, "status": "processing"},
        headers=headers,
        timeout=15,
    )
    if create_res.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail="Could not create import record.")
    import_id = create_res.json()[0]["id"]

    background_tasks.add_task(
        process_in_background,
        req.pdf_url, import_id, req.supabase_url, req.supabase_key,
        req.source_filename,
    )

    return {
        "success": True,
        "import_id": import_id,
        "message": "Extraction started - check election_day_pdf_imports for status.",
    }

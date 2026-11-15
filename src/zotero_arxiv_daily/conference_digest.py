"""Build a daily slate from published and accepted 2026 conference papers."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
import gzip
from hashlib import sha256
from http.client import HTTPException
import json
import re
from unicodedata import normalize
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zlib import error as CompressionError

from loguru import logger
from lxml import etree, html

from .protocol import Paper


DAILY_LIMIT = 20
CANDIDATE_LIMIT = 256
ROTATION_DAYS = 64
ANTHOLOGY_XML = "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml"
CVPR_PROCEEDINGS = "https://openaccess.thecvf.com/CVPR2026?day=all"
ICML_PROCEEDINGS = "https://proceedings.mlr.press/v306/"
USENIX_PROGRAM = "https://www.usenix.org/conference/usenixsecurity26/technical-sessions"
SP_ACCEPTED = "https://sp2026.ieee-security.org/accepted-papers.html"
CCS_ACCEPTED = "https://www.sigsac.org/ccs/CCS2026/program/accepted-papers.html"
NDSS_ACCEPTED = "https://www.ndss-symposium.org/ndss2026/accepted-papers/"
WWW_RESEARCH = "https://www2026.thewebconf.org/accepted/research-tracks.html"
WWW_PROCEEDINGS = "https://www.sigweb.hosting.acm.org/toc/www26a.html"
AAAI_FULL_PAPER_ISSUES = {str(issue) for issue in range(1, 46)}
INCOMPLETE_RESPONSE = (HTTPException, EOFError, CompressionError)
SOURCE_ERRORS = (OSError, ValueError, KeyError, ElementTree.ParseError, etree.LxmlError, *INCOMPLETE_RESPONSE)

DETAIL_ABSTRACT_XPATH = {
    "CVPR": "//*[@id='abstract']",
    "ICML": "//*[@id='abstract']",
    "ICLR": "//*[@id='abstractText']",
    "NeurIPS": "//*[@id='abstractText']",
    "NDSS": "//div[@class='paper-data']/p[not(strong) and normalize-space()]",
}
ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def _get(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "zotero-arxiv-daily (https://github.com/Herren-Hsing/zotero-arxiv-daily)",
            "Accept-Encoding": "gzip",
        },
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    return gzip.decompress(body)
                return body
        except INCOMPLETE_RESPONSE as error:
            if attempt:
                raise
            logger.warning(f"Incomplete catalogue response from {url}; retrying once: {error}")


def _title_key(title: str) -> str:
    title = normalize("NFKC", title).casefold()
    return "".join(char for char in title if char.isalnum())


def _xml_text(element: ElementTree.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def _anthology_papers(venue: str, volume: str) -> list[Paper]:
    catalogue = ElementTree.fromstring(_get(f"{ANTHOLOGY_XML}/2026.{venue.lower()}.xml"))
    papers = []
    for entry in catalogue.findall(f"./volume[@id='{volume}']/paper"):
        identifier = f"2026.{venue.lower()}-{volume}.{entry.attrib['id']}"
        papers.append(Paper(
            source=venue,
            title=_xml_text(entry.find("title")),
            authors=[
                " ".join(filter(None, [_xml_text(author.find("first")), _xml_text(author.find("last"))]))
                for author in entry.findall("author")
            ],
            abstract=_xml_text(entry.find("abstract")),
            url=f"https://aclanthology.org/{identifier}/",
        ))
    return papers


def _usenix_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(USENIX_PROGRAM))
    papers = []
    for entry in catalogue.xpath("//article[contains(concat(' ', @class, ' '), ' node-paper ')]"):
        links = entry.xpath("./h2/a[starts-with(@href, '/conference/usenixsecurity26/presentation/')]")
        if not links:
            continue
        link = links[0]
        authors = entry.xpath(".//div[contains(@class, 'field-name-field-paper-people-text')]//p/text()")
        abstract = entry.xpath("string(.//div[contains(@class, 'field-name-field-paper-description-long')])")
        papers.append(Paper(
            source="USENIX Security",
            title=" ".join(link.text_content().split()),
            authors=[name.strip(" ,;\r\n\t") for name in authors if name.strip(" ,;\r\n\t")],
            abstract=" ".join(abstract.split()),
            url=urljoin(USENIX_PROGRAM, link.attrib["href"]),
        ))
    return papers


def _text_link(url: str, title: str) -> str:
    # Some venues expose only a conference-wide list. Link to the paper's title within that list.
    return f"{url}#:~:text={quote(title, safe='').replace('-', '%2D')}"


def _sp_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(SP_ACCEPTED))
    papers = []
    for entry in catalogue.xpath("//div[starts-with(@id, 'cycle')]/div[@class='list-group-item'][b/a]"):
        title = " ".join(entry.xpath("string(./b/a)").split())
        author_text = "".join(entry.xpath("./div[contains(@class, 'authorlist')]/br[1]/preceding-sibling::text()"))
        papers.append(Paper(
            source="IEEE S&P",
            title=title,
            authors=[name.strip() for name in author_text.split(",") if name.strip()],
            abstract="",
            url=_text_link(SP_ACCEPTED, title),
        ))
    return papers


def _ccs_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(CCS_ACCEPTED))
    papers = []
    for entry in catalogue.xpath("//table//tr[count(td)=2]"):
        title = " ".join(entry.xpath("string(./td[1])").split())
        author_lines = entry.xpath("./td[2]/text()")
        papers.append(Paper(
            source="CCS",
            title=title,
            authors=[name.split(" (", 1)[0].strip() for name in author_lines if name.strip()],
            abstract="",
            url=_text_link(CCS_ACCEPTED, title),
        ))
    return papers


def _ndss_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(NDSS_ACCEPTED))
    papers = []
    for link in catalogue.xpath("//h2[@class='pt-cv-title']/a[starts-with(@href, 'https://www.ndss-symposium.org/ndss-paper/')]"):
        author_text = link.xpath("string(../../div//div[contains(@class, 'pt-cv-ctf-display_authors')])")
        author_lines = re.split(r"\)\s*,\s*", author_text)
        papers.append(Paper(
            source="NDSS",
            title=" ".join(link.text_content().split()),
            authors=[name.split(" (", 1)[0].strip() for name in author_lines if name.strip()],
            abstract="",
            url=link.attrib["href"],
        ))
    return papers


def _www_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(WWW_RESEARCH))
    papers = []
    for entry in catalogue.xpath("//div[@class='papers-section']//li[span[@class='paper-id']]"):
        title = " ".join(entry.xpath("string(span[@class='paper-id']/following-sibling::text()[1])").split())
        title = title.rstrip(" —")
        author_text = entry.xpath("string(span[@class='paper-authors'])")
        papers.append(Paper(
            source="WWW",
            title=title,
            authors=[name.strip() for name in re.split(r",| and ", " ".join(author_text.split())) if name.strip()],
            abstract="",
            url=_text_link(WWW_RESEARCH, title),
        ))
    return papers


def _cvpr_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(CVPR_PROCEEDINGS))
    papers = []
    for link in catalogue.xpath("//dt[@class='ptitle']/a[starts-with(@href, '/content/CVPR2026/html/')]"):
        papers.append(Paper(
            source="CVPR",
            title=link.text_content().strip(),
            authors=link.xpath("../following-sibling::dd[1]//input[@name='query_author']/@value"),
            abstract="",
            url=urljoin(CVPR_PROCEEDINGS, link.attrib["href"]),
        ))
    return papers


def _icml_papers() -> list[Paper]:
    catalogue = html.fromstring(_get(ICML_PROCEEDINGS))
    papers = []
    for entry in catalogue.xpath("//div[@class='paper'][p[@class='title']]"):
        links = entry.xpath(".//a[normalize-space()='abs']/@href")
        if links and links[0].startswith(ICML_PROCEEDINGS):
            authors = entry.xpath("string(.//span[@class='authors'])")
            papers.append(Paper(
                source="ICML",
                title=entry.xpath("string(p[@class='title'])").strip(),
                authors=[author.strip() for author in authors.split(",") if author.strip()],
                abstract="",
                url=links[0],
            ))
    return papers


def _virtual_papers(venue: str) -> list[Paper]:
    host = f"https://{venue.lower()}.cc"
    url = f"{host}/static/virtual/data/{venue.lower()}-2026-orals-posters.json"
    main_track = f"https://openreview.net/group?id={venue}.cc/2026/Conference"
    papers = []
    for entry in json.loads(_get(url))["results"]:
        path = entry.get("virtualsite_url") or ""
        decision = entry.get("decision") or ""
        if (
            entry.get("sourceurl") != main_track
            or not entry.get("visible")
            or not decision.casefold().startswith("accept (")
            or not path.startswith("/virtual/2026/poster/")
        ):
            continue
        papers.append(Paper(
            source=venue,
            title=entry["name"],
            authors=[author["fullname"] for author in entry.get("authors") or [] if author.get("fullname")],
            abstract="",
            url=urljoin(host, path),
        ))
    return papers


def _fetch_detail_abstract(paper: Paper) -> str:
    try:
        page = html.fromstring(_get(paper.url))
        sections = page.xpath(DETAIL_ABSTRACT_XPATH[paper.source])
        return " ".join(" ".join(section.text_content() for section in sections).split())
    except SOURCE_ERRORS as error:
        logger.warning(f"Unable to load the conference abstract from {paper.url}: {error}")
        return ""


def _fetch_www_abstracts() -> dict[str, str]:
    page = html.fromstring(_get(WWW_PROCEEDINGS))
    return {
        _title_key(link.text_content()): " ".join(
            link.xpath("string(../following-sibling::*[2][@class='DLabstract'])").split()
        )
        for link in page.xpath("//a[@class='DLtitleLink'][starts-with(@href, 'https://dl.acm.org/doi/10.1145/3774904.')]")
    }


def _author_surnames(names: list[str]) -> set[str]:
    return {_title_key(name.split()[-1]) for name in names if name.strip()}


def _fill_from_arxiv(papers: list[Paper]) -> None:
    phrases = [" ".join(re.findall(r"\w+", paper.title)) for paper in papers]
    query = urlencode({
        "search_query": " OR ".join(f'ti:"{title}"' for title in phrases),
        "max_results": 5 * len(phrases),
    })
    feed = ElementTree.fromstring(_get(f"https://export.arxiv.org/api/query?{query}"))
    by_title: dict[str, list[tuple[set[str], str]]] = {}
    for entry in feed.findall("atom:entry", ATOM):
        title = _title_key(entry.findtext("atom:title", default="", namespaces=ATOM))
        names = [author.text or "" for author in entry.findall("atom:author/atom:name", ATOM)]
        abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM).split())
        if title and abstract:
            by_title.setdefault(title, []).append((_author_surnames(names), abstract))

    for paper in papers:
        authors = _author_surnames(paper.authors)
        matches = {
            abstract
            for names, abstract in by_title.get(_title_key(paper.title), [])
            if not authors or authors & names
        }
        if len(matches) == 1:
            paper.abstract = matches.pop()


def load_conference_abstracts(papers: list[Paper]) -> None:
    """Fill missing abstracts for the selected papers using public metadata."""
    details = [paper for paper in papers if not paper.abstract and paper.source in DETAIL_ABSTRACT_XPATH]
    if details:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for paper, abstract in zip(details, pool.map(_fetch_detail_abstract, details)):
                paper.abstract = abstract

    www_papers = [paper for paper in papers if not paper.abstract and paper.source == "WWW"]
    if www_papers:
        try:
            abstracts = _fetch_www_abstracts()
            for paper in www_papers:
                paper.abstract = abstracts.get(_title_key(paper.title), "")
        except SOURCE_ERRORS as error:
            logger.warning(f"Unable to load the WWW proceedings abstracts: {error}")

    missing = [paper for paper in papers if not paper.abstract]
    if missing:
        try:
            _fill_from_arxiv(missing)
        except SOURCE_ERRORS as error:
            logger.warning(f"Unable to look up conference abstracts on arXiv: {error}")


def _aaai_papers() -> list[Paper]:
    papers = []
    page_size = 1000
    # The 2026 volume has fewer than 10,000 records, within Crossref's offset limit.
    for offset in range(0, 10000, page_size):
        query = urlencode({
            "filter": "issn:2374-3468,from-pub-date:2026-01-01,until-pub-date:2026-12-31",
            "select": "DOI,title,author,abstract,volume,issue",
            "rows": page_size,
            "offset": offset,
            "sort": "published",
            "order": "asc",
        })
        response = json.loads(_get(f"https://api.crossref.org/works?{query}"))["message"]
        for entry in response["items"]:
            # Issues 1–43 are technical tracks, 44–45 are full-paper special tracks.
            # From issue 46 onward AAAI mixes those papers with companion material.
            if entry.get("volume") != "40" or entry.get("issue") not in AAAI_FULL_PAPER_ISSUES:
                continue
            titles = entry.get("title")
            doi = entry.get("DOI", "")
            if not titles or not doi.startswith("10.1609/aaai.v40i"):
                continue
            abstract = entry.get("abstract")
            papers.append(Paper(
                source="AAAI",
                title=titles[0],
                authors=[
                    " ".join(filter(None, [author.get("given"), author.get("family")]))
                    for author in entry.get("author", [])
                ],
                abstract=html.fromstring(abstract).text_content().strip() if abstract else "",
                url=f"https://doi.org/{doi}",
            ))
        if len(response["items"]) < page_size:
            break
    return papers


def fetch_conference_papers() -> list[Paper]:
    """Fetch public conference metadata, retaining the sources that respond."""
    sources = {
        "ACL": lambda: _anthology_papers("ACL", "long"),
        "CVPR": _cvpr_papers,
        "ICML": _icml_papers,
        "ICLR": lambda: _virtual_papers("ICLR"),
        "NeurIPS": lambda: _virtual_papers("NeurIPS"),
        "AAAI": _aaai_papers,
        "EMNLP": lambda: _anthology_papers("EMNLP", "main"),
        "USENIX Security": _usenix_papers,
        "IEEE S&P": _sp_papers,
        "CCS": _ccs_papers,
        "NDSS": _ndss_papers,
        "WWW": _www_papers,
    }
    papers = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {venue: pool.submit(fetch) for venue, fetch in sources.items()}
        for venue, future in pending.items():
            try:
                catalogue = future.result()
            except SOURCE_ERRORS as error:
                logger.warning(f"Unable to load {venue} 2026 conference papers: {error}")
                continue
            logger.info(f"Loaded {len(catalogue)} {venue} 2026 conference papers")
            papers.extend(catalogue)
    return papers


def daily_conference_candidates(
    catalogue: list[Paper], excluded_titles: list[str], day: date,
) -> list[Paper]:
    """Assign each title to a stable date bucket independently of catalogue order."""
    excluded = {_title_key(title) for title in excluded_titles}
    candidates: dict[int, Paper] = {}
    bucket = day.toordinal() % ROTATION_DAYS
    for paper in catalogue:
        key = _title_key(paper.title)
        if not key or key in excluded:
            continue
        fingerprint = int.from_bytes(sha256(key.encode()).digest())
        if fingerprint % ROTATION_DAYS == bucket:
            previous = candidates.get(fingerprint)
            if previous is None or (not previous.abstract and paper.abstract):
                candidates[fingerprint] = paper
    fingerprints = sorted(candidates)
    if len(fingerprints) <= CANDIDATE_LIMIT:
        return [candidates[key] for key in fingerprints]

    # Keep smaller venues represented in the ranking pool before filling the remaining slots.
    by_venue: dict[str, list[int]] = {}
    for key in fingerprints:
        by_venue.setdefault(candidates[key].source, []).append(key)
    reserved_per_venue = CANDIDATE_LIMIT // (2 * len(by_venue))
    selected = {key for venue_keys in by_venue.values() for key in venue_keys[:reserved_per_venue]}
    remaining = [key for key in fingerprints if key not in selected]
    selected.update(remaining[:CANDIDATE_LIMIT - len(selected)])
    return [candidates[key] for key in sorted(selected)]

"""
Large-file downloader: drive JDownloader 2 over the MyJDownloader API, then hand the
finished files to the Telegram uploader in bot.py.

JDownloader handles the parts that plain HTTP can't:
  - FuckingFast (fuckingfast.co): whole site behind a Cloudflare JS challenge.
  - DataNodes (datanodes.to): XFileSharing, 10s wait + Cloudflare Turnstile captcha.
  - FileKeeper (filekeeper.net): XFileSharing, 5s wait, no captcha (trivial).

The bot only needs: detect_host/extract_urls to validate input, and a JDClient to add a
job, poll its progress, list finished files, and cancel.
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger("ai-bot.dl")

try:
    import myjdapi  # type: ignore
    MYJD_AVAILABLE = True
except Exception:  # pragma: no cover
    MYJD_AVAILABLE = False

# Path to finished files as the BOT sees it (shared volume), and as JDownloader sees it.
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/downloads").rstrip("/")
JD_OUTPUT_DIR = os.getenv("JD_OUTPUT_DIR", "/output").rstrip("/")

# Host domain -> friendly label. These are the three FitGirl mirror hosts we support.
SUPPORTED_HOSTS = {
    "fuckingfast.co": "FuckingFast",
    "datanodes.to": "DataNodes",
    "filekeeper.net": "FileKeeper",
}

_URL_RE = re.compile(r"https?://[^\s<>\"'`]+")
# JDownloader marks in-progress files with these suffixes; skip them when listing output.
_PARTIAL_SUFFIXES = (".part", ".tmp", ".jdxfer", ".download")


def extract_urls(text: str) -> list[str]:
    """Pull all http(s) URLs out of a chunk of text, de-duplicated, order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for u in _URL_RE.findall(text or ""):
        u = u.rstrip(".,);]")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def detect_host(url: str) -> str | None:
    """Return the friendly host label for a URL, or None if it's not a supported host."""
    return SUPPORTED_HOSTS.get(_domain(url))


@dataclass
class Job:
    urls: list[str]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    cancelled: bool = False

    @property
    def package_name(self) -> str:
        return f"tgdl-{self.job_id}"

    @property
    def dest_jd(self) -> str:
        """Destination folder as JDownloader writes it."""
        return f"{JD_OUTPUT_DIR}/{self.job_id}"

    @property
    def read_dir(self) -> Path:
        """Same folder as the bot reads it off the shared volume."""
        return Path(DOWNLOAD_DIR) / self.job_id

    def hosts(self) -> list[str]:
        return [detect_host(u) or _domain(u) for u in self.urls]


@dataclass
class Snapshot:
    present: bool = False          # package has shown up in the download list yet
    finished: bool = False
    bytes_total: int = 0
    bytes_loaded: int = 0
    speed: int = 0                 # bytes/sec
    eta: int = -1                  # seconds, -1 if unknown
    status: str = ""               # e.g. "", "Downloading", "Captcha", error text
    error: str = ""                # non-empty if JD reported a hard failure

    @property
    def percent(self) -> float:
        return (self.bytes_loaded / self.bytes_total * 100.0) if self.bytes_total else 0.0


class JDClient:
    """Thin, resilient wrapper over myjdapi. All methods are blocking (requests-based);
    call them from a worker thread (asyncio.to_thread) in the async bot."""

    def __init__(self, email: str, password: str, device_name: str,
                 app_key: str = "oci-arm-host-capacity") -> None:
        self.email = email
        self.password = password
        self.device_name = device_name
        self.app_key = app_key
        self._jd = None
        self._device = None

    # -- connection -------------------------------------------------------
    def connect(self):
        if not MYJD_AVAILABLE:
            raise RuntimeError("myjdapi is not installed")
        jd = myjdapi.Myjdapi()
        jd.set_app_key(self.app_key)
        jd.connect(self.email, self.password)
        jd.update_devices()
        try:
            device = jd.get_device(self.device_name)
        except Exception:
            device = None
        if device is None:
            names = [d.get("name") for d in jd.list_devices()]
            raise RuntimeError(
                f"MyJDownloader device {self.device_name!r} not found. Available: {names or 'none'}"
            )
        self._jd = jd
        self._device = device
        return device

    @property
    def device(self):
        if self._device is None:
            self.connect()
        return self._device

    def _reconnect(self):
        self._jd = self._device = None
        return self.connect()

    def _call(self, fn):
        """Run a device call, reconnecting once if the session expired."""
        try:
            return fn(self.device)
        except Exception as e:  # myjdapi raises on expired tokens / transient relay errors
            log.warning("MyJDownloader call failed (%s); reconnecting", e)
            self._reconnect()
            return fn(self.device)

    # -- jobs -------------------------------------------------------------
    def add_job(self, urls: list[str]) -> Job:
        job = Job(urls=list(urls))
        self._call(lambda d: d.linkgrabber.add_links([{
            "autostart": True,
            "links": "\n".join(job.urls),
            "packageName": job.package_name,
            "destinationFolder": job.dest_jd,
            "overwritePackagizerRules": True,
        }]))
        log.info("Queued job %s (%d link(s)) -> %s", job.job_id, len(job.urls), job.dest_jd)
        return job

    def poll(self, job: Job) -> Snapshot:
        snap = Snapshot()
        # Prefer the downloads list (active transfers). Fall back to the linkgrabber so we
        # can still surface online-check/captcha/offline states before the download starts.
        pkgs = self._call(lambda d: d.downloads.query_packages([{
            "bytesLoaded": True, "bytesTotal": True, "speed": True, "eta": True,
            "finished": True, "running": True, "status": True, "enabled": True,
            "comment": True, "saveTo": True,
        }])) or []
        pkg = next((p for p in pkgs if p.get("name") == job.package_name), None)
        source = "downloads"

        if pkg is None:
            lg = self._call(lambda d: d.linkgrabber.query_packages([{
                "bytesTotal": True, "status": True, "availableOnlineCount": True,
                "availableOfflineCount": True, "comment": True, "saveTo": True,
            }])) or []
            pkg = next((p for p in lg if p.get("name") == job.package_name), None)
            source = "linkgrabber"

        if pkg is None:
            return snap  # not present yet (still being parsed)

        snap.present = True
        snap.bytes_total = int(pkg.get("bytesTotal") or 0)
        snap.bytes_loaded = int(pkg.get("bytesLoaded") or 0)
        snap.speed = int(pkg.get("speed") or 0)
        snap.eta = int(pkg.get("eta", -1))
        status = (pkg.get("status") or "").strip()
        snap.status = status

        low = status.lower()
        if any(k in low for k in ("error", "offline", "not found", "failed", "temporarily unavailable")):
            snap.error = status or "download failed"
        if "captcha" in low:
            snap.status = "Waiting for captcha (solve it in the JDownloader UI)"

        if source == "downloads":
            snap.finished = bool(pkg.get("finished")) or (
                snap.bytes_total > 0 and snap.bytes_loaded >= snap.bytes_total
            )
        return snap

    def finished_files(self, job: Job) -> list[Path]:
        """Completed files for a job, read off the shared volume."""
        if not job.read_dir.exists():
            return []
        files = [
            p for p in sorted(job.read_dir.rglob("*"))
            if p.is_file() and not p.name.lower().endswith(_PARTIAL_SUFFIXES)
        ]
        return files

    def cancel(self, job: Job) -> None:
        job.cancelled = True
        # Remove from both the download list and the linkgrabber.
        try:
            pkgs = self._call(lambda d: d.downloads.query_packages([{"saveTo": True}])) or []
            uuids = [p["uuid"] for p in pkgs if p.get("name") == job.package_name and "uuid" in p]
            if uuids:
                self._call(lambda d: d.downloads.remove_links([], uuids))
        except Exception as e:
            log.warning("cancel (downloads) failed for %s: %s", job.job_id, e)
        try:
            lg = self._call(lambda d: d.linkgrabber.query_packages([{}])) or []
            uuids = [p["uuid"] for p in lg if p.get("name") == job.package_name and "uuid" in p]
            if uuids:
                self._call(lambda d: d.linkgrabber.remove_links([], uuids))
        except Exception as e:
            log.warning("cancel (linkgrabber) failed for %s: %s", job.job_id, e)

    # -- crawl (discover links on a page, grouped by host) ----------------
    def add_crawl(self, urls: list[str]) -> str:
        """Add page/link URLs to the linkgrabber WITHOUT downloading, so JDownloader
        crawls them and resolves the real hoster links. Returns the crawl package name."""
        crawl_name = f"tgcrawl-{uuid.uuid4().hex[:12]}"
        self._call(lambda d: d.linkgrabber.add_links([{
            "autostart": False,
            "links": "\n".join(urls),
            "packageName": crawl_name,
            "overwritePackagizerRules": True,
        }]))
        log.info("Crawling %d URL(s) under %s", len(urls), crawl_name)
        return crawl_name

    def crawl_links(self, crawl_name: str) -> list[dict]:
        """Links found so far for a crawl package: each has url/host/name/bytesTotal/availability."""
        pkgs = self._call(lambda d: d.linkgrabber.query_packages([{"saveTo": True}])) or []
        uuids = [p["uuid"] for p in pkgs if p.get("name") == crawl_name and "uuid" in p]
        if not uuids:
            return []
        links = self._call(lambda d: d.linkgrabber.query_links([{
            "url": True, "host": True, "name": True, "bytesTotal": True,
            "availability": True, "status": True, "packageUUIDs": uuids,
        }])) or []
        return links

    def remove_crawl(self, crawl_name: str) -> None:
        try:
            pkgs = self._call(lambda d: d.linkgrabber.query_packages([{}])) or []
            uuids = [p["uuid"] for p in pkgs if p.get("name") == crawl_name and "uuid" in p]
            if uuids:
                self._call(lambda d: d.linkgrabber.remove_links([], uuids))
        except Exception as e:
            log.warning("remove_crawl failed for %s: %s", crawl_name, e)

import os
import re
import sys
import hashlib
import os.path
import pathlib
import urllib
import urllib.error
import urllib.request
from tqdm import tqdm
from urllib.parse import urlparse
from typing import Any, Optional, Union
from .extract import extract_archive

USER_AGENT = "pytorch/spaic"

def calculate_md5(fpath: Union[str, pathlib.Path], chunk_size: int = 1024 * 1024) -> str:
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def check_md5(fpath: Union[str, pathlib.Path], md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath: Union[str, pathlib.Path], md5: Optional[str] = None) -> bool:
    if not os.path.exists(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)



def _urlretrieve(url: str, filename: Union[str, pathlib.Path], chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        with open(filename, "wb") as fh, tqdm(total=response.length) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))

def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )

def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def download_file_from_google_drive(
    file_id: str,
    root: Union[str, pathlib.Path],
    filename: Optional[Union[str, pathlib.Path]] = None,
    md5: Optional[str] = None,
):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    try:
        import gdown
    except ModuleNotFoundError:
        raise RuntimeError(
            "To download files from GDrive, 'gdown' is required. You can install it with 'pip install gdown'."
        )

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f"Using downloaded {'and verified ' if md5 else ''}file: {fpath}")
        return

    gdown.download(id=file_id, output=fpath, quiet=False, user_agent=USER_AGENT)

    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def download_url(
    url: str,
    root: Union[str, pathlib.Path],
    filename: Optional[Union[str, pathlib.Path]] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead. Downloading " + url + " to " + fpath)
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    print(fpath, md5)
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")

def download_and_extract_archive(
    url: str,
    download_root: Union[str, pathlib.Path],
    extract_root: Optional[Union[str, pathlib.Path]] = None,
    filename: Optional[Union[str, pathlib.Path]] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    extract_archive(archive, extract_root, remove_finished)
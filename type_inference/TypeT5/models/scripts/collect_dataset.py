import json
import logging
import os
import shutil
import subprocess
import time
import pickle
import random

import libcst as cst
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typet5.data import GitRepo, get_dataset_dir
from typet5.type_env import collect_annots_info, mypy_checker
from typet5.utils import proj_root, read_file, write_file, not_none

os.chdir(proj_root())

"""
1. download all candidate repos
"""
# 仓库列表
all_repos = json.loads(read_file("data/mypy-dependents-by-stars.json"))
all_repos = [GitRepo.from_json(r) for r in all_repos]
# all_repos=all_repos[:10] # for testing

# 获取数据集文件目录：先从配置文件中读取数据根目录进行拼接；否则直接构造路径返回
repos_dir = get_dataset_dir("ManyTypes4Py") / "repos"


def clear_downloaded_repos(repos_dir):
    shutil.rmtree(repos_dir)


def download_repos(
        to_download: list[GitRepo], repos_dir, download_timeout=10.0, max_workers=10
) -> list[GitRepo]:
    """ 下载单个 Repo """

    def download_single(repo: GitRepo):
        try:
            if repo.download(repos_dir, timeout=download_timeout):
                repo.read_last_update(repos_dir)
                return repo
            else:
                return None
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logging.warning(f"Failed to download {repo.name}. Exception: {e}")
            return None

    print("Downloading repos from Github...")
    t_start = time.time()
    """ 多线程下载 Repo """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fs = [executor.submit(download_single, repo) for repo in to_download]
        rs = [f.result() for f in tqdm(as_completed(fs), total=len(fs))]
    print(f"Downloading took {time.time() - t_start} seconds.")
    downloaded = [r for r in rs if r is not None]
    return downloaded


if not repos_dir.exists():
    (repos_dir / "downloading").mkdir(parents=True)
    (repos_dir / "downloaded").mkdir(parents=True)
    downloaded_repos = download_repos(all_repos, repos_dir)
    print("Deleting failed repos...")
    shutil.rmtree(repos_dir / "downloading")
else:
    print("Repos already downloaded.")
    downloaded_dirs = set(d.name for d in (repos_dir / "downloaded").iterdir())
    downloaded_repos = [r for r in all_repos if r.authorname() in downloaded_dirs]
    print("Reading last updates...")
    for r in tqdm(downloaded_repos):
        r.read_last_update(repos_dir)
# 【output】Downloaded 2415/5996 repos.
print(f"Downloaded {len(downloaded_repos)}/{len(all_repos)} repos.")

"""
2. filter out repos that are too old or too big
"""
from datetime import datetime, timezone

date_threshold = datetime(2021, 4, 20)
# 过滤掉在 data_threshold 之后没有更新的代码；这里下载的很多 repo 没有 r.last_update 属性，会产生 assert
# new_repos = [r for r in downloaded_repos if not_none(r.last_update) > date_threshold]
# 修改让没有 last_update 属性的 repo 也保留
new_repos = [r for r in downloaded_repos if r.last_update == None or not_none(r.last_update) > date_threshold]
# 【output】 728 / 2415 repos are updated within a year.
print(f"{len(new_repos)} / {len(downloaded_repos)} repos are updated within a year.")
loc_limit = 50000

small_repos = []
for rep in tqdm(new_repos):
    try:
        loc = rep.count_lines_of_code(repos_dir)
        if loc < loc_limit:
            small_repos.append(rep)
    except UnicodeDecodeError:
        # nothing we can do
        pass
    except Exception as e:
        logging.warning(f"Failed to count lines of code for {rep.name}. Exception: {e}")

# 【output】715/728 repos are within the size limit (50000 LOC).
print(
    f"{len(small_repos)}/{len(new_repos)} repos are within the size limit ({loc_limit} LOC)."
)

"""
3. filter away repos with too few annotations
"""


def count_repo_annots(rep):
    try:
        # rep.count_annotations(repos_dir)
        rep.collect_annotations(repos_dir)
        if rep.n_type_annots / rep.lines_of_code > 0.05:
            return rep
    except Exception as e:
        logging.warning(f"Failed to count annotations for {rep.name}. Exception: {e}")
        return None


with ProcessPoolExecutor(max_workers=30) as executor:
    fs = [executor.submit(count_repo_annots, rep) for rep in small_repos]
    rs = [f.result() for f in tqdm(as_completed(fs), total=len(fs))]
useful_repos: list[GitRepo] = [
    r for r in rs if r is not None and "typeshed" not in r.name
]

# 414/715 repos are parsable and have enough portions of type annotations.
print(
    f"{len(useful_repos)}/{len(small_repos)} repos are parsable and have enough portions of type annotations."
)

"""
4. Some summary statistics
"""

# print total number of manual annotations
n_total_annots = sum(not_none(rep.n_type_annots) for rep in useful_repos)
print("Total number of manual annotations:", n_total_annots)

# print total number of type places
n_total_places = sum(not_none(rep.n_type_places) for rep in useful_repos)
print("Total number of type places:", n_total_places)

# print total number of lines of code
n_total_lines = sum(not_none(rep.lines_of_code) for rep in useful_repos)
print("Total number of lines of code:", n_total_lines)

# print average number of type annotations per line of code excluding projects with more than 1000 lines of code
n_avg_annots = (
        sum(not_none(rep.n_type_annots) for rep in useful_repos if rep.lines_of_code < 1000)
        / n_total_lines
)

# 414/715 repos are parsable and have enough portions of type annotations.
# Total number of manual annotations: 169620
# Total number of type places: 265453
# Total number of lines of code: 1656532


"""
5. 保存有用的数据集
"""
import pickle

useful_repos_path = proj_root() / "scripts" / "useful_repos.pkl"
with useful_repos_path.open("wb") as f:
    pickle.dump(useful_repos, f)
print(f"Saved {len(useful_repos)} useful repos to {useful_repos_path}.")
with useful_repos_path.open("rb") as f:
    print(pickle.load(f)[:3])

# Saved 414 useful repos to /home/csc/Code/TypeInference/TypeT5/scripts/useful_repos.pkl.

# """
# 6. 根据论文中的划分，划分数据集
# """
from typet5.utils import pickle_load, Path, proj_root, tqdm
import shutil
from typet5.data import GitRepo, get_dataset_dir

os.chdir(proj_root())

# 论文中的数据集划分方式
repos_split = pickle_load(Path("data/repos_split.pkl"))
repos_dir = get_dataset_dir("ManyTypes4Py") / "repos"
# 实际下载后数据集路径如下（原代码中没有 ”download“ 路径，所以会所有文件都找不到）
src_repos_dir = get_dataset_dir("ManyTypes4Py") / "repos" / "downloaded"
exist = 0
all = 0

for split, repos in repos_split.items():
    for r in tqdm(repos, desc=f"Moving {split} repos."):
        r: GitRepo
        split: str
        src = src_repos_dir / r.authorname()
        (repos_dir / split).mkdir(parents=True, exist_ok=True)
        dest = repos_dir / split / r.authorname()
        all = all + 1
        if src.exists():
            shutil.move(src, dest)
            exist = exist + 1
        else:
            print(f"Repo {r.name} not found.")

# 361/663 exits, 302/663 not found!
# 划分得到的具体数量为，train: 312, valid: 21, test: 28
print("{}/{} exits, {}/{} not found!".format(exist, all, all - exist, all))

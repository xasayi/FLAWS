import os
import re
import arxiv
import tarfile
import openreview


def extract_tar(tar: str, des: str) -> None:
    """
    Extract the tar and save at destination.
    """
    tar_file = tarfile.open(tar, "r:gz")
    tar_file.extractall(des)
    tar_file.close()


def download_arxiv_source(
    arxiv_id: str, download_dir: str = "./", filename: str | None = None
) -> None:
    """
    Search for arciv source and download it.
    """
    try:
        # Search for the paper by its arXiv ID
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
        extract_dir = os.path.join(download_dir, arxiv_id)
        tar_path = os.path.join(download_dir, f"{arxiv_id}.tar.gz")

        # Download the source file
        if filename:
            paper.download_source(dirpath=download_dir, filename=f"{filename}.tar.gz")
        else:
            paper.download_source(dirpath=download_dir)

        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        os.remove(tar_path)
        print(f"Source file for arXiv ID {arxiv_id} downloaded to {download_dir}")

    except StopIteration:
        print(f"Error: No paper found with arXiv ID {arxiv_id}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Create output directory
    output_dir = "data/icml"
    os.makedirs(output_dir, exist_ok=True)

    # Enter your password and username for openreview
    password = ""
    username = ""

    # change this to whichever venue you can to get papers from
    venue_id = "ICML.cc/2025/Conference"

    openreview_client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )
    all_submissions = openreview_client.get_all_notes(content={"venueid": venue_id})
    for sub in all_submissions:
        title = re.search(r"title={.*}", sub.content["_bibtex"]["value"]).group(0)[7:-1]

        search = arxiv.Search(
            query=f'all:"{title}"',
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in search.results():
            if title == result.title:
                arxiv_id = result.get_short_id()

                if int(arxiv_id[:4]) >= 2501:
                    print(arxiv_id)
                    download_arxiv_source(arxiv_id, output_dir, arxiv_id)
                    break

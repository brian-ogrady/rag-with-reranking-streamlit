import os
import time
from typing import Optional

import arxiv

from tqdm import tqdm


def download_arxiv_papers(
    search_query: str,
    output_dir: str,
    max_results: Optional[int] = 100,
    total_pdf_size: Optional[int] = 500,
) -> None:
    """
    Download arXiv papers based on a search query.
    
    Args:
        search_query: Query string for arXiv search
        max_results: Maximum number of papers to download
        output_dir: Directory to save the papers
        total_pdf_size: Max MB to download (ie download 100MB of PDFs)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    client = arxiv.Client()
    
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    
    total_size = 0
    papers_downloaded = 0
    
    for result in tqdm(client.results(search), total=max_results, desc="Downloading papers"):
        pdf_filename = f"{result.get_short_id().replace('/', '_')}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        if os.path.exists(pdf_path):
            print(f"Skipping {pdf_filename} (already exists)")
            continue
        
        try:
            result.download_pdf(filename=pdf_path)
            
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)
            total_size += file_size
            papers_downloaded += 1
            
            print(f"Downloaded {pdf_filename} ({file_size:.2f} MB)")
            print(f"Total size so far: {total_size:.2f} MB")
            
            if total_size >= total_pdf_size:
                print(f"Reached target size of 500 MB. Downloaded {papers_downloaded} papers.")
                break

            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading {pdf_filename}: {e}")
    
    print(f"Downloaded {papers_downloaded} papers, total size: {total_size:.2f} MB")

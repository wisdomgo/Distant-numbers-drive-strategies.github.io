import planetary_computer
from pystac_client import Client
import requests
from datetime import datetime
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
from PIL import Image
import io
import numpy as np
import argparse

def inspect_blank(image_bytes, threshold=0.1):
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)
    
    if img.mode == 'RGBA':
        blank_pixels = np.sum(img_array[:, :, 3] == 0)
    else:
        blank_pixels = np.sum(np.all(img_array == [255, 255, 255], axis=-1))
    
    total_pixels = img_array.shape[0] * img_array.shape[1]
    blank_ratio = blank_pixels / total_pixels
    
    return blank_ratio

def search_imgs(collections=["sentinel-2-l2a"],
                datetime=None,
                cloud_cover=0.1,
                granule_id=None,
                content="visual"):
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=planetary_computer.sign_inplace)

    search = catalog.search(
        collections=collections,
        datetime=datetime,
        query={"eo:cloud_cover": {"lt": cloud_cover},
               "s2:mgrs_tile": {"eq": granule_id}}
    )

    items = list(search.items())
    print(f"Found {len(items)} items")
    # getting TIFF images with the least cloud cover and blank ratio
    best_item = None
    min_cloud_cover = float('inf')
    min_blank_ratio = float('inf')
    
    for item in items:
        try:
            # image_url = item.assets[content].href
            rendered_url = item.assets["rendered_preview"].href
            # response = session.get(image_url, timeout=30)
            # response.raise_for_status()
            response_rendered = session.get(rendered_url, timeout=30)
            response_rendered.raise_for_status()
            
            blank_ratio = inspect_blank(response_rendered.content)
            item_cloud_cover = item.properties["eo:cloud_cover"]
            print(f"Item {item.id} has cloud cover {item_cloud_cover} and blank ratio {blank_ratio}")

            if blank_ratio <= min_blank_ratio:
                best_item = item
                min_cloud_cover = item_cloud_cover
                min_blank_ratio = blank_ratio
                print(f"Found a better image: {item.id}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image for {item.id}: {e}")

    return best_item, session

def download_best_image(best_item, session, image_dir, content="visual"):
    if best_item is None:
        print("No suitable image found to download.")
        return False
    
    _id = best_item.id
    download_path = os.path.join(image_dir, f"{_id}_{content}.tif")
    print(f"Start downloading the best image: {download_path}")
    try:
        image_url = best_item.assets[content].href
        response = session.get(image_url, timeout=30)
        response.raise_for_status()
        
        with open(download_path, "wb") as f:
            f.write(response.content)
        print(f"Best image saved!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error occurred when saving {_id}.tif: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-2 images")
    parser.add_argument("--start_date", type=str, default="2024-01-01", help="Start date of the search")
    parser.add_argument("--end_date", type=str, default="2024-12-19", help="End date of the search")
    parser.add_argument("--cloud_cover", type=float, default=5, help="Cloud cover threshold")
    parser.add_argument("--threshold", type=float, default=0.5, help="Blank area threshold")
    parser.add_argument("--granule_id", type=str, default="48RUS", help="Granule ID")
    parser.add_argument("--content", type=str, default="visual", help="Content to download")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    _datetime = [start_date.isoformat(), end_date.isoformat()]
    cloud_cover = args.cloud_cover
    granule_id = args.granule_id
    content = args.content

    base_dir = "raw_tif"
    image_dir = os.path.join(base_dir, f"{granule_id}")
    os.makedirs(image_dir, exist_ok=True)

    best_item, session = search_imgs(datetime=_datetime, cloud_cover=cloud_cover, granule_id=granule_id, content=content)
    
    if best_item:
        download_best_image(best_item, session, image_dir, content=content)
    else:
        print("No images found with the specified conditions.")

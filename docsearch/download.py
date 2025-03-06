import os
import requests

from bs4 import BeautifulSoup

def download_minutes(url, download_dir):
    # Make a GET request to fetch the raw PDF content
    file=url.split('/')[-1]
    file=os.path.join(download_dir,file)
    response = requests.get(url)
    response.raise_for_status()  # raises exception when not a 2xx response

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content with the appropriate PDF file name
        with open(file, 'wb') as f:
            f.write(response.content)
        print("The PDF file has been downloaded")
    else:
        print("Failed to retrieve the PDF")

def parse_and_download_wvu_minutes(download_dir, url='https://bog.wvu.edu/minutes'):
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Fetch the web page content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links in the page
    links = soup.find_all('a')

    for link in links:
        href = link.get('href')
        has_text= len(link.text.strip()) >0
        is_pdf= href.endswith('.pdf')
        if href and is_pdf and has_text:

            download_minutes(href, download_dir)
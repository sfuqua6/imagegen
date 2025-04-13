import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_current_chapter(driver, base_url, output_folder):
    """
    Scrape the current chapter's PNG images.
    Finds all <div> elements with an id starting with "images-",
    extracts the page number info from <span class="page-num">,
    and downloads the PNG image from the contained <img>.
    """
    page_html = driver.page_source
    soup = BeautifulSoup(page_html, "html.parser")

    images_divs = soup.find_all("div", id=re.compile(r"^images-\d+"))
    if not images_divs:
        print("No chapter image containers found on this page.")
        return False  

    os.makedirs(output_folder, exist_ok=True)

    page_data = []
    for div in images_divs:
        page_num_span = div.find("span", class_="page-num")
        if not page_num_span:
            continue
        page_text = page_num_span.get_text(strip=True) 
        match = re.search(r"(\d+)\s*/\s*(\d+)", page_text)
        if not match:
            continue
        current_page = int(match.group(1))
        total_pages = int(match.group(2))
        img_tag = div.find("img")
        if not img_tag:
            continue
        img_url = img_tag.get("data-src") or img_tag.get("src")
        if not img_url:
            continue
        img_url = urljoin(base_url, img_url).split("?")[0]
        if not img_url.lower().endswith(".png"):
            continue
        page_data.append((current_page, total_pages, img_url))

    if not page_data:
        print("No valid PNG images found in this chapter.")
        return False

    page_data.sort(key=lambda x: x[0])
    print(f"Detected {len(page_data)} pages (max page: {max(d[1] for d in page_data)}) in this chapter.")

    for (cur_page, total_pages, img_url) in page_data:
        filename = f"page_{cur_page}.png"
        filepath = os.path.join(output_folder, filename)
        try:
            resp = requests.get(img_url, headers={"Referer": base_url}, timeout=15)
            if resp.ok:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                print(f"Saved page {cur_page} of {total_pages} to {filepath}")
            else:
                print(f"Failed to download {img_url} (Status: {resp.status_code})")
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")

    return True

def scrape_manga_chapters(start_url, chapter_goal=100, base_output_folder="manga_chapters"):
    options = uc.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36"
    )
    driver = uc.Chrome(options=options)
    actions = ActionChains(driver)

    try:
        driver.get(start_url)
        print(f"Initial page loaded: {start_url}")

        print("\n" + "="*50)
        print("MANUAL CAPTCHA STEP REQUIRED".center(50))
        print("Solve any captcha in the browser window, then press Enter here...")
        print("="*50 + "\n")
        input("Press Enter after solving captcha to continue scraping...")

        current_chapter = 1
        base_url = start_url  
        while current_chapter <= chapter_goal:
            print(f"\n=== Scraping Chapter {current_chapter} ===")

            last_height = driver.execute_script("return document.body.scrollHeight")
            current_scroll = 0
            while current_scroll < last_height:
                driver.execute_script("window.scrollBy(0, 300);")
                current_scroll += 300
                time.sleep(0.3) 
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)  

            chapter_folder = os.path.join(base_output_folder, f"chapter_{current_chapter}")
            success = scrape_current_chapter(driver, base_url, chapter_folder)
            if not success:
                print(f"Skipping chapter {current_chapter} due to missing images.")
            
            if current_chapter == chapter_goal:
                print("Chapter goal reached.")
                break

            print("Navigating to next chapter...")
            body = driver.find_element(By.TAG_NAME, "body")
            body.click()  
            time.sleep(0.5)
            actions.send_keys("n").perform()
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[id^='images-'] img"))
                )
            except Exception as e:
                print("Next chapter did not load properly:", e)
                break

            current_chapter += 1

        print("Finished scraping chapters.")

    finally:
        driver.quit()
        print("Browser closed. Scraping completed.")

if __name__ == "__main__":
    start_url = "https://allmanga.to/read/SFrub9DDGMrmdZWyh/chapter-1-sub"
    scrape_manga_chapters(start_url, chapter_goal=100, base_output_folder="manga_chapters")

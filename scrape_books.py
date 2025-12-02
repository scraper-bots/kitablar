import asyncio
import aiohttp
import csv
import logging
from typing import List, Dict
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class BookScraper:
    def __init__(self, total_pages: int = 295, per_page: int = 24):
        self.base_url = "https://mp-catalog.umico.az/api/v1/products"
        self.total_pages = total_pages
        self.per_page = per_page
        self.category_id = 1438
        self.sort = "global_popular_score"
        self.headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'az',
            'content-language': 'az',
            'origin': 'https://birmarket.az',
            'referer': 'https://birmarket.az/',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
        }
        self.all_products = []
        self.failed_pages = []
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async def get_total_count(self) -> int:
        """Get total number of products from API"""
        url = f"{self.base_url}?page=1&category_id={self.category_id}&per_page={self.per_page}&sort={self.sort}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        total = data.get("meta", {}).get("total", 0)
                        logger.info(f"Total products from API: {total}")
                        return total
            except Exception as e:
                logger.error(f"Error getting total count: {str(e)}")
                return 0

    async def fetch_page(self, session: aiohttp.ClientSession, page: int, retry: int = 3) -> Dict:
        """Fetch a single page with retry logic"""
        url = f"{self.base_url}?page={page}&category_id={self.category_id}&per_page={self.per_page}&sort={self.sort}"

        for attempt in range(retry):
            try:
                async with self.semaphore:
                    async with session.get(url, headers=self.headers, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully fetched page {page}/{self.total_pages}")
                            return {"page": page, "data": data}
                        else:
                            logger.warning(f"Page {page} returned status {response.status}, attempt {attempt + 1}/{retry}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on page {page}, attempt {attempt + 1}/{retry}")
            except Exception as e:
                logger.error(f"Error fetching page {page}, attempt {attempt + 1}/{retry}: {str(e)}")

            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to fetch page {page} after {retry} attempts")
        self.failed_pages.append(page)
        return {"page": page, "data": None}

    async def fetch_all_pages(self):
        """Fetch all pages concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_page(session, page) for page in range(1, self.total_pages + 1)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {str(result)}")
                elif result["data"] is not None:
                    products = result["data"].get("products", [])
                    self.all_products.extend(products)

    def extract_product_data(self, product: Dict) -> Dict:
        """Extract relevant fields from product"""
        default_offer = product.get("default_offer", {})
        seller = default_offer.get("seller", {})
        marketing_name = seller.get("marketing_name", {})
        category = product.get("category", {})
        ratings = product.get("ratings", {})
        main_img = product.get("main_img", {})

        # Get product labels if any
        product_labels = product.get("product_labels", [])
        product_labels_str = ", ".join([label.get("name", "") for label in product_labels if isinstance(label, dict)])

        # Get offer labels if any
        offer_labels = default_offer.get("product_offer_labels", [])
        offer_labels_str = ", ".join([label.get("name", "") for label in offer_labels if isinstance(label, dict)])

        return {
            "id": product.get("id"),
            "name": product.get("name"),
            "slugged_name": product.get("slugged_name"),
            "status": product.get("status"),
            "brand": product.get("brand"),
            "category_id": product.get("category_id"),
            "category_name": category.get("name"),
            "retail_price": default_offer.get("retail_price"),
            "old_price": default_offer.get("old_price"),
            "discount_start_date": default_offer.get("discount_effective_start_date"),
            "discount_end_date": default_offer.get("discount_effective_end_date"),
            "installment_enabled": default_offer.get("installment_enabled"),
            "max_installment_months": default_offer.get("max_installment_months"),
            "seller_ext_id": seller.get("ext_id"),
            "seller_name": marketing_name.get("name"),
            "seller_rating": seller.get("rating"),
            "seller_vat_payer": seller.get("vat_payer"),
            "seller_role": seller.get("role_name"),
            "rating_value": ratings.get("rating_value"),
            "rating_count": ratings.get("session_count"),
            "assessment_id": ratings.get("assessment_id"),
            "image_big": main_img.get("big"),
            "image_medium": main_img.get("medium"),
            "image_small": main_img.get("small"),
            "avail_check": product.get("avail_check"),
            "preorder_available": product.get("preorder_available"),
            "min_qty": product.get("min_qty"),
            "qty": default_offer.get("qty", 0),
            "show_stock_qty_threshold": default_offer.get("show_stock_qty_threshold"),
            "offer_uuid": default_offer.get("uuid"),
            "product_labels": product_labels_str,
            "offer_labels": offer_labels_str
        }

    def save_to_csv(self, filename: str = None):
        """Save all products to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"books_data_{timestamp}.csv"

        if not self.all_products:
            logger.error("No products to save!")
            return

        # Extract data from all products
        extracted_data = [self.extract_product_data(product) for product in self.all_products]

        # Get all unique keys
        fieldnames = list(extracted_data[0].keys())

        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(extracted_data)

            logger.info(f"Successfully saved {len(extracted_data)} products to {filename}")

            # Save failed pages info if any
            if self.failed_pages:
                failed_filename = f"failed_pages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(failed_filename, 'w') as f:
                    f.write(f"Failed pages: {', '.join(map(str, self.failed_pages))}\n")
                    f.write(f"Total failed: {len(self.failed_pages)}\n")
                logger.warning(f"Failed pages saved to {failed_filename}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            # Backup save as JSON in case of CSV error
            backup_filename = f"books_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_filename, 'w', encoding='utf-8') as f:
                json.dump(self.all_products, f, ensure_ascii=False, indent=2)
            logger.info(f"Backup saved to {backup_filename}")

    async def run(self):
        """Main execution method"""
        logger.info(f"Starting scraping of {self.total_pages} pages...")
        start_time = datetime.now()

        await self.fetch_all_pages()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Scraping completed in {duration:.2f} seconds")
        logger.info(f"Total products collected: {len(self.all_products)}")
        logger.info(f"Failed pages: {len(self.failed_pages)}")

        self.save_to_csv()

        # Summary
        print("\n" + "="*50)
        print("SCRAPING SUMMARY")
        print("="*50)
        print(f"Total pages: {self.total_pages}")
        print(f"Successfully scraped: {self.total_pages - len(self.failed_pages)}")
        print(f"Failed pages: {len(self.failed_pages)}")
        print(f"Total products: {len(self.all_products)}")
        print(f"Duration: {duration:.2f} seconds")
        print("="*50)


async def main():
    scraper = BookScraper(total_pages=295, per_page=24)
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())

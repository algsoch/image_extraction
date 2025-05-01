import os
import re
import csv
import time
import json
import uuid
import asyncio
import aiofiles
import logging
import zipfile
from PIL import Image
import pytesseract  # Add OCR library
from bs4 import BeautifulSoup
from io import BytesIO, StringIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("image-scraper")

# Constants
DEFAULT_TIMEOUT = 30  # Default timeout for HTTP requests in seconds

# Initialize FastAPI app
app = FastAPI(
    title="Image Extractor Tool",
    description="Extract images from websites and save details to CSV",
    version="1.0.0",
)

# Create static directory for assets
os.makedirs("static", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    SVG = "svg"
    ALL = "all"

class ScrapeRequest(BaseModel):
    url: str  # Using string to allow more flexible URL formats
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    formats: List[ImageFormat] = [ImageFormat.ALL]
    download_images: bool = False
    max_images: Optional[int] = None
    follow_links: bool = False
    css_selector: Optional[str] = None
    
    # Add a validator to fix URLs without scheme
    @validator('url')
    def ensure_url_scheme(cls, v):
        if not v.startswith(('http://', 'https://')):
            return 'https://' + v
        return v

class ImageInfo(BaseModel):
    url: str
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    size: Optional[int] = None
    format: Optional[str] = None
    alt_text: Optional[str] = None
    source_page: str
    extracted_text: Optional[str] = None  # New field for OCR extracted text

class ScrapeResult(BaseModel):
    task_id: str
    url: str
    total_images: int
    extracted_images: int
    processing_time: float
    images: List[ImageInfo]
    download_images: bool = False  # Added this field to enable ZIP download option

# Storage for tasks
tasks = {}

# Utility functions
def get_file_extension(url: str) -> str:
    """Extract file extension from URL"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    extension = os.path.splitext(path)[1].lower().lstrip('.')
    return extension if extension else "unknown"

def is_valid_image_url(url: str, allowed_formats: List[ImageFormat]) -> bool:
    """Check if URL points to a valid image based on extension"""
    if ImageFormat.ALL in allowed_formats:
        return True
    
    extension = get_file_extension(url)
    return extension in [fmt.value for fmt in allowed_formats]

def extract_text_from_image(image):
    """Extract text from an image using OCR"""
    try:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)
        # Clean up extracted text (remove extra whitespace)
        text = ' '.join(text.split())
        return text if text else None
    except Exception as e:
        logger.warning(f"Error extracting text from image: {str(e)}")
        return None

def extract_urls_from_json(json_data, urls=None):
    """Recursively extract URLs from JSON data"""
    if urls is None:
        urls = []
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            # Check if the key suggests this might be a URL
            if any(url_hint in key.lower() for url_hint in ['url', 'src', 'link', 'href', 'image']):
                if isinstance(value, str) and (value.startswith('http') or value.startswith('/')):
                    urls.append(value)
            # Recursively check nested dictionaries and lists
            if isinstance(value, (dict, list)):
                extract_urls_from_json(value, urls)
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, (dict, list)):
                extract_urls_from_json(item, urls)
            elif isinstance(item, str) and (item.startswith('http') or item.startswith('/')):
                urls.append(item)
    
    return urls

def is_likely_image_url(url):
    """Check if a URL is likely an image URL based on common patterns"""
    # Check for common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.tiff']
    url_lower = url.lower()
    
    if any(url_lower.endswith(ext) for ext in image_extensions):
        return True
    
    # Check for common image URL patterns
    image_patterns = [
        '/images/', '/img/', '/photos/', '/thumbnails/', 
        'image', 'photo', 'picture', 'asset', 'media',
        'width=', 'height=', 'w=', 'h='
    ]
    
    if any(pattern in url_lower for pattern in image_patterns):
        # Avoid common non-image URLs that might contain these patterns
        exclude_patterns = ['.js', '.css', '.html', '.php?', '?page=', '/api/']
        if not any(exclude in url_lower for exclude in exclude_patterns):
            return True
    
    return False

async def get_image_metadata(client, image_url: str, source_url: str) -> Optional[ImageInfo]:
    """Fetch metadata for an image URL"""
    try:
        # Parse URL to get filename
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path) or f"image_{uuid.uuid4()}.jpg"
        
        # Initialize metadata
        metadata = {
            "url": image_url,
            "filename": filename,
            "source_page": source_url
        }
        
        # Fetch image to get size and dimensions
        response = await client.get(image_url, follow_redirects=True, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            content = response.content
            
            # Try to determine dimensions
            try:
                image = Image.open(BytesIO(content))
                width, height = image.size
                metadata["width"] = width
                metadata["height"] = height
                metadata["format"] = image.format.lower() if image.format else None
                metadata["size"] = len(content)
                
                # Extract text using OCR
                extracted_text = extract_text_from_image(image)
                if extracted_text:
                    metadata["extracted_text"] = extracted_text
            except Exception as e:
                logger.warning(f"Error processing image {image_url}: {str(e)}")
        
        return ImageInfo(**metadata)
    except Exception as e:
        logger.error(f"Error fetching image metadata for {image_url}: {str(e)}")
        return None

async def scrape_images(request: ScrapeRequest) -> ScrapeResult:
    """Main function to scrape images from a URL"""
    start_time = time.time()
    task_id = str(uuid.uuid4())
    
    logger.info(f"Scraping images from {request.url}")
    
    # Setup HTTP client with custom headers to bypass simple anti-scraping measures
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }
    
    timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=30.0)
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
        try:
            # Fetch the main page
            logger.info(f"Fetching URL: {request.url}")
            response = await client.get(str(request.url))
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Log partial HTML for debugging
            html_preview = response.text[:1000] + "..." if len(response.text) > 1000 else response.text
            logger.info(f"Received HTML response: {html_preview}")
            
            # Extract image URLs
            image_urls = set()
            
            # Find images based on CSS selector or default to all img tags
            if request.css_selector:
                for img in soup.select(request.css_selector):
                    if img.get('src'):
                        image_urls.add(img['src'])
                    elif img.get('data-src'):  # Some sites use data-src for lazy loading
                        image_urls.add(img['data-src'])
                    # Look for additional attributes that might contain image URLs
                    for attr in ['data-lazy-src', 'data-original', 'data-srcset']:
                        if img.get(attr):
                            image_urls.add(img[attr].split(',')[0].strip().split(' ')[0])
            else:
                # Get all img tags with various src attributes
                for img in soup.find_all('img'):
                    for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                        if img.get(attr):
                            image_urls.add(img[attr])
                    
                    # Handle srcset attribute
                    if img.get('srcset'):
                        for srcset_url in img['srcset'].split(','):
                            url = srcset_url.strip().split(' ')[0]
                            if url:
                                image_urls.add(url)
                
                # Check for image URLs in JSON data (common in modern sites)
                scripts = soup.find_all('script', type='application/json')
                for script in scripts:
                    try:
                        if script.string:
                            data = json.loads(script.string)
                            # Look for URLs in the JSON data recursively
                            urls_from_json = extract_urls_from_json(data)
                            for url in urls_from_json:
                                if is_likely_image_url(url):
                                    image_urls.add(url)
                    except json.JSONDecodeError:
                        continue
                
                # Also get CSS background images
                for tag in soup.find_all(['div', 'section', 'span', 'a']):
                    if tag.get('style') and 'background-image' in tag.get('style'):
                        # Extract URL from inline style
                        bg_url_match = re.search(r'background-image:\s*url\([\'"]?([^\'"]+)[\'"]?\)', tag['style'])
                        if bg_url_match:
                            image_urls.add(bg_url_match.group(1))
                    
                    # Check data-bg attribute (used by some lazy load plugins)
                    if tag.get('data-bg'):
                        bg_url_match = re.search(r'url\([\'"]?([^\'"]+)[\'"]?\)', tag['data-bg'])
                        if bg_url_match:
                            image_urls.add(bg_url_match.group(1))
            
            # Log found image URLs
            logger.info(f"Found {len(image_urls)} raw image URLs")
            
            # Follow links if requested (simple implementation for demo)
            if request.follow_links:
                link_urls = set()
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    # Skip anchors and javascript
                    if not href.startswith('#') and not href.startswith('javascript:'):
                        # Make URL absolute
                        try:
                            absolute_link = str(httpx.URL(str(request.url)).join(httpx.URL(href)))
                            # Only follow links on the same domain
                            if urlparse(absolute_link).netloc == urlparse(str(request.url)).netloc:
                                link_urls.add(absolute_link)
                        except Exception as e:
                            logger.warning(f"Error processing link URL: {str(e)}")
                
                # Limit to 5 links for demo
                for link_url in list(link_urls)[:5]:
                    try:
                        link_response = await client.get(link_url)
                        if link_response.status_code == 200:
                            link_soup = BeautifulSoup(link_response.text, 'html.parser')
                            for img in link_soup.find_all('img'):
                                if img.get('src'):
                                    # Make URL absolute
                                    try:
                                        absolute_img_url = str(httpx.URL(link_url).join(httpx.URL(img['src'])))
                                        image_urls.add(absolute_img_url)
                                    except Exception as e:
                                        logger.warning(f"Error making image URL absolute: {str(e)}")
                    except Exception as e:
                        logger.warning(f"Error fetching link {link_url}: {str(e)}")
            
            # Make image URLs absolute
            absolute_image_urls = []
            for img_url in image_urls:
                # Skip data URLs
                if img_url.startswith('data:'):
                    continue
                
                # Make URL absolute
                try:
                    absolute_url = str(httpx.URL(str(request.url)).join(httpx.URL(img_url)))
                    absolute_image_urls.append(absolute_url)
                except Exception as e:
                    logger.warning(f"Error making URL absolute {img_url}: {str(e)}")
            
            logger.info(f"Processed {len(absolute_image_urls)} absolute image URLs")
            
            # Apply format filtering
            if not ImageFormat.ALL in request.formats:
                absolute_image_urls = [url for url in absolute_image_urls if is_valid_image_url(url, request.formats)]
            
            # Limit number of images if requested
            total_images = len(absolute_image_urls)
            if request.max_images:
                absolute_image_urls = absolute_image_urls[:request.max_images]
            
            # Create directory for downloaded images if needed
            download_dir = None
            if request.download_images:
                download_dir = os.path.join("results", f"images_{task_id}")
                os.makedirs(download_dir, exist_ok=True)
                
                # Create a text file to store extracted text
                text_file_path = os.path.join(download_dir, "extracted_text.txt")
            
            # Fetch metadata for each image
            image_metadata = []
            
            for img_url in absolute_image_urls:
                try:
                    # Parse filename from URL
                    parsed_url = urlparse(img_url)
                    filename = os.path.basename(parsed_url.path) or f"image_{len(image_metadata)}.jpg"
                    
                    # Initialize metadata
                    metadata = {
                        "url": img_url,
                        "filename": filename,
                        "source_page": str(request.url)
                    }
                    
                    # Try to get alt text if this is from an img tag
                    alt_text = None
                    for img in soup.find_all('img', src=lambda src: src and img_url.endswith(src)):
                        alt_text = img.get('alt')
                        break
                    if alt_text:
                        metadata["alt_text"] = alt_text
                    
                    # Fetch image to get dimensions and possibly save
                    try:
                        response = await client.get(img_url, follow_redirects=True, timeout=DEFAULT_TIMEOUT)
                        if response.status_code == 200:
                            content = response.content
                            
                            # Try to load image to get dimensions
                            image = Image.open(BytesIO(content))
                            width, height = image.size
                            metadata["width"] = width
                            metadata["height"] = height
                            metadata["format"] = image.format.lower() if image.format else None
                            metadata["size"] = len(content)
                            
                            # Extract text using OCR
                            extracted_text = extract_text_from_image(image)
                            if extracted_text:
                                metadata["extracted_text"] = extracted_text
                            
                            # Save the file if downloading is enabled
                            if request.download_images:
                                # Determine file extension
                                ext = metadata.get("format") or get_file_extension(img_url) or "jpg"
                                if not ext.startswith('.'):
                                    ext = f".{ext}"
                                
                                # Sanitize filename
                                safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
                                if not safe_filename:
                                    safe_filename = f"image_{len(image_metadata)}{ext}"
                                elif not safe_filename.endswith(ext):
                                    safe_filename = f"{safe_filename}{ext}"
                                
                                # Save the file
                                file_path = os.path.join(download_dir, safe_filename)
                                async with aiofiles.open(file_path, 'wb') as f:
                                    await f.write(content)
                                
                                # Update URL to point to local file for preview
                                metadata["local_url"] = f"/results/images_{task_id}/{safe_filename}"
                                
                                # Save extracted text to file if any
                                if extracted_text:
                                    async with aiofiles.open(text_file_path, 'a', encoding='utf-8') as f:
                                        await f.write(f"File: {safe_filename}\n")
                                        await f.write(f"Text: {extracted_text}\n")
                                        await f.write("-" * 50 + "\n")
                    except Exception as e:
                        logger.warning(f"Error processing image {img_url}: {str(e)}")
                    
                    # Add metadata even if we couldn't get everything
                    image_metadata.append(ImageInfo(**metadata))
                except Exception as e:
                    logger.warning(f"Error processing metadata for {img_url}: {str(e)}")
            
            # Filter out failed metadata fetches and apply size filtering
            filtered_metadata = []
            for meta in image_metadata:
                # Apply size filtering if specified
                if request.min_width and (not meta.width or meta.width < request.min_width):
                    continue
                if request.min_height and (not meta.height or meta.height < request.min_height):
                    continue
                
                filtered_metadata.append(meta)
            
            # Create a ZIP file of the images if we downloaded any
            if request.download_images and filtered_metadata:
                # Ensure the directory exists (could have been created but no images downloaded)
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir, exist_ok=True)
                
                zip_path = os.path.join("results", f"images_{task_id}.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(download_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.relpath(file_path, download_dir))
            
            # Create CSV file
            csv_data = StringIO()
            csv_writer = csv.writer(csv_data)
            
            # Write header with extracted text column
            csv_writer.writerow(["URL", "Filename", "Width", "Height", "Size (bytes)", "Format", "Alt Text", "Source Page", "Extracted Text"])
            
            # Write data
            for meta in filtered_metadata:
                csv_writer.writerow([
                    meta.url,
                    meta.filename,
                    meta.width or "",
                    meta.height or "",
                    meta.size or "",
                    meta.format or "",
                    meta.alt_text or "",
                    meta.source_page,
                    meta.extracted_text or ""
                ])
            
            # Save CSV
            csv_path = os.path.join("results", f"images_{task_id}.csv")
            async with aiofiles.open(csv_path, 'w', encoding='utf-8') as f:
                await f.write(csv_data.getvalue())
            
            # Serve static files from results directory
            if not any(mount.name == "results" for mount in app.routes):
                app.mount("/results", StaticFiles(directory="results"), name="results")
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Create result
            result = ScrapeResult(
                task_id=task_id,
                url=str(request.url),
                total_images=total_images,
                extracted_images=len(filtered_metadata),
                processing_time=processing_time,
                images=filtered_metadata,
                download_images=request.download_images
            )
            
            # Store task result
            tasks[task_id] = result
            
            # Log result summary
            logger.info(f"Completed task {task_id}: Found {total_images} images, extracted {len(filtered_metadata)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {request.url}: {str(e)}")
            # More descriptive error for debugging
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error details: {error_details}")
            raise HTTPException(status_code=500, detail=f"Error extracting images: {str(e)}")

# API endpoints
@app.post("/api/scrape", response_model=ScrapeResult)
async def scrape_images_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Extract images from a website and save to CSV"""
    try:
        # Ensure URL is properly formatted - this will use the validator in the ScrapeRequest model
        # No need to manually fix the URL here since the Pydantic model will handle it
        url = request.url  # The validator in the model will have already added https:// if needed
        
        # Log the URL being processed
        logger.info(f"Processing scrape request for URL: {url}")
        
        # Run the task immediately if not too large
        if request.max_images and request.max_images <= 20 and not request.follow_links:
            return await scrape_images(request)
        
        # Otherwise run in background
        task_id = str(uuid.uuid4())
        
        # Create a placeholder result
        result = ScrapeResult(
            task_id=task_id,
            url=url,
            total_images=0,
            extracted_images=0,
            processing_time=0,
            images=[]
        )
        
        # Store task
        tasks[task_id] = result
        
        # Run in background
        background_tasks.add_task(process_background_task, task_id, request)
        
        return result
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error extracting images: {str(e)}")

async def process_background_task(task_id: str, request: ScrapeRequest):
    """Process a scraping task in the background"""
    try:
        result = await scrape_images(request)
        tasks[task_id] = result
    except Exception as e:
        logger.error(f"Error processing background task {task_id}: {str(e)}")

@app.get("/api/task/{task_id}", response_model=ScrapeResult)
async def get_task(task_id: str):
    """Get the result of a scraping task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return tasks[task_id]

@app.get("/api/download/csv/{task_id}")
async def download_csv(task_id: str):
    """Download the CSV file for a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    csv_path = os.path.join("results", f"images_{task_id}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file for task {task_id} not found")
    
    return StreamingResponse(
        open(csv_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=images_{task_id}.csv"}
    )

@app.get("/api/download/zip/{task_id}")
async def download_zip(task_id: str):
    """Download the ZIP file of images for a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    zip_path = os.path.join("results", f"images_{task_id}.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail=f"ZIP file for task {task_id} not found")
    
    return StreamingResponse(
        open(zip_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=images_{task_id}.zip"}
    )

@app.post("/api/upload/html")
async def upload_html(html_file: UploadFile = File(...)):
    """Process HTML file for image extraction"""
    # Create a temporary file
    temp_file_path = f"temp_{uuid.uuid4()}.html"
    try:
        # Save uploaded file
        content = await html_file.read()
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)
        
        # Parse HTML
        with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract image URLs
        image_urls = set()
        
        # Standard img tags
        for img in soup.find_all('img'):
            if img.get('src'):
                image_urls.add(img['src'])
        
        # CSS background images
        for tag in soup.find_all(['div', 'section', 'span', 'a']):
            if tag.get('style') and 'background-image' in tag.get('style'):
                # Extract URL from inline style
                bg_url_match = re.search(r'background-image:\s*url\([\'"]?([^\'"]+)[\'"]?\)', tag['style'])
                if bg_url_match:
                    image_urls.add(bg_url_match.group(1))
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create directory for downloaded images
        download_dir = os.path.join("results", f"images_{task_id}")
        os.makedirs(download_dir, exist_ok=True)
        
        # Create a text file to store extracted text
        text_file_path = os.path.join(download_dir, "extracted_text.txt")
        
        # Create image metadata and try to download images if possible
        image_metadata = []
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=DEFAULT_TIMEOUT) as client:
            for img_url in image_urls:
                # Skip data URLs
                if img_url.startswith('data:'):
                    continue
                
                # Handle relative URLs - they won't work since we don't have a base URL
                filename = os.path.basename(img_url) or f"image_{len(image_metadata)}.jpg"
                extension = get_file_extension(img_url)
                
                # Get alt text if available
                alt_text = None
                for img in soup.find_all('img', src=img_url):
                    alt_text = img.get('alt')
                    break
                
                # Try to download and process the image if it's a valid URL
                extracted_text = None
                try:
                    # Only try to download if it looks like a URL
                    if img_url.startswith(('http://', 'https://', '//')):
                        full_url = img_url
                        if img_url.startswith('//'):
                            full_url = 'https:' + img_url
                            
                        response = await client.get(full_url)
                        if response.status_code == 200:
                            content = response.content
                            
                            # Save the image
                            safe_filename = re.sub(r'[^\w\-\.]', '_', filename)
                            if not safe_filename:
                                safe_filename = f"image_{len(image_metadata)}.jpg"
                                
                            file_path = os.path.join(download_dir, safe_filename)
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(content)
                                
                            # Extract text using OCR
                            try:
                                image = Image.open(BytesIO(content))
                                extracted_text = extract_text_from_image(image)
                                
                                # Save extracted text to file if any
                                if extracted_text:
                                    async with aiofiles.open(text_file_path, 'a', encoding='utf-8') as f:
                                        await f.write(f"File: {safe_filename}\n")
                                        await f.write(f"Text: {extracted_text}\n")
                                        await f.write("-" * 50 + "\n")
                            except Exception as e:
                                logger.warning(f"Error extracting text from image {img_url}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error downloading image {img_url}: {str(e)}")
                
                metadata = ImageInfo(
                    url=img_url,
                    filename=filename,
                    format=extension,
                    alt_text=alt_text,
                    source_page=f"Uploaded HTML file: {html_file.filename}",
                    extracted_text=extracted_text
                )
                image_metadata.append(metadata)
        
        # Create a ZIP file of the downloaded images
        zip_path = os.path.join("results", f"images_{task_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(download_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, download_dir))
        
        # Create CSV
        csv_data = StringIO()
        csv_writer = csv.writer(csv_data)
        
        # Write header
        csv_writer.writerow(["URL", "Filename", "Width", "Height", "Size (bytes)", "Format", "Alt Text", "Source Page", "Extracted Text"])
        
        # Write data
        for meta in image_metadata:
            csv_writer.writerow([
                meta.url,
                meta.filename,
                meta.width or "",
                meta.height or "",
                meta.size or "",
                meta.format or "",
                meta.alt_text or "",
                meta.source_page,
                meta.extracted_text or ""
            ])
        
        # Save CSV
        csv_path = os.path.join("results", f"images_{task_id}.csv")
        async with aiofiles.open(csv_path, 'w', encoding='utf-8') as f:
            await f.write(csv_data.getvalue())
        
        # Create result
        result = ScrapeResult(
            task_id=task_id,
            url=f"Uploaded HTML file: {html_file.filename}",
            total_images=len(image_metadata),
            extracted_images=len(image_metadata),
            processing_time=0.0,
            images=image_metadata,
            download_images=True  # Enable ZIP download
        )
        
        # Store task
        tasks[task_id] = result
        
        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/upload/images")
async def upload_images(background_tasks: BackgroundTasks, image_files: List[UploadFile] = File(...)):
    """Process uploaded image files"""
    # Generate task ID
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Create directory for uploaded images
        upload_dir = os.path.join("results", f"images_{task_id}")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create a text file to store extracted text
        text_file_path = os.path.join(upload_dir, "extracted_text.txt")
        
        # Process each image file
        image_metadata = []
        
        for i, img_file in enumerate(image_files):
            try:
                # Read image content
                content = await img_file.read()
                
                # Determine file name and extension
                original_filename = img_file.filename
                file_extension = os.path.splitext(original_filename)[1].lower()
                if not file_extension:
                    file_extension = ".jpg"  # Default extension
                
                # Create a sanitized filename
                safe_filename = re.sub(r'[^\w\-\.]', '_', original_filename)
                if not safe_filename or safe_filename.startswith('.'):
                    safe_filename = f"image_{i}{file_extension}"
                
                # Save the file
                file_path = os.path.join(upload_dir, safe_filename)
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)
                
                # Get image metadata
                try:
                    image = Image.open(BytesIO(content))
                    width, height = image.size
                    image_format = image.format.lower() if image.format else file_extension.lstrip('.')
                    
                    # Extract text from image using OCR
                    extracted_text = extract_text_from_image(image)
                    
                    # Save extracted text to file if any
                    if extracted_text:
                        async with aiofiles.open(text_file_path, 'a', encoding='utf-8') as f:
                            await f.write(f"File: {safe_filename}\n")
                            await f.write(f"Text: {extracted_text}\n")
                            await f.write("-" * 50 + "\n")
                    
                    # Create metadata
                    metadata = ImageInfo(
                        url=f"/results/images_{task_id}/{safe_filename}",  # Relative URL
                        filename=safe_filename,
                        width=width,
                        height=height,
                        size=len(content),
                        format=image_format,
                        alt_text=original_filename,
                        source_page=f"Uploaded image: {original_filename}",
                        extracted_text=extracted_text  # Add extracted text to metadata
                    )
                    image_metadata.append(metadata)
                except Exception as e:
                    logger.warning(f"Error processing image metadata for {original_filename}: {str(e)}")
                    # Add basic metadata without image dimensions
                    metadata = ImageInfo(
                        url=f"/results/images_{task_id}/{safe_filename}",
                        filename=safe_filename,
                        size=len(content),
                        format=file_extension.lstrip('.'),
                        alt_text=original_filename,
                        source_page=f"Uploaded image: {original_filename}"
                    )
                    image_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Error processing image file {img_file.filename}: {str(e)}")
        
        # Create a ZIP file of the images
        zip_path = os.path.join("results", f"images_{task_id}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(upload_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, upload_dir))
        
        # Create CSV file
        csv_data = StringIO()
        csv_writer = csv.writer(csv_data)
        
        # Write header with new extracted_text column
        csv_writer.writerow(["URL", "Filename", "Width", "Height", "Size (bytes)", "Format", "Alt Text", "Source Page", "Extracted Text"])
        
        # Write data
        for meta in image_metadata:
            csv_writer.writerow([
                meta.url,
                meta.filename,
                meta.width or "",
                meta.height or "",
                meta.size or "",
                meta.format or "",
                meta.alt_text or "",
                meta.source_page,
                meta.extracted_text or ""
            ])
        
        # Save CSV
        csv_path = os.path.join("results", f"images_{task_id}.csv")
        async with aiofiles.open(csv_path, 'w', encoding='utf-8') as f:
            await f.write(csv_data.getvalue())
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create result
        result = ScrapeResult(
            task_id=task_id,
            url=f"Uploaded images: {len(image_metadata)} files",
            total_images=len(image_metadata),
            extracted_images=len(image_metadata),
            processing_time=processing_time,
            images=image_metadata,
            download_images=True  # Enable ZIP download
        )
        
        # Store task
        tasks[task_id] = result
        
        # Serve static files from results directory
        if not any(mount.name == "results" for mount in app.routes):
            app.mount("/results", StaticFiles(directory="results"), name="results")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing uploaded images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

# Main HTML interface
@app.get("/", response_class=HTMLResponse)
async def get_html_page():
    """Return the HTML page for the image extractor tool"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Extractor Tool</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                padding-top: 2rem;
                padding-bottom: 2rem;
                background-color: #f8f9fa;
            }
            .card {
                border-radius: 1rem;
                box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
                margin-bottom: 2rem;
            }
            .card-header {
                background-color: #4a6bff;
                color: white;
                border-radius: 1rem 1rem 0 0 !important;
                padding: 1rem;
            }
            .results-container {
                display: none;
                margin-top: 2rem;
            }
            .image-preview {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin-top: 1rem;
            }
            .image-preview-item {
                width: 150px;
                height: 150px;
                object-fit: cover;
                border-radius: 0.5rem;
                border: 1px solid #dee2e6;
                cursor: pointer;
                transition: transform 0.2s;
            }
            .image-preview-item:hover {
                transform: scale(1.05);
                box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.15);
            }
            .spinner-container {
                display: none;
                margin: 2rem auto;
                text-align: center;
            }
            .result-links {
                margin-top: 1rem;
            }
            .result-links a {
                margin-right: 1rem;
            }
            #uploadForm, #imageUploadForm {
                display: none;
            }
            .form-toggle {
                margin-bottom: 1rem;
                text-align: center;
            }
            .form-toggle button {
                margin: 0 0.5rem;
            }
            .advanced-options {
                margin-top: 1rem;
                display: none;
            }
            .advanced-toggle {
                cursor: pointer;
                color: #4a6bff;
                margin-bottom: 1rem;
            }
            .drop-zone {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                transition: all 0.3s;
                margin-bottom: 1rem;
            }
            .drop-zone.active {
                border-color: #4a6bff;
                background-color: rgba(74, 107, 255, 0.1);
            }
            .drop-zone-prompt {
                margin-bottom: 1rem;
                font-size: 1.2rem;
                color: #6c757d;
            }
            /* Modal styles */
            .image-modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
                overflow: auto;
            }
            .modal-content {
                position: relative;
                margin: 2% auto;
                padding: 20px;
                width: 90%;
                max-width: 1200px;
                background-color: #fff;
                border-radius: 10px;
            }
            .modal-close {
                position: absolute;
                top: 10px;
                right: 20px;
                color: #333;
                font-size: 30px;
                font-weight: bold;
                cursor: pointer;
            }
            .modal-image-container {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .modal-image {
                max-width: 100%;
                max-height: 70vh;
                margin-bottom: 1rem;
                border-radius: 5px;
            }
            .modal-info {
                margin-top: 1rem;
                width: 100%;
            }
            .modal-actions {
                margin-top: 1rem;
                display: flex;
                gap: 1rem;
                justify-content: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">Image Extractor Tool</h1>
            <p class="text-center mb-4">Extract images from websites, HTML files, or upload your own images</p>
            
            <div class="form-toggle">
                <button id="urlToggle" class="btn btn-primary active">Extract from URL</button>
                <button id="fileToggle" class="btn btn-secondary">Upload HTML File</button>
                <button id="imageToggle" class="btn btn-secondary">Upload Images</button>
            </div>
            
            <div class="card" id="urlForm">
                <div class="card-header">
                    <h5 class="card-title mb-0">Extract Images from Website</h5>
                </div>
                <div class="card-body">
                    <form id="scrapeForm">
                        <div class="mb-3">
                            <label for="url" class="form-label">Website URL</label>
                            <input type="url" class="form-control" id="url" placeholder="https://example.com" required>
                        </div>
                        
                        <div class="advanced-toggle">
                            <i class="bi bi-chevron-down"></i> Show Advanced Options
                        </div>
                        
                        <div class="advanced-options">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="minWidth" class="form-label">Minimum Width (pixels)</label>
                                        <input type="number" class="form-control" id="minWidth" placeholder="e.g. 300">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label for="minHeight" class="form-label">Minimum Height (pixels)</label>
                                        <input type="number" class="form-control" id="minHeight" placeholder="e.g. 300">
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="maxImages" class="form-label">Maximum Images to Extract</label>
                                <input type="number" class="form-control" id="maxImages" placeholder="e.g. 50">
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Image Formats</label>
                                <div class="form-check">
                                    <input class="form-check-input format-check" type="checkbox" value="ALL" id="formatAll" checked>
                                    <label class="form-check-label" for="formatAll">All Formats</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input format-check" type="checkbox" value="JPEG" id="formatJpeg">
                                    <label class="form-check-label" for="formatJpeg">JPEG</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input format-check" type="checkbox" value="PNG" id="formatPng">
                                    <label class="form-check-label" for="formatPng">PNG</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input format-check" type="checkbox" value="GIF" id="formatGif">
                                    <label class="form-check-label" for="formatGif">GIF</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input format-check" type="checkbox" value="WEBP" id="formatWebp">
                                    <label class="form-check-label" for="formatWebp">WEBP</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input format-check" type="checkbox" value="SVG" id="formatSvg">
                                    <label class="form-check-label" for="formatSvg">SVG</label>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="followLinks">
                                    <label class="form-check-label" for="followLinks">
                                        Follow Links on Page (finds more images)
                                    </label>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="downloadImages">
                                    <label class="form-check-label" for="downloadImages">
                                        Download Images (creates ZIP file)
                                    </label>
                                </div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="cssSelector" class="form-label">CSS Selector (advanced)</label>
                                <input type="text" class="form-control" id="cssSelector" placeholder="e.g. .article-content img">
                                <div class="form-text">Only extract images matching this CSS selector</div>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary">Extract Images</button>
                    </form>
                </div>
            </div>
            
            <div class="card" id="uploadForm">
                <div class="card-header">
                    <h5 class="card-title mb-0">Extract Images from HTML File</h5>
                </div>
                <div class="card-body">
                    <form id="uploadHtmlForm" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="htmlFile" class="form-label">HTML File</label>
                            <input type="file" class="form-control" id="htmlFile" accept=".html,.htm" required>
                        </div>
                        <button type="submit" class="btn btn-primary">Extract Images</button>
                    </form>
                </div>
            </div>
            
            <div class="card" id="imageUploadForm">
                <div class="card-header">
                    <h5 class="card-title mb-0">Upload Images</h5>
                </div>
                <div class="card-body">
                    <form id="imageForm" enctype="multipart/form-data">
                        <div class="drop-zone" id="dropZone">
                            <div class="drop-zone-prompt">
                                <p>Drag & drop image files here</p>
                                <p>- or -</p>
                            </div>
                            <input type="file" class="form-control" id="imageFiles" accept="image/*" multiple>
                            <div class="mt-2 text-muted">Support for JPEG, PNG, GIF, WEBP, and SVG files</div>
                        </div>
                        <div class="selected-files" id="selectedFiles"></div>
                        <button type="submit" class="btn btn-primary">Process Images</button>
                    </form>
                </div>
            </div>
            
            <div class="spinner-container">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Processing images, please wait...</p>
            </div>
            
            <div class="results-container">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Results</h5>
                    </div>
                    <div class="card-body">
                        <div id="resultsInfo"></div>
                        <div class="result-links">
                            <a href="#" id="csvLink" class="btn btn-success">Download CSV</a>
                            <a href="#" id="zipLink" class="btn btn-info" style="display:none;">Download Images (ZIP)</a>
                        </div>
                        <h5 class="mt-4">Images (Click to Preview)</h5>
                        <div class="image-preview" id="imagePreview"></div>
                    </div>
                </div>
            </div>
            
            <!-- Image Modal -->
            <div id="imageModal" class="image-modal">
                <div class="modal-content">
                    <span class="modal-close">&times;</span>
                    <div class="modal-image-container">
                        <img src="" id="modalImage" class="modal-image">
                        <div class="modal-info" id="modalInfo"></div>
                        <div class="modal-actions">
                            <a href="#" id="downloadImageBtn" class="btn btn-primary" download>Download Image</a>
                            <button id="closeModalBtn" class="btn btn-secondary">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Store extracted images data
            let extractedImages = [];
            
            document.addEventListener('DOMContentLoaded', function() {
                // Form toggle
                const urlToggle = document.getElementById('urlToggle');
                const fileToggle = document.getElementById('fileToggle');
                const imageToggle = document.getElementById('imageToggle');
                const urlForm = document.getElementById('urlForm');
                const uploadForm = document.getElementById('uploadForm');
                const imageUploadForm = document.getElementById('imageUploadForm');
                
                urlToggle.addEventListener('click', function() {
                    resetToggles();
                    urlToggle.classList.add('active', 'btn-primary');
                    urlToggle.classList.remove('btn-secondary');
                    urlForm.style.display = 'block';
                });
                
                fileToggle.addEventListener('click', function() {
                    resetToggles();
                    fileToggle.classList.add('active', 'btn-primary');
                    fileToggle.classList.remove('btn-secondary');
                    uploadForm.style.display = 'block';
                });
                
                imageToggle.addEventListener('click', function() {
                    resetToggles();
                    imageToggle.classList.add('active', 'btn-primary');
                    imageToggle.classList.remove('btn-secondary');
                    imageUploadForm.style.display = 'block';
                });
                
                function resetToggles() {
                    // Reset toggle buttons
                    [urlToggle, fileToggle, imageToggle].forEach(toggle => {
                        toggle.classList.remove('active', 'btn-primary');
                        toggle.classList.add('btn-secondary');
                    });
                    
                    // Hide all forms
                    [urlForm, uploadForm, imageUploadForm].forEach(form => {
                        form.style.display = 'none';
                    });
                }
                
                // Advanced options toggle
                const advancedToggle = document.querySelector('.advanced-toggle');
                const advancedOptions = document.querySelector('.advanced-options');
                
                advancedToggle.addEventListener('click', function() {
                    if (advancedOptions.style.display === 'block') {
                        advancedOptions.style.display = 'none';
                        advancedToggle.innerHTML = '<i class="bi bi-chevron-down"></i> Show Advanced Options';
                    } else {
                        advancedOptions.style.display = 'block';
                        advancedToggle.innerHTML = '<i class="bi bi-chevron-up"></i> Hide Advanced Options';
                    }
                });
                
                // Format checkboxes
                const formatAll = document.getElementById('formatAll');
                const formatChecks = document.querySelectorAll('.format-check:not(#formatAll)');
                
                formatAll.addEventListener('change', function() {
                    if (this.checked) {
                        formatChecks.forEach(check => {
                            check.checked = false;
                            check.disabled = true;
                        });
                    } else {
                        formatChecks.forEach(check => {
                            check.disabled = false;
                        });
                    }
                });
                
                formatChecks.forEach(check => {
                    check.addEventListener('change', function() {
                        if (this.checked) {
                            formatAll.checked = false;
                        }
                        
                        // If no format is selected, select All
                        const anyChecked = Array.from(formatChecks).some(c => c.checked);
                        if (!anyChecked) {
                            formatAll.checked = true;
                            formatChecks.forEach(c => c.disabled = true);
                        }
                    });
                });
                
                // Scrape form submit
                const scrapeForm = document.getElementById('scrapeForm');
                const spinnerContainer = document.querySelector('.spinner-container');
                const resultsContainer = document.querySelector('.results-container');
                
                scrapeForm.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    // Show spinner, hide results
                    spinnerContainer.style.display = 'block';
                    resultsContainer.style.display = 'none';
                    
                    // Get form values
                    const url = document.getElementById('url').value;
                    const minWidth = document.getElementById('minWidth').value || null;
                    const minHeight = document.getElementById('minHeight').value || null;
                    const maxImages = document.getElementById('maxImages').value || null;
                    const followLinks = document.getElementById('followLinks').checked;
                    const downloadImages = document.getElementById('downloadImages').checked;
                    const cssSelector = document.getElementById('cssSelector').value || null;
                    
                    // Get selected formats
                    let formats = [];
                    if (formatAll.checked) {
                        formats = ['ALL'];
                    } else {
                        formatChecks.forEach(check => {
                            if (check.checked) {
                                formats.push(check.value);
                            }
                        });
                    }
                    
                    // Prepare request data
                    const requestData = {
                        url: url,
                        formats: formats,
                        follow_links: followLinks,
                        download_images: downloadImages
                    };
                    
                    if (minWidth) requestData.min_width = parseInt(minWidth);
                    if (minHeight) requestData.min_height = parseInt(minHeight);
                    if (maxImages) requestData.max_images = parseInt(maxImages);
                    if (cssSelector) requestData.css_selector = cssSelector;
                    
                    try {
                        // Send request
                        const response = await fetch('/api/scrape', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(requestData)
                        });
                        
                        if (!response.ok) {
                            throw new Error('Error extracting images');
                        }
                        
                        const result = await response.json();
                        
                        // If task is still processing (background task)
                        if (result.extracted_images === 0 && result.task_id) {
                            await pollTaskStatus(result.task_id);
                        } else {
                            displayResults(result);
                        }
                    } catch (error) {
                        console.error('Error:', error);
                        alert('Error extracting images: ' + error.message);
                        spinnerContainer.style.display = 'none';
                    }
                });
                
                // Upload HTML form submit
                const uploadHtmlForm = document.getElementById('uploadHtmlForm');
                
                uploadHtmlForm.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    // Show spinner, hide results
                    spinnerContainer.style.display = 'block';
                    resultsContainer.style.display = 'none';
                    
                    // Get form values
                    const htmlFile = document.getElementById('htmlFile').files[0];
                    
                    // Create form data - parameter name must match server-side
                    const formData = new FormData();
                    formData.append('html_file', htmlFile); // Changed from 'htmlFile' to 'html_file'
                    
                    try {
                        // Send request
                        const response = await fetch('/api/upload/html', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`Error processing HTML file: ${response.status} ${response.statusText}`);
                        }
                        
                        const result = await response.json();
                        displayResults(result);
                    } catch (error) {
                        console.error('Error:', error);
                        alert('Error processing HTML file: ' + error.message);
                        spinnerContainer.style.display = 'none';
                    }
                });
                
                // Image upload functionality
                const imageForm = document.getElementById('imageForm');
                const dropZone = document.getElementById('dropZone');
                const imageFilesInput = document.getElementById('imageFiles');
                const selectedFilesDiv = document.getElementById('selectedFiles');
                
                // Drag and drop functionality
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    dropZone.addEventListener(eventName, preventDefaults, false);
                });
                
                function preventDefaults(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                
                ['dragenter', 'dragover'].forEach(eventName => {
                    dropZone.addEventListener(eventName, highlight, false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    dropZone.addEventListener(eventName, unhighlight, false);
                });
                
                function highlight() {
                    dropZone.classList.add('active');
                }
                
                function unhighlight() {
                    dropZone.classList.remove('active');
                }
                
                dropZone.addEventListener('drop', handleDrop, false);
                
                function handleDrop(e) {
                    const dt = e.dataTransfer;
                    const files = dt.files;
                    imageFilesInput.files = files;
                    updateFileList();
                }
                
                imageFilesInput.addEventListener('change', updateFileList);
                
                function updateFileList() {
                    const files = imageFilesInput.files;
                    
                    if (files.length > 0) {
                        let fileList = '<ul class="list-group">';
                        for (let i = 0; i < files.length; i++) {
                            fileList += `<li class="list-group-item">${files[i].name} (${formatFileSize(files[i].size)})</li>`;
                        }
                        fileList += '</ul>';
                        selectedFilesDiv.innerHTML = fileList;
                    } else {
                        selectedFilesDiv.innerHTML = '<p class="text-muted">No files selected</p>';
                    }
                }
                
                function formatFileSize(bytes) {
                    if (bytes < 1024) return bytes + ' bytes';
                    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
                    else return (bytes / 1048576).toFixed(1) + ' MB';
                }
                
                imageForm.addEventListener('submit', async function(e) {
                    e.preventDefault();
                    
                    // Show spinner, hide results
                    spinnerContainer.style.display = 'block';
                    resultsContainer.style.display = 'none';
                    
                    // Get form values
                    const files = imageFilesInput.files;
                    
                    if (files.length === 0) {
                        alert('Please select at least one image file');
                        spinnerContainer.style.display = 'none';
                        return;
                    }
                    
                    // Create form data
                    const formData = new FormData();
                    for (let i = 0; i < files.length; i++) {
                        formData.append('image_files', files[i]);
                    }
                    
                    try {
                        // Send request
                        const response = await fetch('/api/upload/images', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error('Error processing image files');
                        }
                        
                        const result = await response.json();
                        displayResults(result);
                    } catch (error) {
                        console.error('Error:', error);
                        alert('Error processing image files: ' + error.message);
                        spinnerContainer.style.display = 'none';
                    }
                });
                
                // Poll task status
                async function pollTaskStatus(taskId) {
                    try {
                        const response = await fetch(`/api/task/${taskId}`);
                        
                        if (!response.ok) {
                            throw new Error('Error fetching task status');
                        }
                        
                        const result = await response.json();
                        
                        // If task is complete
                        if (result.extracted_images > 0) {
                            displayResults(result);
                        } else {
                            // Poll again after 1 second
                            setTimeout(() => pollTaskStatus(taskId), 1000);
                        }
                    } catch (error) {
                        console.error('Error polling task status:', error);
                        alert('Error checking task status: ' + error.message);
                        spinnerContainer.style.display = 'none';
                    }
                }
                
                // Modal functionality
                const imageModal = document.getElementById('imageModal');
                const modalImage = document.getElementById('modalImage');
                const modalInfo = document.getElementById('modalInfo');
                const downloadImageBtn = document.getElementById('downloadImageBtn');
                const closeModalBtn = document.getElementById('closeModalBtn');
                const modalClose = document.querySelector('.modal-close');
                
                // Close modal when clicking the X
                modalClose.addEventListener('click', closeModal);
                
                // Close modal when clicking the close button
                closeModalBtn.addEventListener('click', closeModal);
                
                // Close modal when clicking outside of it
                window.addEventListener('click', function(event) {
                    if (event.target === imageModal) {
                        closeModal();
                    }
                });
                
                function closeModal() {
                    imageModal.style.display = 'none';
                }
                
                function openImageModal(imageIndex) {
                    const image = extractedImages[imageIndex];
                    
                    // Set image source
                    modalImage.src = image.url;
                    modalImage.alt = image.alt_text || image.filename;
                    
                    // Set download link
                    downloadImageBtn.href = image.url;
                    downloadImageBtn.download = image.filename;
                    
                    // Display image info
                    modalInfo.innerHTML = `
                        <div class="table-responsive">
                            <table class="table table-bordered">
                                <tbody>
                                    <tr>
                                        <th>Filename</th>
                                        <td>${image.filename}</td>
                                    </tr>
                                    <tr>
                                        <th>Dimensions</th>
                                        <td>${image.width || 'Unknown'} × ${image.height || 'Unknown'}</td>
                                    </tr>
                                    <tr>
                                        <th>Format</th>
                                        <td>${image.format || 'Unknown'}</td>
                                    </tr>
                                    <tr>
                                        <th>Size</th>
                                        <td>${image.size ? formatFileSize(image.size) : 'Unknown'}</td>
                                    </tr>
                                    ${image.alt_text ? `<tr><th>Alt Text</th><td>${image.alt_text}</td></tr>` : ''}
                                    ${image.extracted_text ? `<tr><th>Extracted Text</th><td>${image.extracted_text}</td></tr>` : ''}
                                </tbody>
                            </table>
                        </div>
                    `;
                    
                    // Show modal
                    imageModal.style.display = 'block';
                }
                
                // Display results
                function displayResults(result) {
                    // Store images data for modal
                    extractedImages = result.images;
                    
                    // Hide spinner, show results
                    spinnerContainer.style.display = 'none';
                    resultsContainer.style.display = 'block';
                    
                    // Update results info
                    const resultsInfo = document.getElementById('resultsInfo');
                    resultsInfo.innerHTML = `
                        <p><strong>Source:</strong> ${result.url}</p>
                        <p><strong>Total Images Found:</strong> ${result.total_images}</p>
                        <p><strong>Images Extracted:</strong> ${result.extracted_images}</p>
                        <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)} seconds</p>
                    `;
                    
                    // Update download links
                    const csvLink = document.getElementById('csvLink');
                    csvLink.href = `/api/download/csv/${result.task_id}`;
                    
                    const zipLink = document.getElementById('zipLink');
                    if (result.download_images) {
                        zipLink.href = `/api/download/zip/${result.task_id}`;
                        zipLink.style.display = 'inline-block';
                    } else {
                        zipLink.style.display = 'none';
                    }
                    
                    // Display image previews
                    const imagePreview = document.getElementById('imagePreview');
                    imagePreview.innerHTML = '';
                    
                    // Show images as preview
                    result.images.forEach((image, index) => {
                        // Skip if no URL
                        if (!image.url) return;
                        
                        // Skip data URLs
                        if (image.url.startsWith('data:')) return;
                        
                        // Create container for the image and its info
                        const container = document.createElement('div');
                        container.className = 'image-item';
                        container.style.margin = '10px';
                        container.style.textAlign = 'center';
                        
                        // Create image element
                        const img = document.createElement('img');
                        img.className = 'image-preview-item';
                        img.src = image.url;
                        img.alt = image.alt_text || image.filename;
                        img.title = `${image.filename} (Click to preview)`;
                        img.dataset.index = index;
                        
                        // Add error handling
                        img.onerror = function() {
                            this.src = 'https://via.placeholder.com/150?text=Image+Not+Available';
                        };
                        
                        // Add event listener to open modal
                        img.addEventListener('click', function() {
                            openImageModal(this.dataset.index);
                        });
                        
                        // Add caption with filename
                        const caption = document.createElement('div');
                        caption.className = 'mt-1 small text-truncate';
                        caption.style.maxWidth = '150px';
                        caption.title = image.filename;
                        caption.textContent = image.filename;
                        
                        // Add to container
                        container.appendChild(img);
                        container.appendChild(caption);
                        imagePreview.appendChild(container);
                    });
                    
                    // If no images to preview
                    if (result.images.length === 0) {
                        imagePreview.innerHTML = '<p>No images available for preview.</p>';
                    }
                }
            });
        </script>
    </body>
    </html>
    """

# For testing with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_extractor:app", host="0.0.0.0", port=8000, reload=True)

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from fastapi import HTTPException
import traceback
import logging

logger = logging.getLogger("image-scraper")

async def scrape_images_from_url(url):
    """Fetch and extract images from a URL"""
    try:
        # Setup HTTP client with proper headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30) as client:
            logger.info(f"Fetching URL: {url}")
            # Fetch the webpage
            response = await client.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all image tags
            image_elements = soup.find_all('img')
            images = []
            
            # Extract image URLs and metadata
            for img in image_elements:
                src = img.get('src')
                if (src):
                    # Skip data URLs
                    if src.startswith('data:'):
                        continue
                        
                    # Convert relative URLs to absolute
                    if not src.startswith(('http://', 'https://')):
                        src = urljoin(url, src)
                    
                    # Extract alt text and other attributes
                    alt_text = img.get('alt', '')
                    width = img.get('width', '')
                    height = img.get('height', '')
                    
                    images.append({
                        'url': src,
                        'alt_text': alt_text,
                        'width': width,
                        'height': height,
                        'source_page': url
                    })
            
            # Also look for background images in CSS
            for tag in soup.find_all(['div', 'section', 'span', 'a']):
                if tag.get('style') and 'background-image' in tag.get('style'):
                    # Extract URL from inline style
                    import re
                    bg_url_match = re.search(r'background-image:\s*url\([\'"]?([^\'"]+)[\'"]?\)', tag['style'])
                    if bg_url_match:
                        src = bg_url_match.group(1)
                        # Convert relative URLs to absolute
                        if not src.startswith(('http://', 'https://')):
                            src = urljoin(url, src)
                        
                        images.append({
                            'url': src,
                            'alt_text': '',
                            'width': '',
                            'height': '',
                            'source_page': url
                        })
            
            logger.info(f"Found {len(images)} images at {url}")
            return {'images': images, 'count': len(images), 'source_url': url}
            
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors (4xx, 5xx responses)
        error_detail = f"HTTP Error: {e.response.status_code} - {e.response.reason_phrase}"
        logger.error(f"Error details: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, 
                            detail=f"Error accessing URL: {error_detail}")
    
    except httpx.RequestError as e:
        # Handle request errors (connection issues, timeouts, etc.)
        error_detail = f"Request Error: {str(e)}"
        logger.error(f"Error details: {error_detail}")
        raise HTTPException(status_code=500, 
                            detail=f"Network error when accessing URL: {str(e)}")
    
    except Exception as e:
        # Catch and provide detailed traceback for debugging
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        raise HTTPException(status_code=500, 
                            detail=f"Error extracting images: {str(e)}")

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """WebSocket endpoint for real-time image processing updates"""
    # Generate a client ID if not provided
    if client_id is None:
        client_id = str(uuid.uuid4())
    
    try:
        # Accept the connection
        await manager.connect(websocket, client_id)
        
        # Send a welcome message
        await manager.send_message(json.dumps({
            "type": "connection_established",
            "client_id": client_id,
            "message": "Connected to image extractor WebSocket"
        }), client_id)
        
        # Keep the connection open and process messages
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            
            try:
                # Parse the message as JSON
                message = json.loads(data)
                message_type = message.get("type", "")
                
                # Process different message types
                if message_type == "ping":
                    # Simple ping-pong to keep connection alive
                    await manager.send_message(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }), client_id)
                
                elif message_type == "scrape_url":
                    # Extract URL from message
                    url = message.get("url")
                    if not url:
                        await manager.send_message(json.dumps({
                            "type": "error",
                            "message": "URL is required"
                        }), client_id)
                        continue
                    
                    # Send acknowledgment
                    await manager.send_message(json.dumps({
                        "type": "processing",
                        "message": f"Processing URL: {url}"
                    }), client_id)
                    
                    # Process the URL in the background
                    try:
                        result = await scrape_images_from_url(url)
                        # Send results back
                        await manager.send_message(json.dumps({
                            "type": "result",
                            "result": result
                        }), client_id)
                    except Exception as e:
                        await manager.send_message(json.dumps({
                            "type": "error",
                            "message": f"Error processing URL: {str(e)}"
                        }), client_id)
                
                else:
                    # Unknown message type
                    await manager.send_message(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }), client_id)
            
            except json.JSONDecodeError:
                # Handle invalid JSON
                await manager.send_message(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON message"
                }), client_id)
            except Exception as e:
                # Handle other errors
                logger.error(f"WebSocket error: {str(e)}")
                await manager.send_message(json.dumps({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                }), client_id)
    
    except WebSocketDisconnect:
        # Client disconnected
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        # Handle connection errors
        logger.error(f"WebSocket connection error: {str(e)}")
        # Try to disconnect if possible
        try:
            manager.disconnect(client_id)
        except:
            pass

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for client_id, websocket in self.active_connections.items():
            await websocket.send_text(message)

# Initialize the connection manager
manager = ConnectionManager()
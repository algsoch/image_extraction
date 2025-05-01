import os
import re
import csv
import time
import json
import uuid
import base64
import logging
import zipfile
import httpx
import asyncio
import aiofiles
from PIL import Image
from bs4 import BeautifulSoup
from io import BytesIO, StringIO
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("image-scraper")

# Constants
CACHE_TTL = 3600  # Cache time to live in seconds
DEFAULT_TIMEOUT = 30  # Default timeout for HTTP requests in seconds
MAX_WORKERS = 5  # Maximum number of concurrent workers
RATE_LIMIT = 5  # Number of requests allowed per minute per IP

# Initialize FastAPI app
app = FastAPI(
    title="Image Scraper Chatbot",
    description="Interactive chatbot for extracting images from websites",
    version="1.0.0",
    docs_url="/api/docs",  # Move docs to /api/docs to make root available for chat UI
    redoc_url="/api/redoc",
)

# Create static directory for chat interface files
os.makedirs("static", exist_ok=True)

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

# In-memory caching
cache = {}
rate_limits = {}
active_connections = {}

# Simple API key auth
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "test_api_key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Models
class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    SVG = "svg"
    ALL = "all"

class ScrapingDepth(str, Enum):
    BASIC = "basic"
    MEDIUM = "medium"
    DEEP = "deep"

class SortOrder(str, Enum):
    SIZE_ASC = "size_asc"
    SIZE_DESC = "size_desc"
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"
    DATE_ASC = "date_asc"
    DATE_DESC = "date_desc"

# Existing models
class ScraperRequest(BaseModel):
    url: HttpUrl
    formats: List[ImageFormat] = [ImageFormat.ALL]
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    max_images: Optional[int] = None
    include_base64: bool = False
    depth: ScrapingDepth = ScrapingDepth.BASIC
    follow_links: bool = False
    sort_by: SortOrder = SortOrder.SIZE_DESC
    download_images: bool = False
    custom_css_selector: Optional[str] = None
    exclude_patterns: List[str] = []
    
    @validator('url')
    def validate_url(cls, v):
        parsed = urlparse(str(v))
        if not parsed.netloc:
            raise ValueError("Invalid URL: must contain domain")
        return v

class ImageMetadata(BaseModel):
    url: str
    filename: str
    alt_text: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    format: Optional[str] = None
    aspect_ratio: Optional[float] = None
    source_page: Optional[str] = None
    extraction_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    base64_data: Optional[str] = None

class ScraperResponse(BaseModel):
    request_id: str
    url: str
    total_images_found: int
    images_extracted: int
    processing_time_ms: float
    cache_hit: bool = False
    images: List[ImageMetadata]

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None
    result_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

# New chatbot models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage] = []
    current_task_id: Optional[str] = None
    context: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    message: str
    suggestions: List[str] = []
    data: Optional[Dict[str, Any]] = None
    
# Chat storage
chat_sessions = {}

# Background tasks storage
tasks_store = {}

# Utility functions (keeping your existing ones)
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

def generate_cache_key(request: ScraperRequest) -> str:
    """Generate a unique cache key based on request parameters"""
    # Convert request to a string and create a hash
    request_dict = request.dict()
    request_str = json.dumps(request_dict, sort_keys=True)
    return hashlib.md5(request_str.encode()).hexdigest()

# Keeping your existing image processing functions (fetch_image_metadata, scrape_images_from_url, etc.)
# ...

async def fetch_image_metadata(
    client: httpx.AsyncClient, 
    image_url: str, 
    source_page: str, 
    include_base64: bool = False
) -> Optional[ImageMetadata]:
    """Fetch and extract metadata for a single image"""
    try:
        # Parse filename from URL
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        
        # Initialize metadata
        metadata = {
            "url": image_url,
            "filename": filename,
            "source_page": source_page,
            "extraction_time": datetime.now().isoformat()
        }
        
        # Fetch image head first to check content type
        head_response = await client.head(image_url, follow_redirects=True, timeout=DEFAULT_TIMEOUT)
        if head_response.status_code == 200:
            content_type = head_response.headers.get("content-type", "")
            if "image" in content_type:
                metadata["format"] = content_type.split("/")[-1]
            
            content_length = head_response.headers.get("content-length")
            if content_length and content_length.isdigit():
                metadata["size_bytes"] = int(content_length)
        
        # Fetch the actual image to get dimensions
        response = await client.get(image_url, follow_redirects=True, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 200:
            try:
                # Try to load image to get dimensions
                image = PIL.Image.open(BytesIO(response.content))
                width, height = image.size
                metadata["width"] = width
                metadata["height"] = height
                metadata["aspect_ratio"] = width / height if height > 0 else None
                metadata["format"] = image.format.lower() if image.format else metadata.get("format")
                
                # Include base64 data if requested
                if include_base64:
                    buffered = BytesIO()
                    image.save(buffered, format=image.format)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    metadata["base64_data"] = f"data:image/{metadata['format']};base64,{img_str}"
            except Exception as e:
                logger.warning(f"Error processing image {image_url}: {str(e)}")
        
        return ImageMetadata(**metadata)
    except Exception as e:
        logger.error(f"Error fetching image metadata for {image_url}: {str(e)}")
        return None

# NLP Intent Parser (new)
class Intent:
    SCRAPE_IMAGES = "scrape_images"
    GET_STATUS = "get_status"
    DOWNLOAD_RESULTS = "download_results"
    HELP = "help"
    UNKNOWN = "unknown"

def parse_intent(message: str) -> Dict[str, Any]:
    """
    Parse user message to identify intent and extract parameters
    """
    message = message.lower().strip()
    
    # Default result with unknown intent
    result = {
        "intent": Intent.UNKNOWN,
        "params": {},
        "confidence": 0.0
    }
    
    # Check for scrape images intent
    scrape_patterns = [
        r"(?:scrape|extract|get|fetch).+images?.+(?:from|on|at)\s+(?:the\s+)?(?:url|website|page|link)?\s*(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?)",
        r"(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?).+(?:scrape|extract|get|fetch).+images",
        r"(?:find|get)\s+(?:all\s+)?images\s+(?:from|on)\s+(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?)",
        r"(?:https?://)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?)"
    ]
    
    for pattern in scrape_patterns:
        match = re.search(pattern, message)
        if match:
            url = match.group(1)
            # Make sure it's a proper URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            result["intent"] = Intent.SCRAPE_IMAGES
            result["params"]["url"] = url
            result["confidence"] = 0.9
            
            # Look for additional parameters
            if "minimum" in message and ("width" in message or "size" in message):
                width_match = re.search(r"minimum\s+width\s+(?:of\s+)?(\d+)(?:px)?", message)
                if width_match:
                    result["params"]["min_width"] = int(width_match.group(1))
            
            if "minimum" in message and "height" in message:
                height_match = re.search(r"minimum\s+height\s+(?:of\s+)?(\d+)(?:px)?", message)
                if height_match:
                    result["params"]["min_height"] = int(height_match.group(1))
            
            if "limit" in message or "maximum" in message or "max" in message:
                limit_match = re.search(r"(?:limit|maximum|max)\s+(?:of\s+)?(\d+)\s+images", message)
                if limit_match:
                    result["params"]["max_images"] = int(limit_match.group(1))
            
            formats = []
            if "jpeg" in message or "jpg" in message:
                formats.append(ImageFormat.JPEG)
            if "png" in message:
                formats.append(ImageFormat.PNG)
            if "gif" in message:
                formats.append(ImageFormat.GIF)
            if "webp" in message:
                formats.append(ImageFormat.WEBP)
            if "svg" in message:
                formats.append(ImageFormat.SVG)
                
            if formats:
                result["params"]["formats"] = formats
            
            return result
    
    # Check for get status intent
    if any(status_word in message for status_word in ["status", "progress", "how's it going", "how is it going", "update"]):
        task_match = re.search(r"task\s+([a-f0-9-]+)", message, re.IGNORECASE)
        result["intent"] = Intent.GET_STATUS
        if task_match:
            result["params"]["task_id"] = task_match.group(1)
        result["confidence"] = 0.8
        return result
    
    # Check for download results intent
    if any(download_word in message for download_word in ["download", "get results", "save", "export", "csv"]):
        task_match = re.search(r"task\s+([a-f0-9-]+)", message, re.IGNORECASE)
        result["intent"] = Intent.DOWNLOAD_RESULTS
        if task_match:
            result["params"]["task_id"] = task_match.group(1)
        result["confidence"] = 0.8
        return result
    
    # Check for help intent
    if any(help_word in message for help_word in ["help", "guide", "how to", "instructions", "examples"]):
        result["intent"] = Intent.HELP
        result["confidence"] = 0.9
        return result
    
    return result

# Function to generate chat response
async def generate_chat_response(session: ChatSession, message: str) -> ChatResponse:
    """
    Generate a response to a chat message
    """
    if not message:
        return ChatResponse(
            session_id=session.session_id,
            message="I didn't receive any message. Please try again.",
            suggestions=["Help me scrape images", "Enter a website URL"]
        )
    
    # Parse intent
    parsed = parse_intent(message)
    intent = parsed["intent"]
    params = parsed["params"]
    
    # Default response values
    response_message = ""
    suggestions = []
    data = {}
    session_id = session.session_id
    
    # Process based on intent
    if intent == Intent.SCRAPE_IMAGES:
        url = params.get("url")
        if url:
            try:
                # Determine if we should use async or immediate processing
                # Use async for deep scraping or image downloads
                use_async = (
                    params.get("depth", ScrapingDepth.BASIC) != ScrapingDepth.BASIC or
                    params.get("follow_links", False) or
                    params.get("download_images", False) or
                    params.get("max_images", 0) > 50  # Large image counts
                )
                
                if use_async:
                    # Set up request parameters
                    request_params = {
                        "url": url,
                        "formats": params.get("formats", [ImageFormat.ALL]),
                        "min_width": params.get("min_width"),
                        "min_height": params.get("min_height"),
                        "max_images": params.get("max_images"),
                        "depth": params.get("depth", ScrapingDepth.BASIC),
                        "follow_links": params.get("follow_links", False),
                        "download_images": params.get("download_images", False)
                    }
                    
                    # Remove None values
                    request_params = {k: v for k, v in request_params.items() if v is not None}
                    
                    # Create scraper request
                    scraper_request = ScraperRequest(**request_params)
                    
                    # Create task
                    task_id = str(uuid.uuid4())
                    task_status = TaskStatus(
                        task_id=task_id,
                        status="pending",
                        message="Task queued"
                    )
                    tasks_store[task_id] = task_status
                    
                    # Start background task
                    asyncio.create_task(process_scraping_task(task_id, scraper_request))
                    
                    # Store task ID in session
                    session.current_task_id = task_id
                    session.context["last_url"] = url
                    
                    response_message = (
                        f"I'm processing the images from {url}. This might take a moment.\n\n"
                        f"Your task ID is: `{task_id}`\n\n"
                        f"You can check the status by asking me 'What's the status of task {task_id}?'"
                    )
                    
                    suggestions = [
                        f"What's the status of task {task_id}?",
                        f"Download results from task {task_id}"
                    ]
                    
                    data = {"task_id": task_id}
                else:
                    # For smaller requests, process immediately
                    request_params = {
                        "url": url,
                        "formats": params.get("formats", [ImageFormat.ALL]),
                        "min_width": params.get("min_width"),
                        "min_height": params.get("min_height"),
                        "max_images": params.get("max_images", 10)  # Limit for direct results
                    }
                    
                    # Remove None values
                    request_params = {k: v for k, v in request_params.items() if v is not None}
                    
                    # Create scraper request
                    scraper_request = ScraperRequest(**request_params)
                    
                    # Process immediately
                    result = await scrape_images_from_url(scraper_request)
                    
                    # Generate CSV data
                    csv_data = await generate_csv_from_images(result.images)
                    
                    # Save CSV to a temporary file
                    task_id = str(uuid.uuid4())
                    csv_path = f"result_{task_id}.csv"
                    async with aiofiles.open(csv_path, 'wb') as f:
                        await f.write(csv_data)
                    
                    session.current_task_id = task_id
                    session.context["last_url"] = url
                    
                    # Create a downloadable task
                    task_status = TaskStatus(
                        task_id=task_id,
                        status="completed",
                        message="Task completed successfully",
                        result_url=f"/api/download/{task_id}",
                        progress=100.0
                    )
                    tasks_store[task_id] = task_status
                    
                    response_message = (
                        f"I found {result.total_images_found} images on {url}.\n\n"
                        f"Here's a summary of what I found:\n"
                        f"- Total images: {result.total_images_found}\n"
                        f"- Extracted images: {result.images_extracted}\n"
                        f"- Processing time: {result.processing_time_ms:.2f}ms\n\n"
                        f"You can download the CSV file with full details using this link: [Download CSV](/api/download/{task_id})\n\n"
                        f"Or ask me to 'show image preview' to see some of the images."
                    )
                    
                    suggestions = [
                        f"Download results from task {task_id}",
                        "Show image preview"
                    ]
                    
                    # Include some image previews in the data
                    preview_images = []
                    for img in result.images[:3]:  # Limit to 3 images for preview
                        preview_images.append({
                            "url": img.url,
                            "filename": img.filename,
                            "width": img.width,
                            "height": img.height,
                            "alt_text": img.alt_text
                        })
                    
                    data = {
                        "task_id": task_id,
                        "preview_images": preview_images,
                        "download_url": f"/api/download/{task_id}"
                    }
            except Exception as e:
                logger.error(f"Error processing scrape request: {str(e)}")
                response_message = f"Sorry, I encountered an error while trying to scrape images from {url}: {str(e)}"
                suggestions = [
                    "Try another website",
                    "Help me scrape images"
                ]
        else:
            response_message = "I need a URL to scrape images from. Could you please provide a valid website URL?"
            suggestions = [
                "Scrape images from https://example.com",
                "Help me scrape images"
            ]
    
    elif intent == Intent.GET_STATUS:
        task_id = params.get("task_id") or session.current_task_id
        
        if task_id and task_id in tasks_store:
            task = tasks_store[task_id]
            if task.status == "pending":
                response_message = f"Your task {task_id} is pending in the queue."
            elif task.status == "processing":
                response_message = f"Your task {task_id} is currently processing. Progress: {task.progress:.1f}%"
            elif task.status == "completed":
                response_message = (
                    f"Your task {task_id} has completed successfully!\n\n"
                    f"You can download the results here: [Download Results](/api/download/{task_id})"
                )
                suggestions = [
                    f"Download results from task {task_id}"
                ]
                data = {"download_url": f"/api/download/{task_id}"}
            else:
                response_message = f"Your task {task_id} has failed: {task.message}"
                suggestions = [
                    "Try scraping another website",
                    "Help me troubleshoot"
                ]
        else:
            if task_id:
                response_message = f"I couldn't find a task with ID {task_id}. It may have expired or been removed."
            else:
                response_message = "I don't have any active tasks for you. Would you like to scrape images from a website?"
            
            suggestions = [
                "Scrape images from https://example.com",
                "Help me scrape images"
            ]
    
    elif intent == Intent.DOWNLOAD_RESULTS:
        task_id = params.get("task_id") or session.current_task_id
        
        if task_id and task_id in tasks_store:
            task = tasks_store[task_id]
            if task.status == "completed":
                response_message = f"You can download your results here: [Download Results](/api/download/{task_id})"
                data = {"download_url": f"/api/download/{task_id}"}
            else:
                response_message = f"Your task {task_id} is not yet completed. Current status: {task.status}"
                suggestions = [
                    f"Check status of task {task_id}"
                ]
        else:
            if task_id:
                response_message = f"I couldn't find a task with ID {task_id}. It may have expired or been removed."
            else:
                response_message = "I don't have any completed tasks for you to download. Would you like to scrape images from a website first?"
            
            suggestions = [
                "Scrape images from https://example.com",
                "Help me scrape images"
            ]
    
    elif intent == Intent.HELP:
        response_message = (
            "# Image Scraper Chatbot Help\n\n"
            "I can help you extract images from websites. Here are some things you can ask me to do:\n\n"
            "- **Scrape images from a website**: Just give me a URL\n"
            "- **Filter images**: Specify minimum width/height or file formats\n"
            "- **Download images**: Request to download the images as a ZIP file\n"
            "- **Check task status**: Ask about the progress of your scraping tasks\n"
            "- **Download results**: Get links to download your CSV or ZIP files\n\n"
            "## Example commands\n\n"
            "- 'Scrape images from https://example.com'\n"
            "- 'Get all images from example.com with minimum width 300px'\n"
            "- 'Extract only JPG and PNG images from example.com'\n"
            "- 'What's the status of my task?'\n"
            "- 'Download results from my last task'\n\n"
            "What would you like to do?"
        )
        
        suggestions = [
            "Scrape images from https://example.com",
            "Extract JPG and PNG images from example.com",
            "Get images with minimum width 300px"
        ]
    
    else:
        response_message = (
            "I'm not sure what you're asking for. I'm designed to help you extract images from websites.\n\n"
            "Would you like to scrape images from a website? Just provide me with a URL, or type 'help' for more information."
        )
        
        suggestions = [
            "Help",
            "Scrape images from https://example.com",
            "Extract images from example.com"
        ]
    
    # Add message to session history
    session.messages.append(ChatMessage(role="user", content=message))
    session.messages.append(ChatMessage(role="assistant", content=response_message))
    session.updated_at = datetime.now()
    
    return ChatResponse(
        session_id=session_id,
        message=response_message,
        suggestions=suggestions,
        data=data
    )

# Create HTML template for the chat interface
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Scraper Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/marked/lib/marked.min.css">
    <style>
        body {
            background-color: #f0f2f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            height: calc(100vh - 80px);
            max-width: 1200px;
        }
        .chat-box {
            height: calc(100% - 180px); /* Adjusted for URL input and upload bar */
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #CBD5E0 #EDF2F7;
        }
        .chat-box::-webkit-scrollbar {
            width: 8px;
        }
        .chat-box::-webkit-scrollbar-track {
            background: #EDF2F7;
        }
        .chat-box::-webkit-scrollbar-thumb {
            background-color: #CBD5E0;
            border-radius: 20px;
            border: 3px solid #EDF2F7;
        }
        .user-message {
            background-color: #dcf8c6;
            border-radius: 15px 15px 0 15px;
            max-width: 75%;
            align-self: flex-end;
        }
        .bot-message {
            background-color: white;
            border-radius: 15px 15px 15px 0;
            max-width: 75%;
            align-self: flex-start;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            padding: 10px 0;
        }
        .suggestion-chip {
            display: inline-block;
            background: #e2e8f0;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            margin: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .suggestion-chip:hover {
            background: #cbd5e0;
        }
        .input-box {
            border-radius: 20px;
            border: 1px solid #E2E8F0;
            padding: 12px 20px;
            outline: none;
            flex-grow: 1;
            font-size: 16px;
        }
        .url-input-container {
            display: flex;
            padding: 10px 15px;
            background-color: #f8fafc;
            border-top: 1px solid #E2E8F0;
            align-items: center;
            transition: all 0.3s ease;
            max-height: 0;
            overflow: hidden;
            opacity: 0;
        }
        .url-input-container.active {
            max-height: 60px;
            opacity: 1;
            padding: 10px 15px;
        }
        .url-input-label {
            font-weight: 500;
            margin-right: 10px;
            white-space: nowrap;
            color: #4a5568;
        }
        .url-input {
            flex-grow: 1;
            padding: 8px 15px;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            font-size: 14px;
            outline: none;
        }
        .url-input:focus {
            border-color: #4299e1;
            box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.2);
        }
        .submit-url-btn {
            background-color: #4299e1;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 15px;
            margin-left: 10px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        .submit-url-btn:hover {
            background-color: #3182ce;
        }
        .upload-container {
            display: flex;
            padding: 10px 15px;
            background-color: #f8fafc;
            border-top: 1px solid #E2E8F0;
            align-items: center;
            transition: all 0.3s ease;
            max-height: 0;
            overflow: hidden;
            opacity: 0;
        }
        .upload-container.active {
            max-height: 60px;
            opacity: 1;
            padding: 10px 15px;
        }
        .upload-btn {
            background-color: #4299e1;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 15px;
            cursor: pointer;
            font-weight: 500;
            display: flex;
            align-items: center;
            transition: all 0.2s;
        }
        .upload-btn:hover {
            background-color: #3182ce;
        }
        .upload-btn i {
            margin-right: 8px;
        }
        .file-name {
            margin-left: 10px;
            font-size: 14px;
            color: #4a5568;
        }
        .options-container {
            display: flex;
            flex-wrap: wrap;
            padding: 15px;
            background-color: #f8fafc;
            border-top: 1px solid #E2E8F0;
            gap: 15px;
            transition: all 0.3s ease;
            max-height: 0;
            overflow: hidden;
            opacity: 0;
        }
        .options-container.active {
            max-height: 300px;
            opacity: 1;
        }
        .option-group {
            display: flex;
            flex-direction: column;
            min-width: 200px;
        }
        .option-label {
            font-weight: 500;
            margin-bottom: 5px;
            color: #4a5568;
            font-size: 14px;
        }
        .option-input {
            padding: 8px;
            border: 1px solid #E2E8F0;
            border-radius: 5px;
            font-size: 14px;
        }
        .option-checkbox {
            display: flex;
            align-items: center;
            margin-top: 5px;
        }
        .option-checkbox input {
            margin-right: 8px;
        }
        .format-options {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 5px;
        }
        .format-option {
            padding: 5px 10px;
            background-color: #e2e8f0;
            border-radius: 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .format-option.selected {
            background-color: #4299e1;
            color: white;
        }
        .format-option:hover:not(.selected) {
            background-color: #cbd5e0;
        }
        .send-button {
            background-color: #4299e1;
            color: white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .send-button:hover {
            background-color: #3182ce;
        }
        .send-button:disabled {
            background-color: #A0AEC0;
            cursor: not-allowed;
        }
        .typing-indicator {
            display: flex;
            padding: 15px 20px;
            border-radius: 15px 15px 15px 0;
            background-color: white;
            margin-bottom: 10px;
        }
        .typing-indicator span {
            height: 10px;
            width: 10px;
            float: left;
            margin: 0 1px;
            background-color: #9E9E9E;
            display: block;
            border-radius: 50%;
            opacity: 0.4;
        }
        .typing-indicator span:nth-of-type(1) {
            animation: typing 1s infinite;
        }
        .typing-indicator span:nth-of-type(2) {
            animation: typing 1s 250ms infinite;
        }
        .typing-indicator span:nth-of-type(3) {
            animation: typing 1s 500ms infinite;
        }
        @keyframes typing {
            0% { opacity: 0.4; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.2); }
            100% { opacity: 0.4; transform: scale(1); }
        }
        .markdown-content h1 {
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .markdown-content h2 {
            font-size: 1.25rem;
            font-weight: bold;
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
        }
        .markdown-content p {
            margin-bottom: 0.75rem;
        }
        .markdown-content ul, .markdown-content ol {
            margin-left: 1.5rem;
            margin-bottom: 0.75rem;
        }
        .markdown-content li {
            margin-bottom: 0.25rem;
            list-style: disc;
        }
        .markdown-content a {
            color: #4299e1;
            text-decoration: underline;
        }
        .markdown-content code {
            background-color: #EDF2F7;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: monospace;
        }
        .markdown-content pre {
            background-color: #2D3748;
            color: white;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            margin-bottom: 0.75rem;
        }
        .image-preview {
            display: flex;
            overflow-x: auto;
            gap: 10px;
            padding: 10px 0;
        }
        .image-preview img {
            max-height: 150px;
            max-width: 200px;
            object-fit: contain;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toggle-options {
            color: #4299e1;
            background: none;
            border: none;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 5px 10px;
            border-radius: 15px;
            transition: all 0.2s;
        }
        .toggle-options:hover {
            background-color: rgba(66, 153, 225, 0.1);
        }
        .toggle-options i {
            transition: transform 0.3s ease;
        }
        .toggle-options.active i {
            transform: rotate(180deg);
        }
        .tool-buttons {
            display: flex;
            gap: 10px;
            margin-right: 10px;
        }
        .tool-button {
            background: none;
            border: none;
            color: #4a5568;
            font-size: 1.2rem;
            cursor: pointer;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s;
        }
        .tool-button:hover {
            background-color: #e2e8f0;
            color: #4299e1;
        }
        .tool-button.active {
            color: #4299e1;
            background-color: rgba(66, 153, 225, 0.1);
        }
        /* Image Gallery */
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            padding: 10px 0;
        }
        .gallery-item {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .gallery-item:hover {
            transform: translateY(-5px);
        }
        .gallery-image {
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: cover;
        }
        .gallery-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 8px;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 12px;
            opacity: 0;
            transform: translateY(100%);
            transition: all 0.3s ease;
        }
        .gallery-item:hover .gallery-overlay {
            opacity: 1;
            transform: translateY(0);
        }
        .gallery-filename {
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .gallery-dimensions {
            font-size: 11px;
            opacity: 0.8;
        }
        #fileInput {
            display: none;
        }
        /* Progress bar */
        .progress-container {
            width: 100%;
            height: 5px;
            background-color: #e2e8f0;
            border-radius: 2.5px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar {
            height: 100%;
            background-color: #4299e1;
            width: 0%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="flex flex-col items-center justify-center w-full">
        <div class="chat-container flex flex-col w-full mt-4 px-4 sm:px-6 md:px-8">
            <div class="bg-white shadow-md rounded-lg overflow-hidden">
                <div class="bg-blue-500 text-white px-4 py-3 flex justify-between items-center">
                    <h1 class="text-xl font-semibold">Advanced Image Scraper</h1>
                    <div class="flex items-center">
                        <span id="connection-status" class="text-sm mr-2">Connecting...</span>
                        <div id="status-indicator" class="w-3 h-3 rounded-full bg-yellow-300"></div>
                    </div>
                </div>
                <div id="chat-box" class="chat-box p-4 bg-gray-50">
                    <div class="message-container">
                        <div class="bot-message p-3 shadow-sm">
                            <div class="markdown-content">
                                <h2>Welcome to the Advanced Image Scraper!</h2>
                                <p>I can help you extract and analyze images from any website. Please:</p>
                                <ol>
                                    <li><strong>Enter a website URL</strong> to scrape images</li>
                                    <li><strong>Upload an HTML file</strong> using the upload button</li>
                                    <li><strong>Provide specific requirements</strong> like minimum image size or format</li>
                                </ol>
                            </div>
                        </div>
                        <div class="flex flex-wrap mt-2">
                            <div class="suggestion-chip" onclick="showUrlInput()">
                                Enter a website URL
                            </div>
                            <div class="suggestion-chip" onclick="showFileUpload()">
                                Upload HTML file
                            </div>
                            <div class="suggestion-chip" onclick="showAdvancedOptions()">
                                Show advanced options
                            </div>
                        </div>
                    </div>
                </div>

                <!-- URL Input Bar -->
                <div id="url-input-container" class="url-input-container">
                    <span class="url-input-label">Website URL:</span>
                    <input type="url" id="url-input" class="url-input" placeholder="https://example.com" />
                    <button id="submit-url-btn" class="submit-url-btn">Scrape Images</button>
                </div>

                <!-- File Upload Bar -->
                <div id="upload-container" class="upload-container">
                    <input type="file" id="fileInput" accept=".html,.htm" />
                    <label for="fileInput" class="upload-btn">
                        <i class="fas fa-upload"></i>
                        Upload HTML File
                    </label>
                    <span id="fileName" class="file-name">No file selected</span>
                </div>

                <!-- Advanced Options Panel -->
                <div id="options-container" class="options-container">
                    <div class="option-group">
                        <label class="option-label">Minimum Width (px)</label>
                        <input type="number" id="min-width" class="option-input" placeholder="e.g. 300" min="1" />
                    </div>
                    <div class="option-group">
                        <label class="option-label">Minimum Height (px)</label>
                        <input type="number" id="min-height" class="option-input" placeholder="e.g. 200" min="1" />
                    </div>
                    <div class="option-group">
                        <label class="option-label">Max Images</label>
                        <input type="number" id="max-images" class="option-input" placeholder="e.g. 50" min="1" />
                    </div>
                    <div class="option-group">
                        <label class="option-label">Image Formats</label>
                        <div class="format-options">
                            <div class="format-option selected" data-format="all">All</div>
                            <div class="format-option" data-format="jpeg">JPEG</div>
                            <div class="format-option" data-format="png">PNG</div>
                            <div class="format-option" data-format="gif">GIF</div>
                            <div class="format-option" data-format="svg">SVG</div>
                            <div class="format-option" data-format="webp">WEBP</div>
                        </div>
                    </div>
                    <div class="option-group">
                        <label class="option-label">Scraping Depth</label>
                        <select id="scraping-depth" class="option-input">
                            <option value="basic">Basic (just the page)</option>
                            <option value="medium">Medium (page + one level)</option>
                            <option value="deep">Deep (page + multiple levels)</option>
                        </select>
                    </div>
                    <div class="option-group">
                        <div class="option-checkbox">
                            <input type="checkbox" id="follow-links" />
                            <label for="follow-links">Follow links on page</label>
                        </div>
                        <div class="option-checkbox">
                            <input type="checkbox" id="download-images" />
                            <label for="download-images">Download all images (ZIP)</label>
                        </div>
                    </div>
                </div>

                <div id="typing-indicator" class="typing-indicator ml-4 hidden">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <div class="p-4 bg-white flex items-center">
                    <div class="tool-buttons">
                        <button id="url-button" class="tool-button" title="Enter URL" onclick="toggleUrlInput()">
                            <i class="fas fa-globe"></i>
                        </button>
                        <button id="upload-button" class="tool-button" title="Upload HTML" onclick="toggleFileUpload()">
                            <i class="fas fa-upload"></i>
                        </button>
                        <button id="options-button" class="tool-button" title="Advanced Options" onclick="toggleAdvancedOptions()">
                            <i class="fas fa-cog"></i>
                        </button>
                    </div>
                    <input id="message-input" class="input-box" type="text" placeholder="Type your message or enter a URL..." autocomplete="off">
                    <button id="send-button" class="send-button" disabled>
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        let sessionId = null;
        let websocket = null;
        let selectedFormats = ["all"];
        
        const chatBox = document.getElementById('chat-box');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const typingIndicator = document.getElementById('typing-indicator');
        const connectionStatus = document.getElementById('connection-status');
        const statusIndicator = document.getElementById('status-indicator');
        const urlInputContainer = document.getElementById('url-input-container');
        const urlInput = document.getElementById('url-input');
        const submitUrlBtn = document.getElementById('submit-url-btn');
        const uploadContainer = document.getElementById('upload-container');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const optionsContainer = document.getElementById('options-container');
        const urlButton = document.getElementById('url-button');
        const uploadButton = document.getElementById('upload-button');
        const optionsButton = document.getElementById('options-button');
        
        // Initialize connection
        initializeWebSocket();
        
        function initializeWebSocket() {
            // Get current protocol
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function(event) {
                connectionStatus.textContent = 'Connected';
                statusIndicator.classList.remove('bg-yellow-300');
                statusIndicator.classList.remove('bg-red-500');
                statusIndicator.classList.add('bg-green-500');
                sendButton.disabled = false;
            };
            
            websocket.onmessage = function(event) {
                typingIndicator.classList.add('hidden');
                const response = JSON.parse(event.data);
                
                if (response.session_id) {
                    sessionId = response.session_id;
                }
                
                if (response.message) {
                    appendBotMessage(response);
                }
                
                // Check for UI directives
                if (response.show_url_input) {
                    showUrlInput();
                }
                
                if (response.show_upload) {
                    showFileUpload();
                }
            };
            
            websocket.onclose = function(event) {
                connectionStatus.textContent = 'Disconnected';
                statusIndicator.classList.remove('bg-green-500');
                statusIndicator.classList.remove('bg-yellow-300');
                statusIndicator.classList.add('bg-red-500');
                sendButton.disabled = true;
                
                // Try to reconnect after 3 seconds
                setTimeout(initializeWebSocket, 3000);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                connectionStatus.textContent = 'Error';
                statusIndicator.classList.remove('bg-green-500');
                statusIndicator.classList.remove('bg-yellow-300');
                statusIndicator.classList.add('bg-red-500');
            };
        }
        
        function sendMessage() {
            const message = messageInput.value.trim();
            if (message) {
                appendUserMessage(message);
                messageInput.value = '';
                
                // Show typing indicator
                typingIndicator.classList.remove('hidden');
                
                // Scroll to bottom
                scrollToBottom();
                
                // Send via WebSocket
                websocket.send(JSON.stringify({
                    message: message,
                    session_id: sessionId
                }));
                
                // Disable send button until response received
                sendButton.disabled = true;
            }
        }
        
        function sendSuggestion(text) {
            messageInput.value = text;
            sendMessage();
        }
        
        function appendUserMessage(message) {
            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container items-end';
            
            const messageElement = document.createElement('div');
            messageElement.className = 'user-message p-3 shadow-sm';
            messageElement.textContent = message;
            
            messageContainer.appendChild(messageElement);
            chatBox.appendChild(messageContainer);
            
            scrollToBottom();
        }
        
        function appendBotMessage(response) {
            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container';
            
            const messageElement = document.createElement('div');
            messageElement.className = 'bot-message p-3 shadow-sm';
            
            const contentElement = document.createElement('div');
            contentElement.className = 'markdown-content';
            contentElement.innerHTML = marked.parse(response.message);
            
            messageElement.appendChild(contentElement);
            messageContainer.appendChild(messageElement);
            
            // Add image previews if available
            if (response.data && response.data.preview_images && response.data.preview_images.length > 0) {
                const previewContainer = document.createElement('div');
                previewContainer.className = 'image-preview';
                
                response.data.preview_images.forEach(img => {
                    const imgElement = document.createElement('img');
                    imgElement.src = img.url;
                    imgElement.alt = img.alt_text || img.filename;
                    imgElement.title = img.filename;
                    
                    previewContainer.appendChild(imgElement);
                });
                
                messageContainer.appendChild(previewContainer);
            }
            
            // Add image gallery if available
            if (response.data && response.data.images && response.data.images.length > 0) {
                const galleryContainer = document.createElement('div');
                galleryContainer.className = 'image-gallery';
                
                response.data.images.forEach(img => {
                    const galleryItem = document.createElement('div');
                    galleryItem.className = 'gallery-item';
                    
                    const imgElement = document.createElement('img');
                    imgElement.className = 'gallery-image';
                    imgElement.src = img.url;
                    imgElement.alt = img.alt_text || img.filename;
                    
                    const overlay = document.createElement('div');
                    overlay.className = 'gallery-overlay';
                    
                    const filename = document.createElement('div');
                    filename.className = 'gallery-filename';
                    filename.textContent = img.filename;
                    
                    const dimensions = document.createElement('div');
                    dimensions.className = 'gallery-dimensions';
                    if (img.width && img.height) {
                        dimensions.textContent = `${img.width} × ${img.height}`;
                    }
                    
                    overlay.appendChild(filename);
                    overlay.appendChild(dimensions);
                    galleryItem.appendChild(imgElement);
                    galleryItem.appendChild(overlay);
                    galleryContainer.appendChild(galleryItem);
                });
                
                messageContainer.appendChild(galleryContainer);
            }
            
            // Add task progress if available
            if (response.data && response.data.task_id && response.data.progress !== undefined) {
                const progressContainer = document.createElement('div');
                progressContainer.className = 'progress-container';
                
                const progressBar = document.createElement('div');
                progressBar.className = 'progress-bar';
                progressBar.style.width = `${response.data.progress}%`;
                
                progressContainer.appendChild(progressBar);
                messageContainer.appendChild(progressContainer);
            }
            
            // Add suggestions if available
            if (response.suggestions && response.suggestions.length > 0) {
                const suggestionsContainer = document.createElement('div');
                suggestionsContainer.className = 'flex flex-wrap mt-2';
                
                response.suggestions.forEach(suggestion => {
                    const chip = document.createElement('div');
                    chip.className = 'suggestion-chip';
                    chip.textContent = suggestion;
                    chip.onclick = function() {
                        sendSuggestion(suggestion);
                    };
                    
                    suggestionsContainer.appendChild(chip);
                });
                
                messageContainer.appendChild(suggestionsContainer);
            }
            
            chatBox.appendChild(messageContainer);
            scrollToBottom();
            
            // Re-enable send button
            sendButton.disabled = false;
        }
        
        function scrollToBottom() {
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        // URL input functions
        function showUrlInput() {
            hideAllPanels();
            urlInputContainer.classList.add('active');
            urlButton.classList.add('active');
            setTimeout(() => urlInput.focus(), 300);
        }
        
        function toggleUrlInput() {
            if (urlInputContainer.classList.contains('active')) {
                urlInputContainer.classList.remove('active');
                urlButton.classList.remove('active');
            } else {
                showUrlInput();
            }
        }
        
        // File upload functions
        function showFileUpload() {
            hideAllPanels();
            uploadContainer.classList.add('active');
            uploadButton.classList.add('active');
        }
        
        function toggleFileUpload() {
            if (uploadContainer.classList.contains('active')) {
                uploadContainer.classList.remove('active');
                uploadButton.classList.remove('active');
            } else {
                showFileUpload();
            }
        }
        
        // Advanced options functions
        function showAdvancedOptions() {
            hideAllPanels();
            optionsContainer.classList.add('active');
            optionsButton.classList.add('active');
        }
        
        function toggleAdvancedOptions() {
            if (optionsContainer.classList.contains('active')) {
                optionsContainer.classList.remove('active');
                optionsButton.classList.remove('active');
            } else {
                showAdvancedOptions();
            }
        }
        
        function hideAllPanels() {
            urlInputContainer.classList.remove('active');
            uploadContainer.classList.remove('active');
            optionsContainer.classList.remove('active');
            urlButton.classList.remove('active');
            uploadButton.classList.remove('active');
            optionsButton.classList.remove('active');
        }
        
        // Format selection handling
        document.querySelectorAll('.format-option').forEach(option => {
            option.addEventListener('click', function() {
                const format = this.getAttribute('data-format');
                
                if (format === 'all') {
                    // If "All" is selected, deselect everything else
                    document.querySelectorAll('.format-option').forEach(opt => {
                        if (opt.getAttribute('data-format') === 'all') {
                            opt.classList.add('selected');
                            selectedFormats = ['all'];
                        } else {
                            opt.classList.remove('selected');
                        }
                    });
                } else {
                    // If a specific format is selected
                    const allOption = document.querySelector('.format-option[data-format="all"]');
                    allOption.classList.remove('selected');
                    
                    // Toggle the selection state of the clicked format
                    if (this.classList.contains('selected')) {
                        this.classList.remove('selected');
                        selectedFormats = selectedFormats.filter(f => f !== format);
                        
                        // If no formats are selected, select "All" again
                        if (selectedFormats.length === 0 || (selectedFormats.length === 1 && selectedFormats[0] === 'all')) {
                            allOption.classList.add('selected');
                            selectedFormats = ['all'];
                        }
                    } else {
                        this.classList.add('selected');
                        // Remove 'all' if it's in the array
                        selectedFormats = selectedFormats.filter(f => f !== 'all');
                        selectedFormats.push(format);
                    }
                }
            });
        });
        
        // Submit URL button
        submitUrlBtn.addEventListener('click', function() {
            const url = urlInput.value.trim();
            if (url) {
                // Collect all options
                const options = {
                    url: url,
                    formats: selectedFormats,
                };
                
                // Add optional parameters if they're set
                const minWidth = document.getElementById('min-width').value.trim();
                if (minWidth) options.min_width = parseInt(minWidth);
                
                const minHeight = document.getElementById('min-height').value.trim();
                if (minHeight) options.min_height = parseInt(minHeight);
                
                const maxImages = document.getElementById('max-images').value.trim();
                if (maxImages) options.max_images = parseInt(maxImages);
                
                const depth = document.getElementById('scraping-depth').value;
                if (depth !== 'basic') options.depth = depth;
                
                const followLinks = document.getElementById('follow-links').checked;
                if (followLinks) options.follow_links = true;
                
                const downloadImages = document.getElementById('download-images').checked;
                if (downloadImages) options.download_images = true;
                
                // Build message from options
                let message = `Scrape images from ${url}`;
                
                // Add format details if not just "all"
                if (!(selectedFormats.length === 1 && selectedFormats[0] === 'all')) {
                    message += ` with formats ${selectedFormats.join(', ')}`;
                }
                
                // Add dimension restrictions
                if (minWidth) message += `, minimum width ${minWidth}px`;
                if (minHeight) message += `, minimum height ${minHeight}px`;
                
                // Add other options
                if (maxImages) message += `, limit to ${maxImages} images`;
                if (depth !== 'basic') message += `, with ${depth} scraping depth`;
                if (followLinks) message += `, follow links`;
                if (downloadImages) message += `, download all images`;
                
                // Send the message
                messageInput.value = message;
                sendMessage();
                
                // Hide the URL input panel
                urlInputContainer.classList.remove('active');
                urlButton.classList.remove('active');
            }
        });
        
        // Handle file upload
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const file = this.files[0];
                fileName.textContent = file.name;
                
                const formData = new FormData();
                formData.append('html_file', file);
                
                // Add parameters from advanced options
                const minWidth = document.getElementById('min-width').value.trim();
                if (minWidth) formData.append('min_width', minWidth);
                
                const minHeight = document.getElementById('min-height').value.trim();
                if (minHeight) formData.append('min_height', minHeight);
                
                const maxImages = document.getElementById('max-images').value.trim();
                if (maxImages) formData.append('max_images', maxImages);
                
                // Add format parameters
                if (!(selectedFormats.length === 1 && selectedFormats[0] === 'all')) {
                    selectedFormats.forEach(format => {
                        formData.append('formats', format);
                    });
                }
                
                // Show user message
                appendUserMessage(`Uploading HTML file: ${file.name}`);
                
                // Show typing indicator
                typingIndicator.classList.remove('hidden');
                
                // Upload the file
                fetch('/api/upload/html', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    // Hide typing indicator
                    typingIndicator.classList.add('hidden');
                    
                    // Show response
                    appendBotMessage({
                        session_id: sessionId,
                        message: `I found ${data.total_images_found} images in the uploaded file "${file.name}".\n\n` +
                                 `Here's a summary of what I found:\n` +
                                 `- Total images: ${data.total_images_found}\n` +
                                 `- Images extracted: ${data.images_extracted}\n` +
                                 `- Processing time: ${data.processing_time_ms.toFixed(2)}ms\n\n` +
                                 `You can download the CSV file with all image details.`,
                        suggestions: [
                            "Show me a preview of the images",
                            "Download the results as CSV"
                        ],
                        data: {
                            preview_images: data.images.slice(0, 3).map(img => ({
                                url: img.url,
                                filename: img.filename,
                                width: img.width,
                                height: img.height,
                                alt_text: img.alt_text
                            })),
                            images: data.images.slice(0, 10)
                        }
                    });
                    
                    // Hide upload panel
                    uploadContainer.classList.remove('active');
                    uploadButton.classList.remove('active');
                })
                .catch(error => {
                    // Hide typing indicator
                    typingIndicator.classList.add('hidden');
                    
                    // Show error
                    appendBotMessage({
                        session_id: sessionId,
                        message: `Sorry, there was an error processing your file: ${error.message}`,
                        suggestions: [
                            "Try another file",
                            "Try scraping a website instead"
                        ]
                    });
                });
            }
        });
        
        // Event listeners
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        messageInput.addEventListener('input', function() {
            sendButton.disabled = messageInput.value.trim() === '';
        });
        
        sendButton.addEventListener('click', sendMessage);
        
        // URL input Enter key handling
        urlInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                submitUrlBtn.click();
            }
        });
        
        // Initial scroll to bottom
        scrollToBottom();
    </script>
</body>
</html>
"""

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

# Create connection manager instance
manager = ConnectionManager()

# Create or access a chat session
def get_or_create_session(session_id: Optional[str] = None) -> ChatSession:
    if session_id and session_id in chat_sessions:
        return chat_sessions[session_id]
    
    # Create new session
    new_session_id = session_id or str(uuid.uuid4())
    session = ChatSession(session_id=new_session_id)
    chat_sessions[new_session_id] = session
    return session

# Main chat endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the chatbot and get a response
    """
    session = get_or_create_session(request.session_id)
    response = await generate_chat_response(session, request.message)
    return response

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    try:
        # Create a session for this connection
        session = get_or_create_session()
        
        # Send session ID to client with a clear prompt for URL
        await websocket.send_json({
            "session_id": session.session_id,
            "message": "## Welcome to the Advanced Image Scraper\n\nI can help you extract and analyze images from any website. Please:\n\n1. **Enter a website URL** to scrape images\n2. **Upload an HTML file** using the upload button\n3. **Provide specific requirements** like minimum image size or format\n\nWhat would you like to do today?",
            "suggestions": [
                "Enter a website URL",
                "I want to upload an HTML file",
                "Show me advanced options"
            ],
            "show_url_input": True,  # Signal to show a dedicated URL input field
            "show_upload": True      # Signal to show a file upload button
        })
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                payload = json.loads(data)
                
                # Get or create session
                if "session_id" in payload and payload["session_id"]:
                    session = get_or_create_session(payload["session_id"])
                
                # Process message
                if "message" in payload:
                    # Generate response
                    response = await generate_chat_response(session, payload["message"])
                    
                    # Send response
                    await websocket.send_json(response.dict())
            except json.JSONDecodeError:
                await websocket.send_json({
                    "session_id": session.session_id,
                    "message": "Invalid message format. Please send a JSON object with a 'message' field."
                })
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

# Root endpoint - serve chat interface
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """
    Serve the chat interface
    """
    return CHAT_HTML

async def scrape_images_from_url(request: ScraperRequest) -> ScraperResponse:
    """
    Main function to scrape images from a URL
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Check cache if not downloading images
    if not request.download_images:
        cache_key = generate_cache_key(request)
        if cache_key in cache and time.time() - cache[cache_key]["timestamp"] < CACHE_TTL:
            logger.info(f"Cache hit for {request.url}")
            return ScraperResponse(
                request_id=request_id,
                url=str(request.url),
                total_images_found=cache[cache_key]["total_images"],
                images_extracted=len(cache[cache_key]["images"]),
                processing_time_ms=cache[cache_key]["processing_time_ms"],
                cache_hit=True,
                images=cache[cache_key]["images"]
            )
    
    logger.info(f"Scraping images from {request.url}")
    
    # Setup HTTP client with timeout and follow redirects
    async with httpx.AsyncClient(follow_redirects=True, timeout=DEFAULT_TIMEOUT) as client:
        try:
            # Fetch the main page
            response = await client.get(str(request.url))
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs
            image_urls = set()
            
            # Standard img tags
            if request.custom_css_selector:
                # Custom selector provided
                for img in soup.select(request.custom_css_selector):
                    if img.get('src'):
                        image_urls.add(img['src'])
            else:
                # Default behavior - find all img tags
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
            
            # Make image URLs absolute
            absolute_image_urls = []
            for img_url in image_urls:
                # Skip data URLs
                if img_url.startswith('data:'):
                    if 'data:image/' in img_url and request.include_base64:
                        absolute_image_urls.append(img_url)
                    continue
                
                # Skip URLs that match exclude patterns
                if any(re.search(pattern, img_url) for pattern in request.exclude_patterns):
                    continue
                
                # Make URL absolute
                try:
                    absolute_url = str(response.url.join(httpx.URL(img_url)))
                    absolute_image_urls.append(absolute_url)
                except Exception as e:
                    logger.warning(f"Error processing URL {img_url}: {str(e)}")
            
            # Follow links if requested (for medium/deep scraping)
            if request.follow_links and request.depth != ScrapingDepth.BASIC:
                # Extract links
                page_links = set()
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    # Skip anchors and javascript links
                    if not href.startswith('#') and not href.startswith('javascript:'):
                        try:
                            absolute_link = str(response.url.join(httpx.URL(href)))
                            # Only follow links on the same domain
                            if urlparse(absolute_link).netloc == urlparse(str(request.url)).netloc:
                                page_links.add(absolute_link)
                        except Exception as e:
                            logger.warning(f"Error processing link URL {href}: {str(e)}")
                
                # Limit the number of links to follow
                max_links = 5 if request.depth == ScrapingDepth.MEDIUM else 10
                follow_links = list(page_links)[:max_links]
                
                # Create tasks for each link
                link_tasks = []
                for link in follow_links:
                    logger.info(f"Following link: {link}")
                    link_tasks.append(client.get(link))
                
                # Execute tasks concurrently
                link_responses = await asyncio.gather(*link_tasks, return_exceptions=True)
                
                # Process responses
                for i, link_resp in enumerate(link_responses):
                    if isinstance(link_resp, Exception):
                        logger.warning(f"Error fetching link {follow_links[i]}: {str(link_resp)}")
                        continue
                    
                    try:
                        # Parse HTML
                        link_soup = BeautifulSoup(link_resp.text, 'html.parser')
                        
                        # Extract image URLs
                        for img in link_soup.find_all('img'):
                            if img.get('src'):
                                # Skip URLs that match exclude patterns
                                if any(re.search(pattern, img['src']) for pattern in request.exclude_patterns):
                                    continue
                                
                                # Make URL absolute
                                try:
                                    absolute_url = str(link_resp.url.join(httpx.URL(img['src'])))
                                    absolute_image_urls.append(absolute_url)
                                except Exception as e:
                                    logger.warning(f"Error processing URL {img['src']}: {str(e)}")
                    except Exception as e:
                        logger.warning(f"Error processing link response: {str(e)}")
            
            # Remove duplicates
            absolute_image_urls = list(set(absolute_image_urls))
            
            # Apply format filtering
            if not ImageFormat.ALL in request.formats:
                absolute_image_urls = [url for url in absolute_image_urls if is_valid_image_url(url, request.formats)]
            
            # Limit number of images if requested
            max_images = min(request.max_images or len(absolute_image_urls), len(absolute_image_urls))
            total_images_found = len(absolute_image_urls)
            absolute_image_urls = absolute_image_urls[:max_images]
            
            # Create tasks for fetching image metadata
            metadata_tasks = []
            for img_url in absolute_image_urls:
                metadata_tasks.append(fetch_image_metadata(
                    client, 
                    img_url, 
                    str(request.url), 
                    request.include_base64
                ))
            
            # Process images concurrently
            image_metadata_results = await asyncio.gather(*metadata_tasks)
            
            # Filter out failed metadata fetches
            image_metadata = [meta for meta in image_metadata_results if meta is not None]
            
            # Apply size filtering
            if request.min_width or request.min_height:
                filtered_metadata = []
                for meta in image_metadata:
                    if ((request.min_width is None or (meta.width is not None and meta.width >= request.min_width)) and
                        (request.min_height is None or (meta.height is not None and meta.height >= request.min_height))):
                        filtered_metadata.append(meta)
                image_metadata = filtered_metadata
            
            # Apply sorting
            if image_metadata:
                if request.sort_by == SortOrder.SIZE_DESC:
                    image_metadata.sort(key=lambda x: x.size_bytes or 0, reverse=True)
                elif request.sort_by == SortOrder.SIZE_ASC:
                    image_metadata.sort(key=lambda x: x.size_bytes or 0)
                elif request.sort_by == SortOrder.NAME_ASC:
                    image_metadata.sort(key=lambda x: x.filename)
                elif request.sort_by == SortOrder.NAME_DESC:
                    image_metadata.sort(key=lambda x: x.filename, reverse=True)
                elif request.sort_by == SortOrder.DATE_ASC:
                    image_metadata.sort(key=lambda x: x.extraction_time)
                elif request.sort_by == SortOrder.DATE_DESC:
                    image_metadata.sort(key=lambda x: x.extraction_time, reverse=True)
            
            # Download images if requested
            if request.download_images and image_metadata:
                # Create a directory for downloaded images
                download_dir = f"downloads_{request_id}"
                os.makedirs(download_dir, exist_ok=True)
                
                # Download each image
                for i, meta in enumerate(image_metadata):
                    try:
                        # Skip data URLs
                        if meta.url.startswith('data:'):
                            continue
                        
                        # Get image content
                        img_response = await client.get(meta.url)
                        if img_response.status_code == 200:
                            # Determine file extension
                            ext = meta.format or get_file_extension(meta.url) or "jpg"
                            if not ext.startswith('.'):
                                ext = f".{ext}"
                            
                            # Sanitize filename
                            safe_filename = re.sub(r'[^\w\-\.]', '_', meta.filename)
                            if not safe_filename:
                                safe_filename = f"image_{i}{ext}"
                            elif not safe_filename.endswith(ext):
                                safe_filename = f"{safe_filename}{ext}"
                            
                            # Save the file
                            file_path = os.path.join(download_dir, safe_filename)
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(img_response.content)
                            
                            # Update metadata with local path
                            meta.local_path = file_path
                    except Exception as e:
                        logger.warning(f"Error downloading image {meta.url}: {str(e)}")
            
            # Calculate processing time
            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Store in cache
            if not request.download_images:
                cache_key = generate_cache_key(request)
                cache[cache_key] = {
                    "timestamp": time.time(),
                    "total_images": total_images_found,
                    "images": image_metadata,
                    "processing_time_ms": processing_time_ms
                }
            
            # Return response
            return ScraperResponse(
                request_id=request_id,
                url=str(request.url),
                total_images_found=total_images_found,
                images_extracted=len(image_metadata),
                processing_time_ms=processing_time_ms,
                images=image_metadata
            )
        
        except Exception as e:
            logger.error(f"Error scraping {request.url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error scraping images: {str(e)}")

async def generate_csv_from_images(images: List[ImageMetadata]) -> bytes:
    """
    Generate a CSV file from image metadata
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        "URL", "Filename", "Width", "Height", "Size (bytes)", 
        "Format", "Aspect Ratio", "Alt Text", "Title", "Source Page"
    ])
    
    # Write data rows
    for img in images:
        writer.writerow([
            img.url,
            img.filename,
            img.width or "",
            img.height or "",
            img.size_bytes or "",
            img.format or "",
            img.aspect_ratio or "",
            img.alt_text or "",
            img.title or "",
            img.source_page or ""
        ])
    
    return output.getvalue().encode('utf-8')

async def process_scraping_task(task_id: str, request: ScraperRequest):
    """
    Process a scraping task in the background
    """
    try:
        # Update task status
        task_status = tasks_store[task_id]
        task_status.status = "processing"
        task_status.message = "Processing started"
        task_status.progress = 10.0
        task_status.updated_at = datetime.now()
        
        # Perform scraping
        result = await scrape_images_from_url(request)
        
        # Update progress
        task_status.progress = 50.0
        task_status.message = "Scraping completed, generating CSV"
        task_status.updated_at = datetime.now()
        
        # Generate CSV
        csv_data = await generate_csv_from_images(result.images)
        
        # Save CSV to a file
        csv_path = f"result_{task_id}.csv"
        async with aiofiles.open(csv_path, 'wb') as f:
            await f.write(csv_data)
        
        # Create ZIP file if images were downloaded
        zip_path = None
        if request.download_images:
            download_dir = f"downloads_{result.request_id}"
            if os.path.exists(download_dir):
                zip_path = f"images_{task_id}.zip"
                # Create zip file
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(download_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, os.path.basename(file_path))
        
        # Complete task
        task_status.status = "completed"
        task_status.progress = 100.0
        task_status.message = "Task completed successfully"
        task_status.result_url = f"/api/download/{task_id}"
        task_status.updated_at = datetime.now()
        
        logger.info(f"Task {task_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        
        # Update task status
        if task_id in tasks_store:
            task_status = tasks_store[task_id]
            task_status.status = "failed"
            task_status.message = f"Task failed: {str(e)}"
            task_status.updated_at = datetime.now()

# Add additional API endpoints for downloading results and task status

@app.get("/api/download/{task_id}")
async def download_results(task_id: str):
    """
    Download results for a task
    """
    if task_id not in tasks_store or tasks_store[task_id].status != "completed":
        raise HTTPException(status_code=404, detail="Task not found or not completed")
    
    # Check for CSV file
    csv_path = f"result_{task_id}.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Check for ZIP file
    zip_path = f"images_{task_id}.zip"
    if os.path.exists(zip_path):
        # Return ZIP file
        return StreamingResponse(
            open(zip_path, "rb"),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=images_{task_id}.zip"}
        )
    
    # Return CSV file
    return StreamingResponse(
        open(csv_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=images_{task_id}.csv"}
    )

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status of a task
    """
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_store[task_id]

@app.post("/api/upload/html")
async def upload_html(html_file: UploadFile = File(...)):
    """
    Process HTML file for image extraction
    """
    # Create a temporary file
    temp_file_path = f"temp_{uuid.uuid4()}.html"
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as temp_file:
            content = await html_file.read()
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
        
        # Create image metadata
        image_metadata = []
        for img_url in image_urls:
            # Skip data URLs or process them if include_base64 is True
            if img_url.startswith('data:'):
                if 'data:image/' in img_url:
                    try:
                        # Extract format from data URL
                        format_match = re.search(r'data:image/([^;]+);base64', img_url)
                        image_format = format_match.group(1) if format_match else "unknown"
                        
                        # Create metadata
                        metadata = ImageMetadata(
                            url=img_url[:100] + "...",  # Truncate for display
                            filename=f"data_image_{len(image_metadata)}.{image_format}",
                            format=image_format,
                            source_page="Uploaded HTML file"
                        )
                        image_metadata.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error processing data URL {img_url}: {str(e)}")
                continue
            
            # Handle relative URLs - just store as is since we don't have a base URL
            filename = os.path.basename(img_url)
            extension = get_file_extension(img_url)
            
            metadata = ImageMetadata(
                url=img_url,
                filename=filename or f"image_{len(image_metadata)}.{extension or 'jpg'}",
                format=extension,
                source_page="Uploaded HTML file"
            )
            image_metadata.append(metadata)
        
        # Generate task ID for result retrieval
        task_id = str(uuid.uuid4())
        
        # Generate CSV
        csv_data = await generate_csv_from_images(image_metadata)
        
        # Save CSV to a file
        csv_path = f"result_{task_id}.csv"
        async with aiofiles.open(csv_path, 'wb') as f:
            await f.write(csv_data)
        
        # Create a task status
        task_status = TaskStatus(
            task_id=task_id,
            status="completed",
            message="HTML file processed successfully",
            result_url=f"/api/download/{task_id}",
            progress=100.0
        )
        tasks_store[task_id] = task_status
        
        # Return response
        start_time = time.time()
        return ScraperResponse(
            request_id=task_id,
            url="Uploaded HTML file",
            total_images_found=len(image_metadata),
            images_extracted=len(image_metadata),
            processing_time_ms=(time.time() - start_time) * 1000,
            images=image_metadata
        )
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

import zipfile

# Cleanup scheduled task
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Image Scraper API...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the Image Scraper API...")
    # Clean up any temporary files
    for filename in os.listdir('.'):
        if filename.startswith(('temp_', 'downloads_', 'result_')):
            try:
                if os.path.isdir(filename):
                    shutil.rmtree(filename)
                else:
                    os.remove(filename)
            except Exception as e:
                logger.error(f"Error cleaning up {filename}: {str(e)}")

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("image_csv:app", host="0.0.0.0", port=8000, reload=True)
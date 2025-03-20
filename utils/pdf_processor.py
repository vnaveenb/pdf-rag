import os
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Tuple, Any, Optional
import re
import io
import base64
from PIL import Image
import numpy as np
import pytesseract

class PDFProcessor:
    """A class for extracting text from PDFs with layout awareness and image/table handling."""
    
    def __init__(self, verbose: bool = False, enable_ocr: bool = True, detect_tables: bool = True):
        """Initialize the PDFProcessor.
        
        Args:
            verbose: Whether to print additional information during processing.
            enable_ocr: Whether to extract text from images using OCR.
            detect_tables: Whether to detect and process tables.
        """
        self.verbose = verbose
        self.enable_ocr = enable_ocr
        self.detect_tables = detect_tables
    
    def extract_text_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from a PDF using pdfplumber with page metadata.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            A list of dictionaries containing page text and metadata.
        """
        extracted_data = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                filename = os.path.basename(file_path)
                
                if self.verbose:
                    print(f"Processing {filename} with {total_pages} pages")
                
                for i, page in enumerate(pdf.pages):
                    page_data = {
                        'filename': filename,
                        'page_num': i + 1,
                        'total_pages': total_pages,
                        'text': '',
                        'tables': [],
                        'images': [],
                        'layout_elements': []
                    }
                    
                    # Extract tables first so we can avoid extracting text from table areas
                    if self.detect_tables:
                        tables = page.find_tables()
                        for t_idx, table in enumerate(tables):
                            try:
                                table_data = table.extract()
                                if table_data:
                                    # Convert table data to string representation
                                    table_str = self._table_to_markdown(table_data)
                                    # Add table to the tables list
                                    page_data['tables'].append({
                                        'table_idx': t_idx,
                                        'rows': len(table_data),
                                        'cols': len(table_data[0]) if table_data else 0,
                                        'content': table_str,
                                        'bbox': table.bbox  # Save bounding box for layout
                                    })
                                    # Also store tables as special layout elements
                                    page_data['layout_elements'].append({
                                        'type': 'table',
                                        'content': table_str,
                                        'bbox': table.bbox
                                    })
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error extracting table on page {i+1}: {e}")
                    
                    # Extract regular text
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    
                    if page_text:
                        # Clean the text
                        page_text = self._clean_text(page_text)
                        page_data['text'] = page_text
                        # Add text as a layout element
                        page_data['layout_elements'].append({
                            'type': 'text',
                            'content': page_text,
                            'bbox': (0, 0, page.width, page.height)  # Full page
                        })
                    else:
                        if self.verbose:
                            print(f"No text extracted from page {i+1} using pdfplumber")
                    
                    # Extract images
                    if self.enable_ocr:
                        try:
                            for img in page.images:
                                if img:
                                    # Store image metadata
                                    img_data = {
                                        'img_idx': len(page_data['images']),
                                        'width': img['width'],
                                        'height': img['height'],
                                        'bbox': (img['x0'], img['top'], img['x1'], img['bottom']),
                                        'text': ''
                                    }
                                    
                                    # Apply OCR to the image if enabled
                                    try:
                                        # Convert image data to PIL Image for OCR
                                        image = Image.open(io.BytesIO(img['stream'].get_data()))
                                        img_text = pytesseract.image_to_string(image)
                                        if img_text:
                                            img_data['text'] = self._clean_text(img_text)
                                            # Add image text as a layout element
                                            page_data['layout_elements'].append({
                                                'type': 'image_text',
                                                'content': img_data['text'],
                                                'bbox': img_data['bbox']
                                            })
                                    except Exception as img_err:
                                        if self.verbose:
                                            print(f"OCR error on page {i+1}: {img_err}")
                                    
                                    page_data['images'].append(img_data)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error extracting images on page {i+1}: {e}")
                    
                    extracted_data.append(page_data)
            
            return extracted_data
        
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {e}")
            return []
    
    def extract_text_with_pymupdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from a PDF using PyMuPDF with page metadata.
        
        Better for handling complex layouts like multiple columns.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            A list of dictionaries containing page text and metadata.
        """
        extracted_data = []
        
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            filename = os.path.basename(file_path)
            
            if self.verbose:
                print(f"Processing {filename} with {total_pages} pages using PyMuPDF")
            
            for i in range(total_pages):
                page = doc[i]
                
                page_data = {
                    'filename': filename,
                    'page_num': i + 1,
                    'total_pages': total_pages,
                    'text': '',
                    'tables': [],
                    'images': [],
                    'layout_elements': []
                }
                
                # Extract text with layout recognition (blocks)
                blocks = page.get_text("dict", sort=True)
                
                # Process each block based on type
                for block in blocks.get("blocks", []):
                    block_type = block.get("type", 0)
                    
                    # Text block (type 0)
                    if block_type == 0:
                        block_text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                block_text += span.get("text", "")
                            block_text += "\n"
                        
                        if block_text:
                            # Add as layout element with bounding box
                            bbox = block.get("bbox", (0, 0, 0, 0))
                            page_data['layout_elements'].append({
                                'type': 'text_block',
                                'content': self._clean_text(block_text),
                                'bbox': bbox,
                                'font_size': self._get_avg_font_size(block),
                                'is_bold': self._is_bold(block)
                            })
                    
                    # Image block (type 1)
                    elif block_type == 1 and self.enable_ocr:
                        try:
                            # Get image data
                            xref = block.get("xref", 0)
                            bbox = block.get("bbox", (0, 0, 0, 0))
                            
                            if xref > 0:
                                img_data = {
                                    'img_idx': len(page_data['images']),
                                    'width': bbox[2] - bbox[0],
                                    'height': bbox[3] - bbox[1],
                                    'bbox': bbox,
                                    'text': ''
                                }
                                
                                # Try to extract the image and perform OCR
                                try:
                                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=bbox)
                                    img_bytes = pix.tobytes(output="png")
                                    image = Image.open(io.BytesIO(img_bytes))
                                    img_text = pytesseract.image_to_string(image)
                                    
                                    if img_text:
                                        img_data['text'] = self._clean_text(img_text)
                                        # Add as layout element
                                        page_data['layout_elements'].append({
                                            'type': 'image_text',
                                            'content': img_data['text'],
                                            'bbox': bbox
                                        })
                                except Exception as ocr_err:
                                    if self.verbose:
                                        print(f"OCR error on page {i+1}: {ocr_err}")
                                
                                page_data['images'].append(img_data)
                        except Exception as img_err:
                            if self.verbose:
                                print(f"Error processing image on page {i+1}: {img_err}")
                
                # Attempt to detect tables using heuristics
                if self.detect_tables:
                    tables = self._detect_tables_heuristic(blocks)
                    for t_idx, table in enumerate(tables):
                        page_data['tables'].append({
                            'table_idx': t_idx,
                            'rows': len(table['data']),
                            'cols': len(table['data'][0]) if table['data'] else 0,
                            'content': self._table_to_markdown(table['data']),
                            'bbox': table['bbox']
                        })
                        # Add as layout element
                        page_data['layout_elements'].append({
                            'type': 'table',
                            'content': self._table_to_markdown(table['data']),
                            'bbox': table['bbox']
                        })
                
                # Combine all text from layout elements for the main text field
                all_text = []
                for elem in page_data['layout_elements']:
                    if elem['type'] in ['text', 'text_block']:
                        all_text.append(elem['content'])
                
                page_data['text'] = "\n\n".join(all_text)
                
                # Only add pages with content
                if page_data['text'] or page_data['tables'] or page_data['images']:
                    extracted_data.append(page_data)
                else:
                    if self.verbose:
                        print(f"No content extracted from page {i+1} using PyMuPDF")
            
            doc.close()
            return extracted_data
        
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {e}")
            return []
    
    def _detect_tables_heuristic(self, blocks_dict):
        """Detect tables in a page using heuristics based on text alignment.
        
        Args:
            blocks_dict: Dictionary of blocks from PyMuPDF.
            
        Returns:
            List of detected tables with their data and bounding boxes.
        """
        tables = []
        # Implementation of table detection heuristics
        # This is a placeholder - real implementation would analyze text blocks
        # for grid-like structure, alignment, and spacing patterns
        
        # For now, we'll return an empty list
        return tables
    
    def _get_avg_font_size(self, block):
        """Calculate average font size for a text block.
        
        Args:
            block: Text block dictionary from PyMuPDF.
            
        Returns:
            Average font size or 0 if not available.
        """
        sizes = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                if size > 0:
                    sizes.append(size)
        
        return sum(sizes) / len(sizes) if sizes else 0
    
    def _is_bold(self, block):
        """Determine if a text block is bold.
        
        Args:
            block: Text block dictionary from PyMuPDF.
            
        Returns:
            Boolean indicating if the text is bold.
        """
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                flags = span.get("flags", 0)
                # Check bold flag in font flags
                if flags & 2:  # 2 is the bold flag
                    return True
        return False
    
    def _table_to_markdown(self, table_data):
        """Convert table data to markdown format.
        
        Args:
            table_data: 2D list of table cell values.
            
        Returns:
            Markdown string representation of the table.
        """
        if not table_data or len(table_data) == 0:
            return ""
        
        result = []
        
        # Add header row
        header_row = " | ".join(str(cell) if cell else "" for cell in table_data[0])
        result.append("| " + header_row + " |")
        
        # Add separator
        separator = " | ".join(["---"] * len(table_data[0]))
        result.append("| " + separator + " |")
        
        # Add data rows
        for row in table_data[1:]:
            row_str = " | ".join(str(cell) if cell else "" for cell in row)
            result.append("| " + row_str + " |")
        
        return "\n".join(result)
    
    def extract_text(self, file_path: str, prefer_method: str = 'auto') -> List[Dict[str, Any]]:
        """Extract text from a PDF with the preferred method.
        
        Args:
            file_path: Path to the PDF file.
            prefer_method: Extraction method ('pdfplumber', 'pymupdf', or 'auto').
            
        Returns:
            A list of dictionaries containing page text and metadata.
        """
        if prefer_method == 'pdfplumber':
            return self.extract_text_with_pdfplumber(file_path)
        elif prefer_method == 'pymupdf':
            return self.extract_text_with_pymupdf(file_path)
        else:  # 'auto' - try both and return the one with more content
            plumber_data = self.extract_text_with_pdfplumber(file_path)
            pymupdf_data = self.extract_text_with_pymupdf(file_path)
            
            # Calculate total content (text + tables + image text)
            def calc_content_size(data):
                total_size = 0
                for page in data:
                    total_size += len(page.get('text', ''))
                    for table in page.get('tables', []):
                        total_size += len(table.get('content', ''))
                    for img in page.get('images', []):
                        total_size += len(img.get('text', ''))
                return total_size
            
            plumber_size = calc_content_size(plumber_data)
            pymupdf_size = calc_content_size(pymupdf_data)
            
            if self.verbose:
                print(f"pdfplumber extracted {plumber_size} chars")
                print(f"PyMuPDF extracted {pymupdf_size} chars")
            
            # Return the method that extracted more content
            return plumber_data if plumber_size >= pymupdf_size else pymupdf_data
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text.
        
        Args:
            text: Text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphenated words across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix strange Unicode characters
        text = text.replace('\uf0b7', '•')  # Replace private use characters
        
        return text.strip()

    def process_pdfs(self, pdf_files: List[str], prefer_method: str = 'auto') -> List[Dict[str, Any]]:
        """Process multiple PDF files and extract text with metadata.
        
        Args:
            pdf_files: List of paths to PDF files.
            prefer_method: Extraction method ('pdfplumber', 'pymupdf', or 'auto').
            
        Returns:
            A list of dictionaries containing page text and metadata from all PDFs.
        """
        all_extracted_data = []
        
        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"File not found: {pdf_file}")
                continue
                
            extracted_data = self.extract_text(pdf_file, prefer_method)
            all_extracted_data.extend(extracted_data)
        
        return all_extracted_data 
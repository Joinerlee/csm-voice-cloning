"""
Utility module for detecting and handling filler words in text.
"""

import re
from typing import List, Dict, Set, Tuple

# Common filler words in English
ENGLISH_FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "I mean", 
    "actually", "basically", "literally", "sort of", "kind of",
    "hmm", "mmm", "right", "okay", "well"
}

def detect_fillers(text: str, language: str = "en") -> List[Tuple[str, int, int]]:
    """
    Detect filler words in the given text.
    
    Args:
        text (str): The text to analyze
        language (str): Language code ('en' for English)
        
    Returns:
        List[Tuple[str, int, int]]: List of (filler_word, start_index, end_index)
    """
    fillers = []
    
    # Only process English filler words
    if language.lower() != "en":
        return []  # Return empty list for non-English text
    
    # Convert text to lowercase for case-insensitive matching
    processed_text = text.lower()
    
    # Find all occurrences of filler words
    for filler in ENGLISH_FILLER_WORDS:
        for match in re.finditer(r'\b' + re.escape(filler) + r'\b', processed_text):
            start_idx, end_idx = match.span()
            fillers.append((text[start_idx:end_idx], start_idx, end_idx))
    
    # Sort by position in text
    fillers.sort(key=lambda x: x[1])
    
    return fillers

def mark_fillers_for_cloning(text: str, language: str = "en") -> Tuple[str, List[Dict]]:
    """
    Marks text with filler words that should be cloned.
    
    Args:
        text (str): Original text
        language (str): Language code
        
    Returns:
        Tuple[str, List[Dict]]: Processed text and list of filler segments
    """
    fillers = detect_fillers(text, language)
    filler_segments = []
    
    # Create marked text and segment info
    for filler, start_idx, end_idx in fillers:
        filler_segments.append({
            "text": filler,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "should_clone": True  # Fillers should be cloned
        })
    
    return text, filler_segments

def split_text_with_fillers(text: str, language: str = "en") -> List[Dict]:
    """
    Splits text into segments, marking fillers for voice cloning.
    
    Args:
        text (str): Original text
        language (str): Language code
        
    Returns:
        List[Dict]: List of text segments with cloning info
    """
    fillers = detect_fillers(text, language)
    
    if not fillers:
        # If no fillers detected, return the whole text as one segment
        return [{"text": text, "should_clone": False}]
    
    # Split text into segments
    segments = []
    last_end = 0
    
    for filler, start_idx, end_idx in fillers:
        # Add segment before the filler if it exists
        if start_idx > last_end:
            non_filler_text = text[last_end:start_idx].strip()
            if non_filler_text:
                segments.append({
                    "text": non_filler_text,
                    "should_clone": False
                })
        
        # Add the filler segment
        segments.append({
            "text": filler,
            "should_clone": True
        })
        
        last_end = end_idx
    
    # Add the remaining text after the last filler
    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            segments.append({
                "text": remaining_text,
                "should_clone": False
            })
    
    return segments 
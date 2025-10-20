#!/usr/bin/env python3
"""
Configuration File Watcher
Automatically synchronizes configurations when config.py is modified
"""

import os
import sys
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_config_sync import AutoConfigSync

logger = logging.getLogger(__name__)

class ConfigFileHandler(FileSystemEventHandler):
    """Handles configuration file changes"""
    
    def __init__(self):
        self.auto_sync = AutoConfigSync()
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only handle config.py changes
        if file_path.name == 'config.py':
            # Avoid duplicate events
            current_time = time.time()
            if file_path in self.last_modified and current_time - self.last_modified[file_path] < 2:
                return
            
            self.last_modified[file_path] = current_time
            
            logger.info(f"🔍 config.py modified - checking for synchronization needs")
            
            try:
                # Run automatic synchronization
                success = self.auto_sync.auto_sync_configurations()
                
                if success:
                    logger.info("✅ Configuration automatically synchronized")
                else:
                    logger.warning("⚠️ Configuration synchronization failed")
                    
            except Exception as e:
                logger.error(f"❌ Error during auto-sync: {str(e)}")

def start_watcher():
    """Start the configuration file watcher"""
    print("👀 Starting Configuration File Watcher")
    print("=" * 50)
    print("Watching for changes to config.py...")
    print("Press Ctrl+C to stop")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    event_handler = ConfigFileHandler()
    
    # Create observer
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    
    # Start watching
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Configuration watcher stopped")
    
    observer.join()

if __name__ == "__main__":
    start_watcher()

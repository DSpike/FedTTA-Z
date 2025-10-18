#!/usr/bin/env python3
"""
Real IPFS Client Integration
Connects to actual IPFS daemon instead of using mock implementation
"""

import requests
import json
import time
import logging
from typing import Dict, Optional, Any
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealIPFSClient:
    """
    Real IPFS client that connects to actual IPFS daemon
    """
    
    def __init__(self, ipfs_url: str = "http://localhost:5001"):
        self.ipfs_url = ipfs_url
        self.api_base = f"{ipfs_url}/api/v0"
        self.gateway_base = f"{ipfs_url.replace('5001', '8080')}/ipfs"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to IPFS daemon"""
        try:
            response = requests.post(f"{self.api_base}/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"✅ Connected to IPFS daemon version: {version_info.get('Version', 'Unknown')}")
                return True
            else:
                logger.warning(f"⚠️ IPFS daemon responded with status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to connect to IPFS daemon: {e}")
            logger.info("💡 Make sure IPFS daemon is running: ipfs daemon")
            return False
    
    def add_data(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Add data to IPFS and return CID
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data, default=str)
            
            # Prepare multipart form data
            files = {
                'file': (None, json_data, 'application/json')
            }
            
            # Add to IPFS
            response = requests.post(
                f"{self.api_base}/add",
                files=files,
                params={'pin': 'true'},  # Pin the data
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                cid = result.get('Hash')
                if cid:
                    logger.info(f"✅ Data added to IPFS with CID: {cid}")
                    return cid
                else:
                    logger.error("❌ No CID returned from IPFS")
                    return None
            else:
                logger.error(f"❌ IPFS add failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to add data to IPFS: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error adding data to IPFS: {e}")
            return None
    
    def get_data(self, cid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from IPFS using CID
        """
        try:
            # Get data from IPFS
            response = requests.post(
                f"{self.api_base}/cat",
                params={'arg': cid},
                timeout=30
            )
            
            if response.status_code == 200:
                # Parse JSON data
                data = json.loads(response.text)
                logger.info(f"✅ Data retrieved from IPFS with CID: {cid}")
                return data
            else:
                logger.error(f"❌ IPFS get failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get data from IPFS: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON data from IPFS: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error getting data from IPFS: {e}")
            return None
    
    def pin_data(self, cid: str) -> bool:
        """
        Pin data to ensure it's not garbage collected
        """
        try:
            response = requests.post(
                f"{self.api_base}/pin/add",
                params={'arg': cid},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Data pinned with CID: {cid}")
                return True
            else:
                logger.warning(f"⚠️ Failed to pin data with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to pin data: {e}")
            return False
    
    def get_pin_list(self) -> list:
        """
        Get list of pinned CIDs
        """
        try:
            response = requests.post(f"{self.api_base}/pin/ls", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                pins = result.get('Keys', {})
                cid_list = list(pins.keys())
                logger.info(f"📌 Found {len(cid_list)} pinned items")
                return cid_list
            else:
                logger.error(f"❌ Failed to get pin list: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get pin list: {e}")
            return []
    
    def get_node_info(self) -> Optional[Dict[str, Any]]:
        """
        Get IPFS node information
        """
        try:
            response = requests.post(f"{self.api_base}/id", timeout=10)
            
            if response.status_code == 200:
                node_info = response.json()
                logger.info(f"🆔 IPFS Node ID: {node_info.get('ID', 'Unknown')}")
                return node_info
            else:
                logger.error(f"❌ Failed to get node info: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get node info: {e}")
            return None

def test_real_ipfs_client():
    """Test the real IPFS client"""
    print("🌐 Testing Real IPFS Client...")
    
    # Create real IPFS client
    ipfs_client = RealIPFSClient()
    
    # Test adding data
    test_data = {
        'test': 'real_ipfs_data',
        'timestamp': time.time(),
        'encrypted_parameters': 'test_encrypted_data_here',
        'client_id': 'test_client',
        'encryption_method': 'fernet'
    }
    
    print("\n1. Adding data to IPFS...")
    cid = ipfs_client.add_data(test_data)
    
    if cid:
        print(f"   ✅ Data added with CID: {cid}")
        
        # Test retrieving data
        print("\n2. Retrieving data from IPFS...")
        retrieved_data = ipfs_client.get_data(cid)
        
        if retrieved_data:
            print(f"   ✅ Data retrieved successfully")
            print(f"   - Client ID: {retrieved_data.get('client_id')}")
            print(f"   - Timestamp: {retrieved_data.get('timestamp')}")
            print(f"   - Encryption Method: {retrieved_data.get('encryption_method')}")
            
            # Test pinning
            print("\n3. Pinning data...")
            pin_success = ipfs_client.pin_data(cid)
            if pin_success:
                print("   ✅ Data pinned successfully")
            else:
                print("   ⚠️ Failed to pin data")
            
            # Get node info
            print("\n4. Getting node info...")
            node_info = ipfs_client.get_node_info()
            if node_info:
                print(f"   ✅ Node ID: {node_info.get('ID', 'Unknown')[:20]}...")
            
            # Get pin list
            print("\n5. Getting pin list...")
            pins = ipfs_client.get_pin_list()
            print(f"   📌 Total pinned items: {len(pins)}")
            
        else:
            print("   ❌ Failed to retrieve data")
    else:
        print("   ❌ Failed to add data to IPFS")
        print("   💡 Make sure IPFS daemon is running: ipfs daemon")

if __name__ == "__main__":
    test_real_ipfs_client()








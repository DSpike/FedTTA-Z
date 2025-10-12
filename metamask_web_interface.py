#!/usr/bin/env python3
"""
Simple MetaMask Web Interface for Blockchain Federated Learning
Provides a web interface for MetaMask authentication
"""

from flask import Flask, render_template_string, request, jsonify
import json
import logging
from blockchain.metamask_auth_system import MetaMaskAuthenticator, DecentralizedIdentityManager
from blockchain.blockchain_ipfs_integration import FEDERATED_LEARNING_ABI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# MetaMask configuration
METAMASK_CONFIG = {
    'rpc_url': 'http://localhost:8545',
    'contract_address': '0x74f2D28CEC2c97186dE1A02C1Bae84D19A7E8BC8',
    'network_id': '1337'  # Ganache default
}

# Global authenticator instance
authenticator = None
identity_manager = None

def initialize_metamask():
    """Initialize MetaMask authenticator"""
    global authenticator, identity_manager
    try:
        authenticator = MetaMaskAuthenticator(
            rpc_url=METAMASK_CONFIG['rpc_url'],
            contract_address=METAMASK_CONFIG['contract_address'],
            contract_abi=FEDERATED_LEARNING_ABI
        )
        identity_manager = DecentralizedIdentityManager(authenticator)
        logger.info("✅ MetaMask authenticator initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize MetaMask: {e}")
        return False

# HTML template for MetaMask interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blockchain Federated Learning - MetaMask Authentication</title>
    <script src="https://cdn.jsdelivr.net/npm/web3@latest/dist/web3.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        .wallet-info { background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .hidden { display: none; }
        .loading { color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Blockchain Federated Learning</h1>
        <h2>MetaMask Authentication</h2>
        
        <div id="status" class="status info">
            <strong>Status:</strong> <span id="status-text">Initializing...</span>
        </div>
        
        <div id="metamask-section">
            <h3>Connect MetaMask</h3>
            <button id="connect-btn" onclick="connectMetaMask()">Connect MetaMask</button>
            <button id="disconnect-btn" onclick="disconnectMetaMask()" class="hidden">Disconnect</button>
            
            <div id="wallet-info" class="wallet-info hidden">
                <h4>Connected Wallet</h4>
                <p><strong>Address:</strong> <span id="wallet-address"></span></p>
                <p><strong>Network:</strong> <span id="network-info"></span></p>
                <p><strong>Balance:</strong> <span id="wallet-balance"></span> ETH</p>
            </div>
        </div>
        
        <div id="auth-section" class="hidden">
            <h3>Authenticate for Federated Learning</h3>
            <button id="auth-btn" onclick="authenticateWallet()">Authenticate Wallet</button>
            
            <div id="auth-result" class="hidden">
                <h4>Authentication Result</h4>
                <div id="auth-details"></div>
            </div>
        </div>
        
        <div id="federated-learning-section" class="hidden">
            <h3>🎯 Federated Learning Ready</h3>
            <p>Your wallet is authenticated and ready for blockchain federated learning!</p>
            <button onclick="startFederatedLearning()">Start Federated Learning</button>
        </div>
    </div>

    <script>
        let web3;
        let account;
        let isConnected = false;

        // Initialize Web3
        async function initWeb3() {
            if (typeof window.ethereum !== 'undefined') {
                web3 = new Web3(window.ethereum);
                updateStatus('MetaMask detected', 'success');
                
                // Check if already connected
                const accounts = await web3.eth.getAccounts();
                if (accounts.length > 0) {
                    account = accounts[0];
                    await updateWalletInfo();
                    showAuthSection();
                }
            } else {
                updateStatus('MetaMask not detected. Please install MetaMask extension.', 'error');
                document.getElementById('connect-btn').disabled = true;
            }
        }

        // Connect to MetaMask
        async function connectMetaMask() {
            try {
                updateStatus('Connecting to MetaMask...', 'info');
                const accounts = await window.ethereum.request({ 
                    method: 'eth_requestAccounts' 
                });
                
                if (accounts.length > 0) {
                    account = accounts[0];
                    isConnected = true;
                    await updateWalletInfo();
                    showAuthSection();
                    updateStatus('Connected to MetaMask successfully!', 'success');
                }
            } catch (error) {
                updateStatus('Failed to connect to MetaMask: ' + error.message, 'error');
            }
        }

        // Disconnect MetaMask
        function disconnectMetaMask() {
            account = null;
            isConnected = false;
            hideWalletInfo();
            hideAuthSection();
            hideFederatedLearningSection();
            updateStatus('Disconnected from MetaMask', 'info');
        }

        // Update wallet information
        async function updateWalletInfo() {
            if (!account) return;

            try {
                const balance = await web3.eth.getBalance(account);
                const networkId = await web3.eth.net.getId();
                
                document.getElementById('wallet-address').textContent = account;
                document.getElementById('network-info').textContent = `Network ID: ${networkId}`;
                document.getElementById('wallet-balance').textContent = web3.utils.fromWei(balance, 'ether');
                
                document.getElementById('wallet-info').classList.remove('hidden');
                document.getElementById('connect-btn').classList.add('hidden');
                document.getElementById('disconnect-btn').classList.remove('hidden');
            } catch (error) {
                console.error('Error updating wallet info:', error);
            }
        }

        // Authenticate wallet
        async function authenticateWallet() {
            if (!account) {
                updateStatus('Please connect MetaMask first', 'error');
                return;
            }

            try {
                updateStatus('Authenticating wallet...', 'info');
                
                // First, get the challenge from the backend
                const challengeResponse = await fetch('/get-challenge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        wallet_address: account
                    })
                });
                
                const challengeData = await challengeResponse.json();
                
                if (!challengeData.success) {
                    updateStatus('Failed to get challenge: ' + challengeData.error, 'error');
                    return;
                }
                
                const challenge = challengeData.challenge;
                
                // Sign the challenge message with MetaMask
                const signature = await window.ethereum.request({
                    method: 'personal_sign',
                    params: [challenge, account]
                });
                
                // Send to backend for authentication
                const response = await fetch('/authenticate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        wallet_address: account,
                        signature: signature,
                        message: challenge
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('Authentication successful!', 'success');
                    showAuthResult(result);
                    showFederatedLearningSection();
                } else {
                    updateStatus('Authentication failed: ' + result.error, 'error');
                }
            } catch (error) {
                updateStatus('Authentication error: ' + error.message, 'error');
            }
        }

        // Start federated learning
        function startFederatedLearning() {
            updateStatus('Redirecting to federated learning system...', 'info');
            // In a real implementation, this would redirect to the main system
            setTimeout(() => {
                window.location.href = '/federated-learning';
            }, 2000);
        }

        // Update status display
        function updateStatus(message, type) {
            const statusEl = document.getElementById('status');
            const statusTextEl = document.getElementById('status-text');
            
            statusEl.className = `status ${type}`;
            statusTextEl.textContent = message;
        }

        // Show/hide sections
        function showAuthSection() {
            document.getElementById('auth-section').classList.remove('hidden');
        }

        function hideAuthSection() {
            document.getElementById('auth-section').classList.add('hidden');
        }

        function showWalletInfo() {
            document.getElementById('wallet-info').classList.remove('hidden');
        }

        function hideWalletInfo() {
            document.getElementById('wallet-info').classList.add('hidden');
        }

        function showAuthResult(result) {
            const authResultEl = document.getElementById('auth-result');
            const authDetailsEl = document.getElementById('auth-details');
            
            authDetailsEl.innerHTML = `
                <p><strong>Identity Hash:</strong> ${result.identity?.identity_hash || 'N/A'}</p>
                <p><strong>Reputation Score:</strong> ${result.identity?.reputation_score || 0}</p>
                <p><strong>Session Token:</strong> ${result.session_token ? 'Generated' : 'N/A'}</p>
            `;
            
            authResultEl.classList.remove('hidden');
        }

        function showFederatedLearningSection() {
            document.getElementById('federated-learning-section').classList.remove('hidden');
        }

        function hideFederatedLearningSection() {
            document.getElementById('federated-learning-section').classList.add('hidden');
        }

        // Initialize when page loads
        window.addEventListener('load', initWeb3);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main MetaMask interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/get-challenge', methods=['POST'])
def get_challenge():
    """Get authentication challenge for wallet"""
    try:
        data = request.get_json()
        wallet_address = data.get('wallet_address')
        
        if not wallet_address:
            return jsonify({'success': False, 'error': 'Wallet address required'})
        
        # Generate challenge
        challenge = authenticator.generate_challenge(wallet_address)
        
        return jsonify({
            'success': True,
            'challenge': challenge,
            'wallet_address': wallet_address
        })
        
    except Exception as e:
        logger.error(f"Challenge generation error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handle wallet authentication"""
    try:
        data = request.get_json()
        wallet_address = data.get('wallet_address')
        signature = data.get('signature')
        message = data.get('message')
        
        if not all([wallet_address, signature, message]):
            return jsonify({'success': False, 'error': 'Missing authentication data'})
        
        # Authenticate with MetaMask (challenge should already be generated)
        auth_result = identity_manager.authenticate_participant(wallet_address, signature)
        
        if auth_result.success:
            return jsonify({
                'success': True,
                'identity': {
                    'wallet_address': auth_result.identity.wallet_address,
                    'identity_hash': auth_result.identity.identity_hash,
                    'reputation_score': auth_result.identity.reputation_score,
                    'verified': auth_result.identity.verified
                },
                'session_token': auth_result.session_token,
                'timestamp': auth_result.timestamp
            })
        else:
            return jsonify({
                'success': False,
                'error': auth_result.error_message
            })
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/federated-learning')
def federated_learning():
    """Federated learning interface (placeholder)"""
    return jsonify({
        'message': 'Federated learning system would start here',
        'status': 'ready',
        'blockchain_enabled': True
    })

@app.route('/status')
def status():
    """System status endpoint"""
    return jsonify({
        'metamask_available': authenticator is not None,
        'blockchain_connected': authenticator.web3.is_connected() if authenticator else False,
        'network_id': authenticator.web3.eth.chain_id if authenticator else None,
        'contract_address': METAMASK_CONFIG['contract_address']
    })

if __name__ == '__main__':
    print("🚀 Starting MetaMask Web Interface...")
    
    # Initialize MetaMask
    if initialize_metamask():
        print("✅ MetaMask authenticator ready")
        print("🌐 Starting web server on http://localhost:5000")
        print("📱 Open your browser and navigate to http://localhost:5000")
        print("🔗 Make sure MetaMask is installed and Ganache is running")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize MetaMask. Please check your blockchain setup.")
        print("💡 Make sure Ganache is running on http://localhost:8545")



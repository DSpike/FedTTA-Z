import json
import re

# Load the results
with open('edgeiiot_results_MITM.json', 'r') as f:
    results = json.load(f)

if results.get('training_history'):
    round_data = results['training_history'][0]
    client_updates = round_data['client_updates']
    
    print('Testing regex patterns:')
    client_update = client_updates[0]
    
    # Test the exact patterns used in the code
    loss_match = re.search(r'training_loss=([0-9.]+)', client_update)
    accuracy_match = re.search(r'validation_accuracy=([0-9.]+)', client_update)
    client_id_match = re.search(r"client_id='([^']+)'", client_update)
    blockchain_match = re.search(r'blockchain_tx_hash=([^,)]+)', client_update)
    
    print('Loss match:', loss_match.group(1) if loss_match else 'None')
    print('Accuracy match:', accuracy_match.group(1) if accuracy_match else 'None')
    print('Client ID match:', client_id_match.group(1) if client_id_match else 'None')
    print('Blockchain match:', blockchain_match.group(1) if blockchain_match else 'None')
    
    print('\nTesting all clients:')
    for i, client_update in enumerate(client_updates):
        loss_match = re.search(r'training_loss=([0-9.]+)', client_update)
        accuracy_match = re.search(r'validation_accuracy=([0-9.]+)', client_update)
        blockchain_match = re.search(r'blockchain_tx_hash=([^,)]+)', client_update)
        
        print(f'Client {i+1}:')
        print(f'  Loss: {loss_match.group(1) if loss_match else "None"}')
        print(f'  Accuracy: {accuracy_match.group(1) if accuracy_match else "None"}')
        print(f'  Blockchain: {blockchain_match.group(1) if blockchain_match else "None"}')

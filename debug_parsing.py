import json
import re

# Load the results
with open('edgeiiot_results_MITM.json', 'r') as f:
    results = json.load(f)

if results.get('training_history'):
    round_data = results['training_history'][0]
    client_updates = round_data['client_updates']
    
    print('=== EXTRACTING SPECIFIC VALUES ===')
    
    # Extract specific values
    client_id_match = re.search(r'client_id=([^,)]+)', client_updates[0])
    training_loss_match = re.search(r'training_loss=([0-9.]+)', client_updates[0])
    validation_accuracy_match = re.search(r'validation_accuracy=([0-9.]+)', client_updates[0])
    blockchain_tx_match = re.search(r'blockchain_tx_hash=([^,)]+)', client_updates[0])
    
    print('Client ID:', client_id_match.group(1) if client_id_match else 'Not found')
    print('Training Loss:', training_loss_match.group(1) if training_loss_match else 'Not found')
    print('Validation Accuracy:', validation_accuracy_match.group(1) if validation_accuracy_match else 'Not found')
    print('Blockchain TX Hash:', blockchain_tx_match.group(1) if blockchain_tx_match else 'Not found')
    
    print('\n=== TESTING ALL CLIENT UPDATES ===')
    for i, client_update in enumerate(client_updates):
        print(f'Client {i+1}:')
        client_id_match = re.search(r'client_id=([^,)]+)', client_update)
        training_loss_match = re.search(r'training_loss=([0-9.]+)', client_update)
        validation_accuracy_match = re.search(r'validation_accuracy=([0-9.]+)', client_update)
        blockchain_tx_match = re.search(r'blockchain_tx_hash=([^,)]+)', client_update)
        
        print(f'  Client ID: {client_id_match.group(1) if client_id_match else "Not found"}')
        print(f'  Training Loss: {training_loss_match.group(1) if training_loss_match else "Not found"}')
        print(f'  Validation Accuracy: {validation_accuracy_match.group(1) if validation_accuracy_match else "Not found"}')
        print(f'  Blockchain TX Hash: {blockchain_tx_match.group(1) if blockchain_tx_match else "Not found"}')
        print()

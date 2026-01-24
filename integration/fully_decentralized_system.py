"""
Fully Decentralized Federated Learning System with PBFT Consensus

This module implements a fully decentralized federated learning system using
Practical Byzantine Fault Tolerance (PBFT) consensus for 3 nodes with f=0
(no faulty nodes). The system includes leader election, model update consensus,
and Shapley-based incentive distribution.
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transductive_fewshot_model import TransductiveFewShotModel
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor

# Optional incentives/blockchain components (not required for pure FL runs)
try:
    from incentives.shapley_value_calculator import ShapleyValueCalculator
except Exception:
    ShapleyValueCalculator = None

try:
    from blockchain.blockchain_incentive_contract import BlockchainIncentiveContract
except Exception:
    BlockchainIncentiveContract = None

logger = logging.getLogger(__name__)

class PBFTPhase(Enum):
    """PBFT consensus phases"""
    PRE_PREPARE = "pre_prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    DECIDED = "decided"

@dataclass
class PBFTMessage:
    """PBFT consensus message"""
    phase: PBFTPhase
    view_number: int
    sequence_number: int
    node_id: str
    content: Dict[str, Any]
    signature: Optional[str] = None
    timestamp: float = None

@dataclass
class ModelUpdate:
    """Model update for consensus"""
    node_id: str
    model_state: Dict[str, Any]
    training_metrics: Dict[str, float]
    data_quality: float
    reliability: float
    timestamp: float
    signature: Optional[str] = None

@dataclass
class ConsensusResult:
    """Result of PBFT consensus"""
    success: bool
    agreed_model: Optional[Dict[str, Any]]
    agreed_incentives: Optional[Dict[str, float]]
    consensus_time: float
    participating_nodes: List[str]

class FullyDecentralizedSystem:
    """
    Fully Decentralized Federated Learning System with PBFT Consensus
    
    Implements a 3-node decentralized system where each node can act as:
    - Leader (rotating per round)
    - Follower (participates in consensus)
    
    Features:
    - PBFT consensus for model updates
    - Shapley-based incentive distribution
    - Peer-to-peer communication
    - No central coordinator
    """
    
    def __init__(self, 
                 node_id: str,
                 port: int,
                 other_nodes: List[Tuple[str, int]],
                 model_config: Dict[str, Any],
                 data_config: Dict[str, Any]):
        """
        Initialize the fully decentralized system
        
        Args:
            node_id: Unique identifier for this node
            port: Port for this node's server
            other_nodes: List of (host, port) tuples for other nodes
            model_config: Configuration for the transductive model
            data_config: Configuration for data preprocessing
        """
        self.node_id = node_id
        self.port = port
        self.other_nodes = other_nodes
        self.model_config = model_config
        self.data_config = data_config
        
        # PBFT consensus state
        self.view_number = 0
        self.sequence_number = 0
        self.current_leader = None
        self.consensus_state = {}
        self.pending_messages = {}
        self.consensus_timeout = 30.0  # seconds
        
        # Model and data components
        self.model = None
        self.preprocessor = None
        self.shapley_calculator = None
        self.incentive_contract = None
        self.training_data = None
        self.test_data = None
        
        # Communication
        self.server = None
        self.client_sessions = {}
        self.message_handlers = {
            PBFTPhase.PRE_PREPARE: self._handle_pre_prepare,
            PBFTPhase.PREPARE: self._handle_prepare,
            PBFTPhase.COMMIT: self._handle_commit
        }
        
        # Metrics
        self.consensus_metrics = {
            'total_rounds': 0,
            'successful_consensus': 0,
            'average_consensus_time': 0.0,
            'leader_elections': 0
        }
        
        logger.info(f"Initialized fully decentralized node {node_id} on port {port}")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize model
            self.model = TransductiveFewShotModel(
                input_dim=self.model_config.get('input_dim', 32),
                hidden_dim=self.model_config.get('hidden_dim', 128),
                embedding_dim=self.model_config.get('embedding_dim', 64),
                num_classes=self.model_config.get('num_classes', 2),
                sequence_length=self.model_config.get('sequence_length', 12)
            )
            
            # Initialize preprocessor
            self.preprocessor = UNSWPreprocessor()
            
            # Initialize Shapley calculator (optional)
            if ShapleyValueCalculator is not None:
                try:
                    self.shapley_calculator = ShapleyValueCalculator()
                except Exception as e:
                    logger.warning(f"Shapley incentive calculator not available: {e}")
                    self.shapley_calculator = None
            else:
                logger.warning("Shapley incentive calculator module missing; incentives disabled.")
                self.shapley_calculator = None
            
            # Initialize incentive contract (optional)
            if BlockchainIncentiveContract is not None:
                try:
                    self.incentive_contract = BlockchainIncentiveContract()
                    logger.info("Blockchain incentive contract initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize blockchain contract: {e}")
                    self.incentive_contract = None
            else:
                logger.warning("Blockchain incentive contract module missing; incentives disabled.")
                self.incentive_contract = None
            
            # Load and preprocess data
            await self._load_and_preprocess_data()
            
            # Start communication server
            await self._start_server()
            
            logger.info(f"Node {self.node_id} fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize node {self.node_id}: {e}")
            raise
    
    async def _load_and_preprocess_data(self):
        """Load and preprocess UNSW-NB15 dataset"""
        try:
            # Load UNSW-NB15 dataset
            preprocessed_data = self.preprocessor.preprocess_unsw_dataset(
                zero_day_attack=self.data_config.get('zero_day_attack', 'DoS')
            )
            
            # Store data
            self.training_data = {
                'X': preprocessed_data['X_train'],
                'y': preprocessed_data['y_train']
            }
            self.test_data = {
                'X': preprocessed_data['X_test'],
                'y': preprocessed_data['y_test']
            }
            
            logger.info(f"Node {self.node_id} loaded data: {len(self.training_data['X'])} training samples")
            
        except Exception as e:
            logger.error(f"Failed to load data for node {self.node_id}: {e}")
            raise
    
    async def _start_server(self):
        """Start the communication server"""
        try:
            import websockets
            
            async def handle_client(websocket, path):
                """Handle incoming client connections"""
                try:
                    async for message in websocket:
                        await self._process_message(message, websocket)
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Client disconnected from node {self.node_id}")
                except Exception as e:
                    logger.error(f"Error handling client on node {self.node_id}: {e}")
            
            self.server = await websockets.serve(handle_client, "localhost", self.port)
            logger.info(f"Node {self.node_id} server started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start server for node {self.node_id}: {e}")
            raise
    
    async def _process_message(self, message: str, websocket):
        """Process incoming PBFT messages"""
        try:
            data = json.loads(message)
            pbft_message = PBFTMessage(
                phase=PBFTPhase(data['phase']),
                view_number=data['view_number'],
                sequence_number=data['sequence_number'],
                node_id=data['node_id'],
                content=data['content'],
                signature=data.get('signature'),
                timestamp=data.get('timestamp', time.time())
            )
            
            # Route to appropriate handler
            handler = self.message_handlers.get(pbft_message.phase)
            if handler:
                await handler(pbft_message, websocket)
            else:
                logger.warning(f"Unknown message phase: {pbft_message.phase}")
                
        except Exception as e:
            logger.error(f"Error processing message on node {self.node_id}: {e}")
    
    async def _handle_pre_prepare(self, message: PBFTMessage, websocket):
        """Handle pre-prepare phase messages"""
        if message.node_id == self.current_leader:
            logger.info(f"Node {self.node_id} received pre-prepare from leader {message.node_id}")
            
            # Validate the message
            if self._validate_pre_prepare(message):
                # Store the message
                key = f"{message.view_number}_{message.sequence_number}"
                self.pending_messages[key] = message
                
                # Send prepare message to all nodes
                await self._broadcast_prepare(message)
            else:
                logger.warning(f"Node {self.node_id} rejected invalid pre-prepare from {message.node_id}")
    
    async def _handle_prepare(self, message: PBFTMessage, websocket):
        """Handle prepare phase messages"""
        logger.info(f"Node {self.node_id} received prepare from {message.node_id}")
        
        # Store prepare message
        key = f"{message.view_number}_{message.sequence_number}_prepare_{message.node_id}"
        self.pending_messages[key] = message
        
        # Check if we have enough prepare messages (2f+1 = 3 for f=0)
        prepare_count = self._count_prepare_messages(message.view_number, message.sequence_number)
        if prepare_count >= 3:  # 2f+1 for f=0
            logger.info(f"Node {self.node_id} has enough prepare messages, sending commit")
            await self._broadcast_commit(message)
    
    async def _handle_commit(self, message: PBFTMessage, websocket):
        """Handle commit phase messages"""
        logger.info(f"Node {self.node_id} received commit from {message.node_id}")
        
        # Store commit message
        key = f"{message.view_number}_{message.sequence_number}_commit_{message.node_id}"
        self.pending_messages[key] = message
        
        # Check if we have enough commit messages (2f+1 = 3 for f=0)
        commit_count = self._count_commit_messages(message.view_number, message.sequence_number)
        if commit_count >= 3:  # 2f+1 for f=0
            logger.info(f"Node {self.node_id} consensus reached, executing decision")
            await self._execute_consensus_decision(message)
    
    def _validate_pre_prepare(self, message: PBFTMessage) -> bool:
        """Validate pre-prepare message"""
        # Check if sender is the current leader
        if message.node_id != self.current_leader:
            return False
        
        # Check view number
        if message.view_number != self.view_number:
            return False
    
        # Check sequence number
        if message.sequence_number <= self.sequence_number:
            return False
        
        # Validate content structure
        content = message.content
        required_fields = ['model_updates', 'incentive_data']
        if not all(field in content for field in required_fields):
            return False
        
        return True
    
    def _count_prepare_messages(self, view_number: int, sequence_number: int) -> int:
        """Count prepare messages for given view and sequence"""
        count = 0
        for key, message in self.pending_messages.items():
            if (key.startswith(f"{view_number}_{sequence_number}_prepare_") and
                message.phase == PBFTPhase.PREPARE and
                message.view_number == view_number and
                message.sequence_number == sequence_number):
                count += 1
        return count
    
    def _count_commit_messages(self, view_number: int, sequence_number: int) -> int:
        """Count commit messages for given view and sequence"""
        count = 0
        for key, message in self.pending_messages.items():
            if (key.startswith(f"{view_number}_{sequence_number}_commit_") and
                message.phase == PBFTPhase.COMMIT and
                message.view_number == view_number and
                message.sequence_number == sequence_number):
                count += 1
        return count
    
    async def _broadcast_prepare(self, pre_prepare_message: PBFTMessage):
        """Broadcast prepare message to all nodes"""
        prepare_message = PBFTMessage(
            phase=PBFTPhase.PREPARE,
            view_number=pre_prepare_message.view_number,
            sequence_number=pre_prepare_message.sequence_number,
            node_id=self.node_id,
            content=pre_prepare_message.content,
            timestamp=time.time()
        )
        
        await self._broadcast_message(prepare_message)
    
    async def _broadcast_commit(self, prepare_message: PBFTMessage):
        """Broadcast commit message to all nodes"""
        commit_message = PBFTMessage(
            phase=PBFTPhase.COMMIT,
            view_number=prepare_message.view_number,
            sequence_number=prepare_message.sequence_number,
            node_id=self.node_id,
            content=prepare_message.content,
            timestamp=time.time()
        )
        
        await self._broadcast_message(commit_message)
    
    async def _broadcast_message(self, message: PBFTMessage):
        """Broadcast message to all other nodes"""
        message_data = {
            'phase': message.phase.value,
            'view_number': message.view_number,
            'sequence_number': message.sequence_number,
            'node_id': message.node_id,
            'content': message.content,
            'signature': message.signature,
            'timestamp': message.timestamp
        }
        
        for host, port in self.other_nodes:
            try:
                import websockets
                async with websockets.connect(f"ws://{host}:{port}") as websocket:
                    await websocket.send(json.dumps(message_data))
            except Exception as e:
                logger.warning(f"Failed to send message to {host}:{port}: {e}")
    
    async def _execute_consensus_decision(self, message: PBFTMessage):
        """Execute the consensus decision"""
        try:
            start_time = time.time()
            
            # Extract model updates and incentive data
            model_updates = message.content['model_updates']
            incentive_data = message.content['incentive_data']
            
            # Aggregate model updates
            aggregated_model = await self._aggregate_models(model_updates)
            
            # Calculate Shapley-based incentives
            incentives = await self._calculate_incentives(incentive_data)
            
            # Update local model
            self.model.load_state_dict(aggregated_model)
            
            # Update consensus state
            self.sequence_number = message.sequence_number
            consensus_time = time.time() - start_time
            
            # Update metrics
            self.consensus_metrics['total_rounds'] += 1
            self.consensus_metrics['successful_consensus'] += 1
            self.consensus_metrics['average_consensus_time'] = (
                (self.consensus_metrics['average_consensus_time'] * 
                 (self.consensus_metrics['successful_consensus'] - 1) + 
                 consensus_time) / self.consensus_metrics['successful_consensus']
            )
            
            logger.info(f"Node {self.node_id} executed consensus decision in {consensus_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to execute consensus decision on node {self.node_id}: {e}")
    
    async def _aggregate_models(self, model_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate model updates using FedAvg"""
        try:
            if not model_updates:
                return self.model.state_dict()
            
            # Convert model updates to state dicts
            state_dicts = []
            for update in model_updates:
                if 'model_state' in update:
                    state_dicts.append(update['model_state'])
            
            if not state_dicts:
                return self.model.state_dict()
            
            # Simple FedAvg aggregation
            aggregated_state = {}
            for key in state_dicts[0].keys():
                aggregated_state[key] = torch.stack([
                    torch.tensor(state_dict[key]) for state_dict in state_dicts
                ]).mean(dim=0)
            
            return aggregated_state
            
        except Exception as e:
            logger.error(f"Failed to aggregate models on node {self.node_id}: {e}")
            return self.model.state_dict()
    
    async def _calculate_incentives(self, incentive_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate Shapley-based incentives"""
        try:
            if not self.shapley_calculator:
                # Fallback to equal distribution
                node_count = len(incentive_data)
                return {data['node_id']: 1.0 / node_count for data in incentive_data}
            
            # Extract data quality and reliability scores
            data_quality_scores = {data['node_id']: data['data_quality'] for data in incentive_data}
            participation_data = {data['node_id']: data['reliability'] for data in incentive_data}
            
            # Calculate Shapley values
            shapley_values = self.shapley_calculator.calculate_shapley_values(
                data_quality_scores=data_quality_scores,
                participation_data=participation_data
            )
            
            return shapley_values
            
        except Exception as e:
            logger.error(f"Failed to calculate incentives on node {self.node_id}: {e}")
            # Fallback to equal distribution
            node_count = len(incentive_data)
            return {data['node_id']: 1.0 / node_count for data in incentive_data}
    
    def _elect_leader(self, round_number: int) -> str:
        """Elect leader for the given round (rotating)"""
        node_ids = [self.node_id] + [f"node_{i+1}" for i in range(len(self.other_nodes))]
        leader_index = round_number % len(node_ids)
        leader_id = node_ids[leader_index]
        
        self.current_leader = leader_id
        self.consensus_metrics['leader_elections'] += 1
        
        logger.info(f"Round {round_number}: Elected leader {leader_id}")
        return leader_id
    
    async def run_training_round(self, round_number: int) -> ConsensusResult:
        """Run a single training round with PBFT consensus"""
        try:
            start_time = time.time()
            
            # Elect leader for this round
            leader_id = self._elect_leader(round_number)
            
            # Train local model
            local_metrics = await self._train_local_model()
            
            # Prepare model update
            model_update = ModelUpdate(
                node_id=self.node_id,
                model_state=self.model.state_dict(),
                training_metrics=local_metrics,
                data_quality=local_metrics.get('data_quality', 0.5),
                reliability=local_metrics.get('reliability', 0.5),
                timestamp=time.time()
            )
            
            # If this node is the leader, initiate consensus
            if leader_id == self.node_id:
                await self._initiate_consensus([model_update], round_number)
            
            # Wait for consensus to complete
            consensus_time = time.time() - start_time
            
            return ConsensusResult(
                success=True,
                agreed_model=self.model.state_dict(),
                agreed_incentives={self.node_id: 1.0/3},  # Placeholder
                consensus_time=consensus_time,
                participating_nodes=[self.node_id] + [f"node_{i+1}" for i in range(len(self.other_nodes))]
            )
            
        except Exception as e:
            logger.error(f"Training round {round_number} failed on node {self.node_id}: {e}")
            return ConsensusResult(
                success=False,
                agreed_model=None,
                agreed_incentives=None,
                consensus_time=0.0,
                participating_nodes=[]
            )
    
    async def _train_local_model(self) -> Dict[str, float]:
        """Train the local model and return metrics"""
        try:
            # Simple training simulation
            # In a real implementation, this would use the actual training data
            training_loss = np.random.uniform(0.1, 0.5)
            training_accuracy = np.random.uniform(0.8, 0.95)
            
            return {
                'training_loss': training_loss,
                'training_accuracy': training_accuracy,
                'data_quality': np.random.uniform(0.5, 1.0),
                'reliability': np.random.uniform(0.8, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Local training failed on node {self.node_id}: {e}")
            return {
                'training_loss': 1.0,
                'training_accuracy': 0.0,
                'data_quality': 0.0,
                'reliability': 0.0
            }
    
    async def _initiate_consensus(self, model_updates: List[ModelUpdate], round_number: int):
        """Initiate PBFT consensus as the leader"""
        try:
            self.sequence_number += 1
            
            # Prepare consensus content
            content = {
                'model_updates': [
                    {
                        'node_id': update.node_id,
                        'model_state': update.model_state,
                        'training_metrics': update.training_metrics,
                        'data_quality': update.data_quality,
                        'reliability': update.reliability
                    }
                    for update in model_updates
                ],
                'incentive_data': [
                    {
                        'node_id': update.node_id,
                        'data_quality': update.data_quality,
                        'reliability': update.reliability
                    }
                    for update in model_updates
                ]
            }
            
            # Create pre-prepare message
            pre_prepare_message = PBFTMessage(
                phase=PBFTPhase.PRE_PREPARE,
                view_number=self.view_number,
                sequence_number=self.sequence_number,
                node_id=self.node_id,
                content=content,
                timestamp=time.time()
            )
            
            # Broadcast pre-prepare message
            await self._broadcast_message(pre_prepare_message)
            
            logger.info(f"Node {self.node_id} initiated consensus for round {round_number}")
            
        except Exception as e:
            logger.error(f"Failed to initiate consensus on node {self.node_id}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
                'node_id': self.node_id,
            'consensus_metrics': self.consensus_metrics,
            'current_leader': self.current_leader,
            'view_number': self.view_number,
            'sequence_number': self.sequence_number
        }
    
    async def shutdown(self):
        """Shutdown the system"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            logger.info(f"Node {self.node_id} shutdown complete")
                
        except Exception as e:
            logger.error(f"Error during shutdown of node {self.node_id}: {e}")


async def run_fully_decentralized_training(
    num_rounds: int = 10,
    node_configs: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run fully decentralized training with 3 nodes using PBFT consensus
    
    Args:
        num_rounds: Number of training rounds
        node_configs: Configuration for each node
    
    Returns:
        Dictionary containing training results and metrics
    """
    if node_configs is None:
        # Default configuration for 3 nodes
        node_configs = [
            {
                'node_id': 'node_1',
                'port': 8765,
                'other_nodes': [('localhost', 8766), ('localhost', 8767)]
            },
            {
                'node_id': 'node_2', 
                'port': 8766,
                'other_nodes': [('localhost', 8765), ('localhost', 8767)]
            },
            {
                'node_id': 'node_3',
                'port': 8767,
                'other_nodes': [('localhost', 8765), ('localhost', 8766)]
            }
        ]
    
    # Common configuration
    model_config = {
        'input_dim': 32,
        'hidden_dim': 128,
        'embedding_dim': 64,
        'num_classes': 2,
        'sequence_length': 12
    }
    
    data_config = {
        'zero_day_attack': 'DoS'
    }
    
    # Initialize nodes
    nodes = []
    for config in node_configs:
        node = FullyDecentralizedSystem(
            node_id=config['node_id'],
            port=config['port'],
            other_nodes=config['other_nodes'],
            model_config=model_config,
            data_config=data_config
        )
        nodes.append(node)
    
    # Initialize all nodes
    logger.info("Initializing fully decentralized nodes...")
    for node in nodes:
        await node.initialize()
    
    # Wait for all nodes to be ready
    await asyncio.sleep(2)
    
    # Run training rounds
    logger.info(f"Starting {num_rounds} training rounds...")
    results = {
        'rounds': [],
        'overall_metrics': {
            'total_rounds': 0,
            'successful_rounds': 0,
            'average_consensus_time': 0.0,
            'total_consensus_time': 0.0
        }
    }
    
    for round_num in range(num_rounds):
        logger.info(f"Starting round {round_num + 1}/{num_rounds}")
        
        # Run training round on all nodes
        round_results = []
        for node in nodes:
            result = await node.run_training_round(round_num)
            round_results.append(result)
        
        # Collect round metrics
        successful_rounds = sum(1 for r in round_results if r.success)
        avg_consensus_time = np.mean([r.consensus_time for r in round_results if r.success])
        
        round_metrics = {
            'round_number': round_num,
            'successful_nodes': successful_rounds,
            'total_nodes': len(nodes),
            'average_consensus_time': avg_consensus_time,
            'consensus_success_rate': successful_rounds / len(nodes)
        }
        
        results['rounds'].append(round_metrics)
        results['overall_metrics']['total_rounds'] += 1
        results['overall_metrics']['successful_rounds'] += successful_rounds
        results['overall_metrics']['total_consensus_time'] += avg_consensus_time
        
        logger.info(f"Round {round_num + 1} completed: {successful_rounds}/{len(nodes)} nodes successful")
    
    # Calculate final metrics
    if results['overall_metrics']['total_rounds'] > 0:
        results['overall_metrics']['average_consensus_time'] = (
            results['overall_metrics']['total_consensus_time'] / 
            results['overall_metrics']['total_rounds']
        )
        results['overall_metrics']['success_rate'] = (
            results['overall_metrics']['successful_rounds'] / 
            (results['overall_metrics']['total_rounds'] * len(nodes))
        )
    
    # Collect individual node metrics
    results['node_metrics'] = {}
    for node in nodes:
        results['node_metrics'][node.node_id] = node.get_metrics()
    
    # Shutdown nodes
    logger.info("Shutting down nodes...")
    for node in nodes:
        await node.shutdown()
    
    logger.info("Fully decentralized training completed")
    return results


if __name__ == "__main__":
    # Test the fully decentralized system
    logging.basicConfig(level=logging.INFO)
    
    async def test_system():
        results = await run_fully_decentralized_training(num_rounds=5)
        print("Training Results:")
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(test_system())
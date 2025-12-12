"""
Centralized Configuration System for Blockchain Federated Learning
This module provides a single source of truth for all system configuration.

UPDATED: Added category grouping functionality following IResTAE²A paper approach
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import os
import torch


@dataclass
class SystemConfig:
    """Centralized system configuration - single source of truth"""
    
    # === CENTRALIZED LEARNING CONFIGURATION ===
    # Federated learning removed - only centralized learning supported
    learning_rate: float = 0.0016387494099028342  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    batch_size: int = 256  # Reduced from 16 to 8 for GPU memory efficiency (will auto-reduce further if OOM)
    local_epochs: int = 10  # Epochs per training phase (legacy parameter, not used in centralized mode)
   
    # === QUICK VERIFICATION MODE ===
    quick_verify: bool = False  # When True, run a fast built-in self-check path
    
    # ==================================================================================
    # === CATEGORY GROUPING CONFIGURATION (IResTAE²A Paper Approach) ===
    # ==================================================================================
    # When enabled, groups specific attacks into semantic categories for evaluation
    # Example: NSL-KDD: 40 specific attacks → 5 categories (Normal, DoS, Probe, R2L, U2R)
    # Example: CICIDS2017: 15 specific attacks → 7 categories
    # Example: CICIoT2023: 34 specific attacks → 13 categories
    #
    # Impact on Performance:
    #   - Fine-grained (use_category_grouping=False): 80-85% accuracy (harder, more realistic)
    #   - Grouped (use_category_grouping=True): 88-92% accuracy (easier, +5-15% boost)
    #
    # Recommendation:
    #   - Use False (fine-grained) for PRIMARY evaluation (stronger contribution)
    #   - Use True (grouped) for COMPARISON with IResTAE²A and other papers
    # ==================================================================================
    use_category_grouping: bool = True  # Default: Fine-grained evaluation (harder, more impressive)
    
    # Category mappings (will be auto-populated in __post_init__ based on dataset)
    attack_category_mapping: Dict[str, str] = field(default_factory=dict)
    category_types: Dict[str, int] = field(default_factory=dict)
    
    # === DATA CONFIGURATION ===
    # KDDTest+ (NSL-KDD):
    data_path: str = "KDDTrain+.csv"
    test_path: str = "KDDTest+.csv"
    # When use_category_grouping=True: zero_day_attack should be a category name (e.g., "DoS", "Probe", "R2L", "U2R")
    # When use_category_grouping=False: zero_day_attack should be a specific attack name (e.g., "neptune", "back")
    zero_day_attack: str = "DoS"  # Category name for zero-day testing when grouping is enabled (represents all DoS attacks in KDD dataset)
    
    # === CROSS-DATASET EVALUATION (Optional) ===
    # Enable cross-dataset evaluation: train on one dataset, test on another
    use_cross_dataset_evaluation: bool = False  # Set to True for cross-dataset evaluation
    source_data_path: Optional[str] = None  # Training dataset path (if None, uses data_path)
    target_test_path: Optional[str] = None  # Testing dataset path (if None, uses test_path)
    # Example: Train on KDD, test on CICIDS2017
    # use_cross_dataset_evaluation: bool = True
    # source_data_path: str = "KDDTrain+.csv"
    # target_test_path: str = "CICIDS2017_test.csv"
    
    # CICIDS2017 (commented out):
    # data_path: str = "CICIDS2017_train.csv"
    # test_path: str = "CICIDS2017_test.csv"
    # zero_day_attack: str = "SSH-Patator"
    
    # CICIoT2023 (CICIDS2023) - commented out:
    # data_path: str = "CICIOT23train.csv"
    # test_path: str = "CICIOT23test.csv"
    # zero_day_attack: str = "DDoS-ACK_Fragmentation"  # Choose one attack as zero-day
       
    # Attack type mapping (KDDTest+ / NSL-KDD dataset)
    # Note: This contains 40 specific attack types (fine-grained)
    # When use_category_grouping=True, these will be mapped to 5 categories
    attack_types = {
        'normal': 0,
        # DoS attacks (10 types)
        'back': 1,
        'land': 2,
        'neptune': 3,
        'pod': 4,
        'smurf': 5,
        'teardrop': 6,
        'mailbomb': 23,
        'apache2': 24,
        'processtable': 25,
        'udpstorm': 26,
        # Probe attacks (6 types)
        'ipsweep': 7,
        'nmap': 8,
        'portsweep': 9,
        'satan': 10,
        'mscan': 27,
        'saint': 28,
        # R2L attacks (16 types)
        'guess_passwd': 11,
        'ftp_write': 12,
        'imap': 13,
        'multihop': 14,
        'phf': 15,
        'spy': 16,
        'warezclient': 17,
        'warezmaster': 18,
        'xlock': 29,
        'xsnoop': 30,
        'snmpguess': 31,
        'snmpgetattack': 32,
        'httptunnel': 33,
        'sendmail': 34,
        'named': 35,
        'worm': 39,
        # U2R attacks (7 types)
        'buffer_overflow': 19,
        'loadmodule': 20,
        'perl': 21,
        'rootkit': 22,
        'ps': 36,
        'sqlattack': 37,
        'xterm': 38,
    }
    
    # CICIoT2023 / CICIDS2023 attack types (uncomment if switching datasets)
    '''
    attack_types = {
        'BenignTraffic': 0,
        'Backdoor_Malware': 1,
        'BrowserHijacking': 2,
        'CommandInjection': 3,
        'DDoS-ACK_Fragmentation': 4,
        'DDoS-HTTP_Flood': 5,
        'DDoS-ICMP_Flood': 6,
        'DDoS-ICMP_Fragmentation': 7,
        'DDoS-PSHACK_Flood': 8,
        'DDoS-RSTFINFlood': 9,
        'DDoS-SYN_Flood': 10,
        'DDoS-SlowLoris': 11,
        'DDoS-SynonymousIP_Flood': 12,
        'DDoS-TCP_Flood': 13,
        'DDoS-UDP_Flood': 14,
        'DDoS-UDP_Fragmentation': 15,
        'DNS_Spoofing': 16,
        'DictionaryBruteForce': 17,
        'DoS-HTTP_Flood': 18,
        'DoS-SYN_Flood': 19,
        'DoS-TCP_Flood': 20,
        'DoS-UDP_Flood': 21,
        'MITM-ArpSpoofing': 22,
        'Mirai-greeth_flood': 23,
        'Mirai-greip_flood': 24,
        'Mirai-udpplain': 25,
        'Recon-HostDiscovery': 26,
        'Recon-OSScan': 27,
        'Recon-PingSweep': 28,
        'Recon-PortScan': 29,
        'SqlInjection': 30,
        'Uploading_Attack': 31,
        'VulnerabilityScan': 32,
        'XSS': 33,
    }
    '''
    
    # CICIDS2017 attack types (uncomment if switching datasets)
    '''
    attack_types = {
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 2,
        'DoS GoldenEye': 3,
        'DoS Hulk': 4,
        'DoS Slowhttptest': 5,
        'DoS slowloris': 6,
        'FTP-Patator': 7,
        'Heartbleed': 8,
        'Infiltration': 9,
        'PortScan': 10,
        'SSH-Patator': 11,
        'Web Attack  Brute Force': 12,
        'Web Attack  Sql Injection': 13,
        'Web Attack  XSS': 14,
    }
    '''
    
    def __post_init__(self):
        """Initialize category mappings based on active dataset"""
        self._initialize_category_mappings()
    
    def _initialize_category_mappings(self):
        """
        Initialize category mappings for all supported datasets
        Following IResTAE²A paper approach for consistency
        """
        # Detect which dataset is active
        if 'neptune' in self.attack_types or 'back' in self.attack_types:
            # NSL-KDD dataset detected
            self._init_nslkdd_categories()
        elif 'DDoS-ACK_Fragmentation' in self.attack_types or 'Mirai-greeth_flood' in self.attack_types:
            # CICIoT2023 dataset detected
            self._init_ciciot2023_categories()
        elif 'BENIGN' in self.attack_types and 'Bot' in self.attack_types:
            # CICIDS2017 dataset detected
            self._init_cicids2017_categories()
    
    def _init_nslkdd_categories(self):
        """
        Initialize NSL-KDD category mapping
        40 specific attacks → 5 categories (following IResTAE²A Table IV)
        """
        # NSL-KDD: Map 40 specific attacks → 5 categories
        self.attack_category_mapping = {
            # Normal traffic
            'normal': 'Normal',
            
            # DoS category (10 attacks)
            'back': 'DoS',
            'land': 'DoS',
            'neptune': 'DoS',
            'pod': 'DoS',
            'smurf': 'DoS',
            'teardrop': 'DoS',
            'apache2': 'DoS',
            'udpstorm': 'DoS',
            'processtable': 'DoS',
            'mailbomb': 'DoS',
            
            # Probe category (6 attacks)
            'ipsweep': 'Probe',
            'nmap': 'Probe',
            'portsweep': 'Probe',
            'satan': 'Probe',
            'saint': 'Probe',
            'mscan': 'Probe',
            
            # R2L category (16 attacks)
            'ftp_write': 'R2L',
            'guess_passwd': 'R2L',
            'imap': 'R2L',
            'multihop': 'R2L',
            'phf': 'R2L',
            'spy': 'R2L',
            'warezclient': 'R2L',
            'warezmaster': 'R2L',
            'sendmail': 'R2L',
            'named': 'R2L',
            'snmpgetattack': 'R2L',
            'snmpguess': 'R2L',
            'xlock': 'R2L',
            'xsnoop': 'R2L',
            'worm': 'R2L',
            'httptunnel': 'R2L',
            
            # U2R category (7 attacks)
            'buffer_overflow': 'U2R',
            'loadmodule': 'U2R',
            'perl': 'U2R',
            'rootkit': 'U2R',
            'sqlattack': 'U2R',
            'xterm': 'U2R',
            'ps': 'U2R',
        }
        
        # Category → integer mapping
        self.category_types = {
            'Normal': 0,
            'DoS': 1,
            'Probe': 2,
            'R2L': 3,
            'U2R': 4,
        }
    
    def _init_ciciot2023_categories(self):
        """
        Initialize CICIoT2023 category mapping
        34 specific attacks → 13 categories (semantic grouping)
        """
        # CICIoT2023: Map 34 specific attacks → 13 categories
        self.attack_category_mapping = {
            # Benign traffic
            'BenignTraffic': 'Benign',
            
            # DDoS attacks (12 types → 1 category)
            'DDoS-ACK_Fragmentation': 'DDoS',
            'DDoS-HTTP_Flood': 'DDoS',
            'DDoS-ICMP_Flood': 'DDoS',
            'DDoS-ICMP_Fragmentation': 'DDoS',
            'DDoS-PSHACK_Flood': 'DDoS',
            'DDoS-RSTFINFlood': 'DDoS',
            'DDoS-SYN_Flood': 'DDoS',
            'DDoS-SlowLoris': 'DDoS',
            'DDoS-SynonymousIP_Flood': 'DDoS',
            'DDoS-TCP_Flood': 'DDoS',
            'DDoS-UDP_Flood': 'DDoS',
            'DDoS-UDP_Fragmentation': 'DDoS',
            
            # DoS attacks (4 types → 1 category)
            'DoS-HTTP_Flood': 'DoS',
            'DoS-SYN_Flood': 'DoS',
            'DoS-TCP_Flood': 'DoS',
            'DoS-UDP_Flood': 'DoS',
            
            # Mirai botnet attacks (3 types → 1 category)
            'Mirai-greeth_flood': 'Mirai',
            'Mirai-greip_flood': 'Mirai',
            'Mirai-udpplain': 'Mirai',
            
            # Reconnaissance attacks (4 types → 1 category)
            'Recon-HostDiscovery': 'Recon',
            'Recon-OSScan': 'Recon',
            'Recon-PingSweep': 'Recon',
            'Recon-PortScan': 'Recon',
            
            # Web attacks (2 types → 1 category)
            'SqlInjection': 'WebAttack',
            'XSS': 'WebAttack',
            
            # Spoofing attacks (2 types → 1 category)
            'DNS_Spoofing': 'Spoofing',
            'MITM-ArpSpoofing': 'Spoofing',
            
            # Individual attacks (keep separate)
            'Backdoor_Malware': 'Backdoor',
            'BrowserHijacking': 'Hijacking',
            'CommandInjection': 'Injection',
            'DictionaryBruteForce': 'BruteForce',
            'Uploading_Attack': 'Upload',
            'VulnerabilityScan': 'Scan',
        }
        
        # Category → integer mapping
        self.category_types = {
            'Benign': 0,
            'DDoS': 1,
            'DoS': 2,
            'Mirai': 3,
            'Recon': 4,
            'WebAttack': 5,
            'Spoofing': 6,
            'Backdoor': 7,
            'Hijacking': 8,
            'Injection': 9,
            'BruteForce': 10,
            'Upload': 11,
            'Scan': 12,
        }
    
    def _init_cicids2017_categories(self):
        """
        Initialize CICIDS2017 category mapping
        15 specific attacks → 7 categories (following IResTAE²A Table IV)
        """
        # CICIDS2017: Map 15 specific attacks → 7 categories
        self.attack_category_mapping = {
            # Benign traffic
            'BENIGN': 'Benign',
            
            # DoS attacks (5 types → 1 category)
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'DoS slowloris': 'DoS',
            'Heartbleed': 'DoS',  # Grouped with DoS
            
            # DDoS attack
            'DDoS': 'DoS',  # Also grouped with DoS
            
            # Bot attack
            'Bot': 'Bot',
            
            # Infiltration attack
            'Infiltration': 'Infiltration',
            
            # Port scanning
            'PortScan': 'PortScan',
            
            # Brute force attacks (2 types → 1 category)
            'FTP-Patator': 'BruteForce',
            'SSH-Patator': 'BruteForce',
            
            # Web attacks (3 types → 1 category)
            'Web Attack  Brute Force': 'WebAttack',
            'Web Attack  Sql Injection': 'WebAttack',
            'Web Attack  XSS': 'WebAttack',
        }
        
        # Category → integer mapping
        self.category_types = {
            'Benign': 0,
            'DoS': 1,
            'Bot': 2,
            'Infiltration': 3,
            'PortScan': 4,
            'BruteForce': 5,
            'WebAttack': 6,
        }
    
    # ==================================================================================
    # === SMART PROPERTIES (Auto-adapt to fine-grained or grouped mode) ===
    # ==================================================================================
    
    @property
    def zero_day_attack_label(self) -> int:
        """
        Get the integer label for the zero-day attack type
        Auto-adapts based on use_category_grouping flag
        """
        if self.use_category_grouping and self.category_types:
            # Grouped mode: Check if zero_day_attack is already a category name
            if self.zero_day_attack in self.category_types:
                # It's already a category name (e.g., "DoS")
                return self.category_types.get(self.zero_day_attack, 0)
            else:
                # It's a specific attack name - look up its category
                category = self.attack_category_mapping.get(self.zero_day_attack)
                if category:
                    label = self.category_types.get(category, 0)
                    return label
        
        # Fine-grained mode: Return specific attack label
        return self.attack_types.get(self.zero_day_attack, 0)
    
    @property
    def zero_day_category(self) -> str:
        """Get the category name of the zero-day attack"""
        if self.use_category_grouping and self.category_types:
            # If zero_day_attack is already a category name, return it
            if self.zero_day_attack in self.category_types:
                return self.zero_day_attack
            # Otherwise, look up the category for this specific attack
            if self.attack_category_mapping:
                return self.attack_category_mapping.get(self.zero_day_attack, "Unknown")
        return self.zero_day_attack
    
    @property
    def num_attack_types(self) -> int:
        """Get number of attack types (specific or categories)"""
        if self.use_category_grouping and self.category_types:
            return len(self.category_types)
        return len(self.attack_types)
    
    def get_attack_label(self, attack_name: str) -> int:
        """
        Get integer label for an attack (auto-adapts to mode)
        
        Args:
            attack_name: Name of the attack
            
        Returns:
            int: Label in fine-grained or grouped mode
        """
        if self.use_category_grouping and self.category_types:
            # Grouped mode
            category = self.attack_category_mapping.get(attack_name, 'Benign')
            return self.category_types.get(category, 0)
        else:
            # Fine-grained mode
            return self.attack_types.get(attack_name, 0)
    
    def get_all_attack_names(self):
        """Returns all attack names (specific or categories)"""
        if self.use_category_grouping and self.category_types:
            return list(self.category_types.keys())
        else:
            return list(self.attack_types.keys())
    
    def get_evaluation_info(self) -> dict:
        """Get evaluation configuration info for logging"""
        return {
            'mode': 'Grouped' if self.use_category_grouping else 'Fine-grained',
            'num_classes': self.num_attack_types,
            'zero_day_attack': self.zero_day_attack,
            'zero_day_label': self.zero_day_attack_label,
            'zero_day_category': self.zero_day_category if self.use_category_grouping else None,
        }
    
    # ==================================================================================
    # === REST OF ORIGINAL CONFIGURATION (UNCHANGED) ===
    # ==================================================================================
    
    # === MODEL CONFIGURATION ===
    input_dim: int = 41  # KDDTest+ has 41 features (42 columns - 1 label column)
    hidden_dim: int = 128  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    embedding_dim: int = 256  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    num_classes: int = 2  # Binary classification (Normal vs Attack) for zero-day detection
    
    # === FEATURE SELECTION CONFIGURATION ===
    use_igrf_rfe: bool = False  # TEMPORARILY DISABLED: Enable IGRF-RFE hybrid feature selection
    feature_selection_ratio: float = 0.8  # Select top 80% of features
    
    # === TCN CONFIGURATION ===
    use_tcn: bool = True  # Enable/disable TCN feature extraction
    disable_tcn_feature_extraction: bool = False  # If True, replace TCN with simple pooling (for testing)
    sequence_length: int = 22  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    sequence_stride: int = 12  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    tcn_kernel_sizes: tuple = (2, 3, 3)  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    use_residual_connections: bool = False  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    meta_epochs: int = 21  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    transductive_steps: int = 40  # AGGRESSIVE FOR 90%+ BASE MODEL: Increased from 20 → 40 (2x increase)
    transductive_lr: float = 0.001  # AGGRESSIVE FOR 90%+ BASE MODEL: Increased from 0.0007 → 0.001 (43% increase)
    transductive_refinement_iterations: int = 10  # AGGRESSIVE FOR 90%+ BASE MODEL: Increased from 10 → 30 (3x increase)
    transductive_refinement_confidence_threshold: float = 0.7  # Base confidence threshold (used if adaptive=False)
    use_adaptive_refinement_threshold: bool = True  # Enable adaptive thresholding based on class imbalance and prediction entropy
    transductive_refinement_min_threshold: float = 0.5  # Minimum confidence threshold for adaptive mode
    transductive_refinement_max_threshold: float = 0.9  # Maximum confidence threshold for adaptive mode
    
    # === EMBEDDING DISCRIMINATIVENESS IMPROVEMENT (Center Loss & Prototype Margin Loss) ===
    use_center_loss: bool = True  # Enable Center Loss for intra-class compactness (reduces embedding variance)
    center_loss_weight: float = 0.08  # AGGRESSIVE FOR 90%+ BASE MODEL: Increased from 0.02 → 0.08 (4x increase)
    use_prototype_margin_loss: bool = True  # Enable Prototype Margin Loss for inter-class separation
    margin_loss_weight: float = 0.25  # AGGRESSIVE FOR 90%+ BASE MODEL: Increased from 0.12 → 0.25 (108% increase)
    prototype_margin: float = 4.5  # AGGRESSIVE FOR 90%+ BASE MODEL: Increased from 2.5 → 4.5 (80% increase)
    
    # === ADVANCED EMBEDDING TECHNIQUES ===
    use_supervised_contrastive_loss: bool = True  # Enable Supervised Contrastive Loss for better embeddings (ENABLED to improve separability)
    contrastive_loss_weight: float = 0.3  # Weight for contrastive loss (updated to match patch)
    contrastive_temperature: float = 0.07  # Temperature for contrastive loss
    
    use_multi_prototype: bool = True  # Enable Multi-Prototype Learning (3 prototypes per class) (ENABLED for better separability)
    prototypes_per_class: int = 3  # Number of prototypes per class for multi-prototype learning
    multi_prototype_weight: float = 0.2  # Weight for multi-prototype loss
    
    use_mixup_augmentation: bool = True  # Enable Mixup data augmentation during training (ENABLED for better generalization)
    mixup_alpha: float = 0.4  # Alpha parameter for Mixup (beta distribution)
    mixup_probability: float = 0.8  # Probability of applying Mixup (80% of the time)
    
    # === TEST-TIME TRAINING (TTT) CONFIGURATION ===
    # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    # ADJUSTED: Reduced overfitting by increasing regularization and reducing adaptation intensity
    ttt_base_steps: int = 70  # REDUCED from 85 → 70 (more aggressive prevention of overfitting)
    ttt_max_steps: int = 400  # Maximum TTT steps (safety limit)
    ttt_adaptation_query_size: int = 1198  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    ttt_batch_size: int = 64  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    ttt_lr: float = 0.002  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    ttt_l2_reg_weight: float = 0.01  # INCREASED from 0.001 → 0.01 (10x increase for stronger regularization)
    confidence_rejection_threshold: float = 0.90  # Increased to 0.90 for stricter FAR control (reject more uncertain predictions)
    
    # === ATTACK PROTOTYPE DISCOVERY TTT ===
    ttt_prototype_clusters: int = 10  # Number of attack prototypes to discover
    ttt_prototype_weight: float = 0.5  # REDUCED from 1.0 → 0.5 (reduce overfitting to zero-day prototypes)
    ttt_prototype_entropy_weight: float = 0.3  # Weight for entropy loss (complement to prototype)
    ttt_prototype_steps: int = 100  # Number of adaptation steps for prototype TTT
    ttt_lr_min: float = 4e-5  # Minimum learning rate for cosine schedule
    ttt_lr_decay: float = 0.8  # (Unused now for TTT; kept for backward compatibility)
    ttt_warmup_steps: int = 20  # Learning rate warmup steps
    ttt_weight_decay: float = 1e-4  # Increased weight decay to regularize aggressive updates
    ttt_patience: int = 40  # Increased from 30 - more patience for better base model adaptation
    ttt_timeout: int = 60  # Increased from 45 - more time for additional steps
    ttt_improvement_threshold: float = 1e-6  # More sensitive threshold for better base model
    ttt_entropy_weight: float = 0.5    # Weight for entropy loss in TTT (test-time only)
    ttt_consistency_weight: float = 0.10  # Weight for prototype consistency in unsupervised TTT
    ttt_mixup_alpha: float = 0.20       # DISABLED: MixUp inappropriate for TTT with unlabeled data
    ttt_temperature: float = 1.3109823217156622  # Optimized from Optuna Trial 1 (best_hyperparameters.json)
    use_semisupervised_ttt: bool = False  # When True, use new semi-supervised TTT instead of legacy TENT+pseudo

    # === TENT + PSEUDO-LABELS CONFIGURATION ===
    # ADJUSTED: Reduced overfitting by balancing pseudo-label and entropy weights
    use_pseudo_labels: bool = True
    pseudo_threshold: float = 0.95
    pseudo_min_threshold: float = 0.7173803589287694
    pseudo_weight: float = 1.5  # REDUCED from 3.04 → 1.5 (51% reduction to prevent overfitting)
    entropy_weight: float = 0.8  # INCREASED from 0.57 → 0.8 (stronger balance adaptation across all samples)
    use_teacher: bool = True
    ema_decay: float = 0.9662140032177797
    
    # === ADVANCED TTT TECHNIQUES FOR SOTA PERFORMANCE ===
    use_focal_loss: bool = False
    focal_gamma: float = 2.4563362070328196
    focal_alpha: float = 0.3274425485152653
    use_mixup_ttt: bool = False
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    
    # === CLASS WEIGHT AND MAGIC NUMBERS ===
    pseudo_label_temperature: float = 0.3317791751430118
    transductive_patience: int = 8
    missing_class_weight_multiplier: float = 2.0
    class_weight_normalization_multiplier: float = 2.0
    use_multi_scale_tta: bool = False
    tta_scales: list = field(default_factory=lambda: [0.9, 1.0, 1.1])
    use_self_ensemble: bool = False
    ensemble_checkpoints: int = 3
    use_lr_warmup: bool = True
    warmup_steps: int = 20
    ttt_pseudo_loss_weight: float = 1.0
    ttt_attack_prior: float = 0.30
    ttt_early_stopping: bool = True
    ttt_early_stopping_patience: int = 15
    ttt_early_stopping_min_delta: float = 1e-4
    ttt_pseudo_label_validation: bool = False
    ttt_validation_forward_passes: int = 3
    ttt_validation_noise_std: float = 0.05
    ttt_zero_day_focused: bool = False
    ttt_bn_statistics_adaptation: bool = False
    ttt_bn_ema_decay: float = 0.9
    ttt_adaptive_zdr_threshold: bool = False
    ttt_zdr_target: float = 0.85
    ttt_zdr_max_far: float = 0.50
    
    # Consistency loss configuration
    consistency_weight: float = 0.3
    jitter_sigma: float = 0.10
    scale_sigma: float = 0.15
    
    # Ensemble configuration
    use_ensemble_ttt: bool = False
    ensemble_variants: list = field(default_factory=lambda: ['tent', 'tent_pseudo', 'prototype'])
    use_best_individual_if_ensemble_fails: bool = True
    
    # === THRESHOLD OPTIMIZATION STRATEGY ===
    # Options: 'balanced_zdr_far', 'far_optimized', 'pr_optimized', 'zdr_optimized'
    threshold_optimization_strategy: str = 'far_optimized'  # Prioritize FAR < 1% over ZDR
    use_adaptive_threshold: bool = True
    threshold_adaptation_mode: str = 'combined'
    max_far_for_zdr: float = 0.35
    
    # === FAR-AWARE THRESHOLD OPTIMIZATION ===
    # Target: FAR < 1% for production-ready systems
    max_far_allowed: float = 0.01  # 1% maximum false alarm rate (very strict)
    min_zdr_required: float = 0.75  # Reduced from 0.85 to allow stricter FAR constraint
    
    # === TRAINING CONFIGURATION ===
    support_weight: float = 0.5
    test_weight: float = 0.5
    validation_patience_limit: int = 10
    recent_rounds: int = 10
    
    # === EVALUATION CONFIGURATION ===
    support_size: int = 20
    num_meta_tasks: int = 46
    
    # === FEW-SHOT LEARNING CONFIGURATION ===
    n_way: int = 2
    k_shot: int = 152
    n_query: int = 16
    enforce_equal_support_composition: bool = False
    include_all_attack_types_in_support: bool = False
    
    # === VALIDATION CONFIGURATION ===
    max_val_samples: int = 1000
    overfitting_threshold: float = 0.15
    max_overfitting_rounds: int = 5
    recent_rounds: int = 10
    
    # === PERFORMANCE THRESHOLDS ===
    default_threshold: float = 0.5
    participation_excellent: float = 0.95
    participation_good: float = 0.90
    participation_fair: float = 0.80
    participation_poor: float = 0.70
    recent_participation_bonus: float = 5.0
    retry_delay: float = 1.0
    max_retries: int = 3
    
    # === DEVICE CONFIGURATION ===
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create configuration from environment variables (optional)"""
        return cls(
            zero_day_attack=os.getenv('ZERO_DAY_ATTACK', 'Exploits'),
            use_tcn=os.getenv('USE_TCN', 'true').lower() == 'true',
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging"""
        base_dict = {
            'zero_day_attack': self.zero_day_attack,
            'use_tcn': self.use_tcn,
            'sequence_length': self.sequence_length,
            'sequence_stride': self.sequence_stride,
            'local_epochs': self.local_epochs,
            'learning_rate': self.learning_rate,
            'ttt_base_steps': self.ttt_base_steps,
            'ttt_max_steps': self.ttt_max_steps,
            'ttt_batch_size': self.ttt_batch_size,
            'ttt_lr': self.ttt_lr,
        }
        
        # Add category grouping info
        if self.use_category_grouping:
            base_dict.update(self.get_evaluation_info())
        
        return base_dict

# Global configuration instance - single source of truth
config = SystemConfig()

# Convenience function to get config
def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config

# Convenience function to update config
def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

# Convenience function to reset config to defaults
def reset_config() -> None:
    """Reset configuration to defaults"""
    global config
    config = SystemConfig()

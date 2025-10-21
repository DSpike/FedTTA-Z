// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title DecentralizedConsensus
 * @dev Smart contract for decentralized federated learning consensus
 * @notice Implements 2-miner consensus mechanism with stake-based voting
 */
contract DecentralizedConsensus {
    
    // Events
    event MinerRegistered(address indexed miner, uint256 stake);
    event ProposalSubmitted(bytes32 indexed proposalHash, address indexed proposer, uint256 round);
    event VoteCast(bytes32 indexed proposalHash, address indexed voter, bool vote, uint256 weight);
    event ConsensusReached(bytes32 indexed proposalHash, uint256 consensusRatio, uint256 round);
    event ModelUpdated(bytes32 indexed modelHash, address indexed proposer, uint256 round);
    
    // Structs
    struct Miner {
        address minerAddress;
        uint256 stake;
        uint256 reputation;
        bool isActive;
        uint256 lastActivity;
    }
    
    struct Proposal {
        bytes32 proposalHash;
        address proposer;
        bytes32 modelHash;
        uint256 round;
        uint256 timestamp;
        uint256 validationScore;
        bool isActive;
    }
    
    struct Vote {
        address voter;
        bool vote; // true = agree, false = disagree
        uint256 confidence;
        uint256 timestamp;
    }
    
    // State variables
    mapping(address => Miner) public miners;
    mapping(bytes32 => Proposal) public proposals;
    mapping(bytes32 => Vote[]) public proposalVotes;
    mapping(uint256 => bytes32) public roundWinners;
    
    address[] public minerList;
    uint256 public totalStake;
    uint256 public consensusThreshold = 67; // 67% required for consensus
    uint256 public minStake = 1000;
    uint256 public currentRound = 0;
    uint256 public proposalTimeout = 300; // 5 minutes
    
    // Modifiers
    modifier onlyMiner() {
        require(miners[msg.sender].isActive, "Only active miners can perform this action");
        _;
    }
    
    modifier validProposal(bytes32 proposalHash) {
        require(proposals[proposalHash].isActive, "Proposal does not exist or is inactive");
        _;
    }
    
    // Functions
    
    /**
     * @dev Register a new miner with stake
     * @param stake Amount of stake to deposit
     */
    function registerMiner(uint256 stake) external payable {
        require(stake >= minStake, "Stake must be at least minimum required");
        require(!miners[msg.sender].isActive, "Miner already registered");
        
        miners[msg.sender] = Miner({
            minerAddress: msg.sender,
            stake: stake,
            reputation: 100, // Initial reputation
            isActive: true,
            lastActivity: block.timestamp
        });
        
        minerList.push(msg.sender);
        totalStake += stake;
        
        emit MinerRegistered(msg.sender, stake);
    }
    
    /**
     * @dev Submit a model aggregation proposal
     * @param modelHash Hash of the aggregated model
     * @param validationScore Quality score of the model
     */
    function submitProposal(bytes32 modelHash, uint256 validationScore) external onlyMiner {
        bytes32 proposalHash = keccak256(abi.encodePacked(
            msg.sender,
            modelHash,
            currentRound,
            block.timestamp
        ));
        
        proposals[proposalHash] = Proposal({
            proposalHash: proposalHash,
            proposer: msg.sender,
            modelHash: modelHash,
            round: currentRound,
            timestamp: block.timestamp,
            validationScore: validationScore,
            isActive: true
        });
        
        emit ProposalSubmitted(proposalHash, msg.sender, currentRound);
    }
    
    /**
     * @dev Vote on a proposal
     * @param proposalHash Hash of the proposal to vote on
     * @param vote True for agreement, false for disagreement
     * @param confidence Confidence level (0-100)
     */
    function voteOnProposal(bytes32 proposalHash, bool vote, uint256 confidence) 
        external 
        onlyMiner 
        validProposal(proposalHash) 
    {
        require(confidence <= 100, "Confidence must be between 0 and 100");
        require(proposals[proposalHash].proposer != msg.sender, "Cannot vote on own proposal");
        
        // Check if already voted
        Vote[] storage votes = proposalVotes[proposalHash];
        for (uint256 i = 0; i < votes.length; i++) {
            require(votes[i].voter != msg.sender, "Already voted on this proposal");
        }
        
        // Add vote
        votes.push(Vote({
            voter: msg.sender,
            vote: vote,
            confidence: confidence,
            timestamp: block.timestamp
        }));
        
        // Update miner activity
        miners[msg.sender].lastActivity = block.timestamp;
        
        emit VoteCast(proposalHash, msg.sender, vote, calculateVoteWeight(msg.sender, confidence));
        
        // Check consensus
        checkConsensus(proposalHash);
    }
    
    /**
     * @dev Check if consensus has been reached on a proposal
     * @param proposalHash Hash of the proposal to check
     */
    function checkConsensus(bytes32 proposalHash) public validProposal(proposalHash) {
        Vote[] storage votes = proposalVotes[proposalHash];
        require(votes.length >= 2, "Need at least 2 votes for consensus");
        
        uint256 totalWeight = 0;
        uint256 agreementWeight = 0;
        
        for (uint256 i = 0; i < votes.length; i++) {
            uint256 weight = calculateVoteWeight(votes[i].voter, votes[i].confidence);
            totalWeight += weight;
            
            if (votes[i].vote) {
                agreementWeight += weight;
            }
        }
        
        uint256 consensusRatio = (agreementWeight * 100) / totalWeight;
        
        if (consensusRatio >= consensusThreshold) {
            // Consensus reached
            proposals[proposalHash].isActive = false;
            roundWinners[currentRound] = proposalHash;
            
            // Update miner reputations
            updateReputations(proposalHash, votes);
            
            emit ConsensusReached(proposalHash, consensusRatio, currentRound);
            emit ModelUpdated(proposals[proposalHash].modelHash, proposals[proposalHash].proposer, currentRound);
            
            currentRound++;
        }
    }
    
    /**
     * @dev Calculate vote weight based on stake and reputation
     * @param voter Address of the voter
     * @param confidence Confidence level of the vote
     * @return Weight of the vote
     */
    function calculateVoteWeight(address voter, uint256 confidence) public view returns (uint256) {
        Miner memory miner = miners[voter];
        return (miner.stake * miner.reputation * confidence) / 10000; // Normalize to reasonable scale
    }
    
    /**
     * @dev Update miner reputations based on consensus results
     * @param proposalHash Hash of the winning proposal
     * @param votes Array of votes on the proposal
     */
    function updateReputations(bytes32 proposalHash, Vote[] memory votes) internal {
        address proposer = proposals[proposalHash].proposer;
        
        // Reward proposer
        miners[proposer].reputation = min(200, miners[proposer].reputation + 10);
        
        // Update voter reputations
        for (uint256 i = 0; i < votes.length; i++) {
            address voter = votes[i].voter;
            
            if (votes[i].vote) {
                // Reward correct vote
                miners[voter].reputation = min(200, miners[voter].reputation + 5);
            } else {
                // Penalize incorrect vote
                miners[voter].reputation = max(10, miners[voter].reputation - 2);
            }
        }
    }
    
    /**
     * @dev Get consensus status for a proposal
     * @param proposalHash Hash of the proposal
     * @return consensusRatio Current consensus ratio
     * @return totalVotes Number of votes cast
     * @return isConsensusReached Whether consensus has been reached
     */
    function getConsensusStatus(bytes32 proposalHash) external view returns (
        uint256 consensusRatio,
        uint256 totalVotes,
        bool isConsensusReached
    ) {
        Vote[] memory votes = proposalVotes[proposalHash];
        require(votes.length > 0, "No votes on this proposal");
        
        uint256 totalWeight = 0;
        uint256 agreementWeight = 0;
        
        for (uint256 i = 0; i < votes.length; i++) {
            uint256 weight = calculateVoteWeight(votes[i].voter, votes[i].confidence);
            totalWeight += weight;
            
            if (votes[i].vote) {
                agreementWeight += weight;
            }
        }
        
        consensusRatio = (agreementWeight * 100) / totalWeight;
        totalVotes = votes.length;
        isConsensusReached = consensusRatio >= consensusThreshold;
    }
    
    /**
     * @dev Get system statistics
     * @return totalMiners Number of active miners
     * @return totalStakeAmount Total stake in the system
     * @return currentRoundNumber Current round number
     */
    function getSystemStats() external view returns (
        uint256 totalMiners,
        uint256 totalStakeAmount,
        uint256 currentRoundNumber
    ) {
        totalMiners = minerList.length;
        totalStakeAmount = totalStake;
        currentRoundNumber = currentRound;
    }
    
    /**
     * @dev Get miner information
     * @param minerAddress Address of the miner
     * @return stake Miner's stake
     * @return reputation Miner's reputation
     * @return isActive Whether miner is active
     */
    function getMinerInfo(address minerAddress) external view returns (
        uint256 stake,
        uint256 reputation,
        bool isActive
    ) {
        Miner memory miner = miners[minerAddress];
        stake = miner.stake;
        reputation = miner.reputation;
        isActive = miner.isActive;
    }
    
    // Helper functions
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
    
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }
}










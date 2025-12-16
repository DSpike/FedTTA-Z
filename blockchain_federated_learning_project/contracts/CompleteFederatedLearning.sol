// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CompleteFederatedLearning {
    // Events
    event ParticipantRegistered(address indexed participant, string role);
    event ModelUpdateSubmitted(bytes32 indexed modelHash, bytes32 indexed ipfsCid, uint256 roundNumber);
    event ClientUpdateSubmitted(address indexed client, bytes32 indexed modelHash, bytes32 indexed ipfsCid, uint256 roundNumber);
    event ContributionSubmitted(address indexed contributor, uint256 roundNumber, bytes32 modelHash, uint256 accuracyImprovement, uint256 dataQuality, uint256 reliability);
    event ContributionEvaluated(address indexed contributor, uint256 roundNumber, uint256 contributionScore, uint256 tokenReward, bool verified);
    event RewardDistributed(address indexed contributor, uint256 amount);

    // State variables
    mapping(address => bool) public participants;
    mapping(address => string) public participantRoles;
    mapping(bytes32 => bool) public submittedModels;
    mapping(address => mapping(uint256 => bool)) public clientUpdates;
    mapping(address => mapping(uint256 => uint256)) public contributionScores;
    mapping(address => uint256) public totalRewards;

    // Modifiers
    modifier onlyParticipant() {
        require(participants[msg.sender], "Not a registered participant");
        _;
    }

    // Functions
    function registerParticipant(address participant, string memory role) external {
        participants[participant] = true;
        participantRoles[participant] = role;
        emit ParticipantRegistered(participant, role);
    }

    function submitModelUpdate(
        bytes32 modelHash,
        bytes32 ipfsCid,
        uint256 roundNumber
    ) external onlyParticipant {
        submittedModels[modelHash] = true;
        emit ModelUpdateSubmitted(modelHash, ipfsCid, roundNumber);
    }

    function submitClientUpdate(
        address client,
        bytes32 modelHash,
        bytes32 ipfsCid,
        uint256 roundNumber
    ) external onlyParticipant {
        clientUpdates[client][roundNumber] = true;
        emit ClientUpdateSubmitted(client, modelHash, ipfsCid, roundNumber);
    }

    function submitContribution(
        address contributor,
        uint256 roundNumber,
        bytes32 modelHash,
        uint256 accuracyImprovement,
        uint256 dataQuality,
        uint256 reliability
    ) external onlyParticipant {
        emit ContributionSubmitted(contributor, roundNumber, modelHash, accuracyImprovement, dataQuality, reliability);
    }

    function evaluateContribution(
        address contributor,
        uint256 roundNumber,
        uint256 contributionScore,
        uint256 tokenReward,
        bool verified
    ) external onlyParticipant {
        contributionScores[contributor][roundNumber] = contributionScore;
        emit ContributionEvaluated(contributor, roundNumber, contributionScore, tokenReward, verified);
    }

    function distributeReward(
        address contributor,
        uint256 amount
    ) external onlyParticipant {
        totalRewards[contributor] += amount;
        emit RewardDistributed(contributor, amount);
    }

t 
    // View functions
    function isParticipant(address participant) external view returns (bool) {
        return participants[participant];
    }

    function getParticipantRole(address participant) external view returns (string memory) {
        return participantRoles[participant];
    }

    function isModelSubmitted(bytes32 modelHash) external view returns (bool) {
        return submittedModels[modelHash];
    }

    function hasClientUpdate(address client, uint256 roundNumber) external view returns (bool) {
        return clientUpdates[client][roundNumber];
    }

    function getContributionScore(address contributor, uint256 roundNumber) external view returns (uint256) {
        return contributionScores[contributor][roundNumber];
    }

    function getTotalRewards(address contributor) external view returns (uint256) {
        return totalRewards[contributor];
    }
}

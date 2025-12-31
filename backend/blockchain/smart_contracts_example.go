package main

import (
	"encoding/json"
	"fmt"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"time"
)

// ModelRegistry 模型注册智能合约
type ModelRegistry struct {
	contractapi.Contract
}

// ModelInfo 模型信息结构体
type ModelInfo struct {
	ModelID      string            `json:"model_id"`
	ModelHash    string            `json:"model_hash"`
	Metadata     map[string]string `json:"metadata"`
	CreatedAt    string            `json:"created_at"`
	UpdatedAt    string            `json:"updated_at"`
	Version      int               `json:"version"`
}

// ModelVersion 模型版本历史
type ModelVersion struct {
	Version     int               `json:"version"`
	ModelHash   string            `json:"model_hash"`
	Metadata    map[string]string `json:"metadata"`
	Timestamp   string            `json:"timestamp"`
	Transaction string            `json:"transaction_id"`
}

// RegisterModel 注册新模型
func (mr *ModelRegistry) RegisterModel(ctx contractapi.TransactionContextInterface, 
	modelID string, modelHash string, metadataJSON string) error {
	
	// 解析元数据
	var metadata map[string]string
	if err := json.Unmarshal([]byte(metadataJSON), &metadata); err != nil {
		return fmt.Errorf("failed to parse metadata: %v", err)
	}
	
	// 检查模型是否已存在
	existingModel, err := ctx.GetStub().GetState(modelID)
	if err != nil {
		return fmt.Errorf("failed to check model existence: %v", err)
	}
	if existingModel != nil {
		return fmt.Errorf("model %s already exists", modelID)
	}
	
	// 创建模型信息
	modelInfo := ModelInfo{
		ModelID:   modelID,
		ModelHash: modelHash,
		Metadata:  metadata,
		CreatedAt: time.Now().Format(time.RFC3339),
		UpdatedAt: time.Now().Format(time.RFC3339),
		Version:   1,
	}
	
	// 序列化并存储模型信息
	modelJSON, err := json.Marshal(modelInfo)
	if err != nil {
		return fmt.Errorf("failed to marshal model info: %v", err)
	}
	
	if err := ctx.GetStub().PutState(modelID, modelJSON); err != nil {
		return fmt.Errorf("failed to put model state: %v", err)
	}
	
	// 记录版本历史
	versionKey := fmt.Sprintf("%s_version_1", modelID)
	version := ModelVersion{
		Version:     1,
		ModelHash:   modelHash,
		Metadata:    metadata,
		Timestamp:   time.Now().Format(time.RFC3339),
		Transaction: ctx.GetStub().GetTxID(),
	}
	
	versionJSON, err := json.Marshal(version)
	if err != nil {
		return fmt.Errorf("failed to marshal version info: %v", err)
	}
	
	if err := ctx.GetStub().PutState(versionKey, versionJSON); err != nil {
		return fmt.Errorf("failed to put version state: %v", err)
	}
	
	return nil
}

// UpdateModelVersion 更新模型版本
func (mr *ModelRegistry) UpdateModelVersion(ctx contractapi.TransactionContextInterface, 
	modelID string, newHash string, metadataJSON string) error {
	
	// 获取现有模型信息
	existingModelJSON, err := ctx.GetStub().GetState(modelID)
	if err != nil {
		return fmt.Errorf("failed to get model: %v", err)
	}
	if existingModelJSON == nil {
		return fmt.Errorf("model %s does not exist", modelID)
	}
	
	var modelInfo ModelInfo
	if err := json.Unmarshal(existingModelJSON, &modelInfo); err != nil {
		return fmt.Errorf("failed to unmarshal model info: %v", err)
	}
	
	// 解析新元数据
	var newMetadata map[string]string
	if err := json.Unmarshal([]byte(metadataJSON), &newMetadata); err != nil {
		return fmt.Errorf("failed to parse metadata: %v", err)
	}
	
	// 更新模型信息
	modelInfo.ModelHash = newHash
	modelInfo.Metadata = newMetadata
	modelInfo.UpdatedAt = time.Now().Format(time.RFC3339)
	modelInfo.Version++
	
	// 保存更新后的模型信息
	updatedModelJSON, err := json.Marshal(modelInfo)
	if err != nil {
		return fmt.Errorf("failed to marshal updated model info: %v", err)
	}
	
	if err := ctx.GetStub().PutState(modelID, updatedModelJSON); err != nil {
		return fmt.Errorf("failed to put updated model state: %v", err)
	}
	
	// 记录新版本
	versionKey := fmt.Sprintf("%s_version_%d", modelID, modelInfo.Version)
	version := ModelVersion{
		Version:     modelInfo.Version,
		ModelHash:   newHash,
		Metadata:    newMetadata,
		Timestamp:   time.Now().Format(time.RFC3339),
		Transaction: ctx.GetStub().GetTxID(),
	}
	
	versionJSON, err := json.Marshal(version)
	if err != nil {
		return fmt.Errorf("failed to marshal version info: %v", err)
	}
	
	if err := ctx.GetStub().PutState(versionKey, versionJSON); err != nil {
		return fmt.Errorf("failed to put version state: %v", err)
	}
	
	return nil
}

// GetModelInfo 获取模型信息
func (mr *ModelRegistry) GetModelInfo(ctx contractapi.TransactionContextInterface, modelID string) (string, error) {
	modelJSON, err := ctx.GetStub().GetState(modelID)
	if err != nil {
		return "", fmt.Errorf("failed to get model: %v", err)
	}
	if modelJSON == nil {
		return "", fmt.Errorf("model %s does not exist", modelID)
	}
	
	return string(modelJSON), nil
}

// VerifyModelIntegrity 验证模型完整性
func (mr *ModelRegistry) VerifyModelIntegrity(ctx contractapi.TransactionContextInterface, 
	modelID string, currentHash string) (string, error) {
	
	modelJSON, err := ctx.GetStub().GetState(modelID)
	if err != nil {
		return "", fmt.Errorf("failed to get model: %v", err)
	}
	if modelJSON == nil {
		return "", fmt.Errorf("model %s does not exist", modelID)
	}
	
	var modelInfo ModelInfo
	if err := json.Unmarshal(modelJSON, &modelInfo); err != nil {
		return "", fmt.Errorf("failed to unmarshal model info: %v", err)
	}
	
	// 验证哈希是否匹配
	result := map[string]interface{}{
		"model_id":    modelID,
		"stored_hash": modelInfo.ModelHash,
		"current_hash": currentHash,
		"integrity_verified": modelInfo.ModelHash == currentHash,
		"verified_at": time.Now().Format(time.RFC3339),
	}
	
	resultJSON, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal result: %v", err)
	}
	
	return string(resultJSON), nil
}

// GetModelHistory 获取模型版本历史
func (mr *ModelRegistry) GetModelHistory(ctx contractapi.TransactionContextInterface, modelID string) (string, error) {
	// 获取所有版本
	versionIterator, err := ctx.GetStub().GetStateByPartialCompositeKey(modelID + "_version", []string{})
	if err != nil {
		return "", fmt.Errorf("failed to get version history: %v", err)
	}
	defer versionIterator.Close()
	
	var versions []ModelVersion
	for versionIterator.HasNext() {
		response, err := versionIterator.Next()
		if err != nil {
			return "", fmt.Errorf("failed to iterate versions: %v", err)
		}
		
		var version ModelVersion
		if err := json.Unmarshal(response.Value, &version); err != nil {
			return "", fmt.Errorf("failed to unmarshal version: %v", err)
		}
		
		versions = append(versions, version)
	}
	
	historyJSON, err := json.Marshal(versions)
	if err != nil {
		return "", fmt.Errorf("failed to marshal history: %v", err)
	}
	
	return string(historyJSON), nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&ModelRegistry{})
	if err != nil {
		fmt.Printf("Error creating model registry chaincode: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting model registry chaincode: %s", err.Error())
	}
}
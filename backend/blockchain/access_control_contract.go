package main

import (
	"encoding/json"
	"fmt"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"time"
)

// AccessControl 权限管理智能合约
type AccessControl struct {
	contractapi.Contract
}

// Permission 权限结构体
type Permission struct {
	ResourceID    string   `json:"resource_id"`
	Permission    string   `json:"permission"`
	GrantedTo     string   `json:"granted_to"`
	GrantedBy     string   `json:"granted_by"`
	GrantedAt     string   `json:"granted_at"`
	ExpiresAt     string   `json:"expires_at"`
	Status        string   `json:"status"` // active, revoked, expired
}

// Role 角色结构体
type Role struct {
	RoleID        string            `json:"role_id"`
	RoleName      string            `json:"role_name"`
	Permissions   []string          `json:"permissions"`
	Description   string            `json:"description"`
	CreatedAt     string            `json:"created_at"`
	UpdatedAt     string            `json:"updated_at"`
}

// AccessLog 访问日志结构体
type AccessLog struct {
	LogID         string            `json:"log_id"`
	UserID        string            `json:"user_id"`
	ResourceID    string            `json:"resource_id"`
	Action        string            `json:"action"`
	Timestamp     string            `json:"timestamp"`
	Status        string            `json:"status"` // success, failed, unauthorized
	Details       map[string]string `json:"details"`
}

// GrantPermission 授予权限
func (ac *AccessControl) GrantPermission(ctx contractapi.TransactionContextInterface, 
	resourceID string, permission string, grantedTo string, grantedBy string, expiresAt string) error {
	
	// 检查权限是否已存在
	permissionKey := fmt.Sprintf("%s_%s_%s", resourceID, permission, grantedTo)
	existingPermission, err := ctx.GetStub().GetState(permissionKey)
	if err != nil {
		return fmt.Errorf("failed to check existing permission: %v", err)
	}
	
	if existingPermission != nil {
		// 如果权限已存在，更新它
		var perm Permission
		if err := json.Unmarshal(existingPermission, &perm); err != nil {
			return fmt.Errorf("failed to unmarshal existing permission: %v", err)
		}
		perm.Status = "active"
		perm.GrantedAt = time.Now().UTC().Format(time.RFC3339)
		perm.ExpiresAt = expiresAt
		perm.GrantedBy = grantedBy
		
		updatedPermission, err := json.Marshal(perm)
		if err != nil {
			return fmt.Errorf("failed to marshal updated permission: %v", err)
		}
		
		return ctx.GetStub().PutState(permissionKey, updatedPermission)
	}
	
	// 创建新权限
	newPermission := Permission{
		ResourceID: resourceID,
		Permission: permission,
		GrantedTo:  grantedTo,
		GrantedBy:  grantedBy,
		GrantedAt:  time.Now().UTC().Format(time.RFC3339),
		ExpiresAt:  expiresAt,
		Status:     "active",
	}
	
	permissionBytes, err := json.Marshal(newPermission)
	if err != nil {
		return fmt.Errorf("failed to marshal permission: %v", err)
	}
	
	return ctx.GetStub().PutState(permissionKey, permissionBytes)
}

// RevokePermission 撤销权限
func (ac *AccessControl) RevokePermission(ctx contractapi.TransactionContextInterface, 
	resourceID string, permission string, grantedTo string) error {
	
	permissionKey := fmt.Sprintf("%s_%s_%s", resourceID, permission, grantedTo)
	existingPermission, err := ctx.GetStub().GetState(permissionKey)
	if err != nil {
		return fmt.Errorf("failed to get permission: %v", err)
	}
	
	if existingPermission == nil {
		return fmt.Errorf("permission does not exist")
	}
	
	var perm Permission
	if err := json.Unmarshal(existingPermission, &perm); err != nil {
		return fmt.Errorf("failed to unmarshal permission: %v", err)
	}
	
	// 更新状态为已撤销
	perm.Status = "revoked"
	perm.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	
	updatedPermission, err := json.Marshal(perm)
	if err != nil {
		return fmt.Errorf("failed to marshal updated permission: %v", err)
	}
	
	return ctx.GetStub().PutState(permissionKey, updatedPermission)
}

// CheckPermission 检查权限
func (ac *AccessControl) CheckPermission(ctx contractapi.TransactionContextInterface, 
	resourceID string, permission string, userID string) (bool, error) {
	
	permissionKey := fmt.Sprintf("%s_%s_%s", resourceID, permission, userID)
	permissionBytes, err := ctx.GetStub().GetState(permissionKey)
	if err != nil {
		return false, fmt.Errorf("failed to get permission: %v", err)
	}
	
	if permissionBytes == nil {
		return false, nil
	}
	
	var perm Permission
	if err := json.Unmarshal(permissionBytes, &perm); err != nil {
		return false, fmt.Errorf("failed to unmarshal permission: %v", err)
	}
	
	// 检查权限是否已撤销或过期
	if perm.Status == "revoked" {
		return false, nil
	}
	
	// 检查是否过期
	if perm.ExpiresAt != "" {
		expiryTime, err := time.Parse(time.RFC3339, perm.ExpiresAt)
		if err != nil {
			return false, fmt.Errorf("failed to parse expiry time: %v", err)
		}
		if time.Now().UTC().After(expiryTime) {
			// 权限已过期，更新状态
			perm.Status = "expired"
			updatedPermission, err := json.Marshal(perm)
			if err != nil {
				return false, fmt.Errorf("failed to marshal updated permission: %v", err)
			}
			ctx.GetStub().PutState(permissionKey, updatedPermission)
			return false, nil
		}
	}
	
	return perm.Status == "active", nil
}

// CreateRole 创建角色
func (ac *AccessControl) CreateRole(ctx contractapi.TransactionContextInterface, 
	roleID string, roleName string, permissions []string, description string) error {
	
	// 检查角色是否已存在
	existingRole, err := ctx.GetStub().GetState(roleID)
	if err != nil {
		return fmt.Errorf("failed to check existing role: %v", err)
	}
	
	if existingRole != nil {
		return fmt.Errorf("role already exists")
	}
	
	// 创建新角色
	newRole := Role{
		RoleID:      roleID,
		RoleName:    roleName,
		Permissions: permissions,
		Description: description,
		CreatedAt:   time.Now().UTC().Format(time.RFC3339),
		UpdatedAt:   time.Now().UTC().Format(time.RFC3339),
	}
	
	roleBytes, err := json.Marshal(newRole)
	if err != nil {
		return fmt.Errorf("failed to marshal role: %v", err)
	}
	
	return ctx.GetStub().PutState(roleID, roleBytes)
}

// AssignRole 分配角色给用户
func (ac *AccessControl) AssignRole(ctx contractapi.TransactionContextInterface, 
	userID string, roleID string, assignedBy string) error {
	
	// 检查角色是否存在
	roleBytes, err := ctx.GetStub().GetState(roleID)
	if err != nil {
		return fmt.Errorf("failed to get role: %v", err)
	}
	
	if roleBytes == nil {
		return fmt.Errorf("role does not exist")
	}
	
	var role Role
	if err := json.Unmarshal(roleBytes, &role); err != nil {
		return fmt.Errorf("failed to unmarshal role: %v", err)
	}
	
	// 为用户分配角色中的所有权限
	for _, perm := range role.Permissions {
		permissionKey := fmt.Sprintf("role_%s_%s_%s", roleID, perm, userID)
		newPermission := Permission{
			ResourceID: fmt.Sprintf("role_%s", roleID),
			Permission: perm,
			GrantedTo:  userID,
			GrantedBy:  assignedBy,
			GrantedAt:  time.Now().UTC().Format(time.RFC3339),
			ExpiresAt:  "", // 角色权限默认不过期
			Status:     "active",
		}
		
		permissionBytes, err := json.Marshal(newPermission)
		if err != nil {
			return fmt.Errorf("failed to marshal permission: %v", err)
		}
		
		if err := ctx.GetStub().PutState(permissionKey, permissionBytes); err != nil {
			return fmt.Errorf("failed to assign permission: %v", err)
		}
	}
	
	return nil
}

// LogAccess 记录访问日志
func (ac *AccessControl) LogAccess(ctx contractapi.TransactionContextInterface, 
	logID string, userID string, resourceID string, action string, status string, details map[string]string) error {
	
	accessLog := AccessLog{
		LogID:     logID,
		UserID:    userID,
		ResourceID: resourceID,
		Action:    action,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Status:    status,
		Details:   details,
	}
	
	logBytes, err := json.Marshal(accessLog)
	if err != nil {
		return fmt.Errorf("failed to marshal access log: %v", err)
	}
	
	// 使用复合键存储访问日志
	compositeKey, err := ctx.GetStub().CreateCompositeKey("accessLog", []string{userID, resourceID, logID})
	if err != nil {
		return fmt.Errorf("failed to create composite key: %v", err)
	}
	
	return ctx.GetStub().PutState(compositeKey, logBytes)
}

// GetAccessLogs 获取访问日志
func (ac *AccessControl) GetAccessLogs(ctx contractapi.TransactionContextInterface, 
	userID string, resourceID string) (string, error) {
	
	// 使用复合键查询访问日志
	iterator, err := ctx.GetStub().GetStateByPartialCompositeKey("accessLog", []string{userID, resourceID})
	if err != nil {
		return "", fmt.Errorf("failed to get access logs: %v", err)
	}
	defer iterator.Close()
	
	var logs []AccessLog
	for iterator.HasNext() {
		response, err := iterator.Next()
		if err != nil {
			return "", fmt.Errorf("failed to iterate access logs: %v", err)
		}
		
		var log AccessLog
		if err := json.Unmarshal(response.Value, &log); err != nil {
			return "", fmt.Errorf("failed to unmarshal access log: %v", err)
		}
		
		logs = append(logs, log)
	}
	
	logsJSON, err := json.Marshal(logs)
	if err != nil {
		return "", fmt.Errorf("failed to marshal access logs: %v", err)
	}
	
	return string(logsJSON), nil
}

// GetPermission 获取权限信息
func (ac *AccessControl) GetPermission(ctx contractapi.TransactionContextInterface, 
	resourceID string, permission string, grantedTo string) (string, error) {
	
	permissionKey := fmt.Sprintf("%s_%s_%s", resourceID, permission, grantedTo)
	permissionBytes, err := ctx.GetStub().GetState(permissionKey)
	if err != nil {
		return "", fmt.Errorf("failed to get permission: %v", err)
	}
	
	if permissionBytes == nil {
		return "", fmt.Errorf("permission does not exist")
	}
	
	return string(permissionBytes), nil
}

// GetRole 获取角色信息
func (ac *AccessControl) GetRole(ctx contractapi.TransactionContextInterface, roleID string) (string, error) {
	
	roleBytes, err := ctx.GetStub().GetState(roleID)
	if err != nil {
		return "", fmt.Errorf("failed to get role: %v", err)
	}
	
	if roleBytes == nil {
		return "", fmt.Errorf("role does not exist")
	}
	
	return string(roleBytes), nil
}

func main() {
	chaincode, err := contractapi.NewChaincode(&AccessControl{})
	if err != nil {
		fmt.Printf("Error creating access control chaincode: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting access control chaincode: %s", err.Error())
	}
}
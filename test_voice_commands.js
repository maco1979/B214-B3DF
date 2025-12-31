// 语音命令测试脚本 - 模拟浏览器语音识别API行为

// 模拟AIControl组件中的命令处理逻辑
function testVoiceCommandProcessing() {
    console.log("=== 语音命令处理测试 ===");
    
    // 模拟命令列表
    const testCommands = [
        "开启主控",
        "启动主控", 
        "激活主控",
        "关闭主控",
        "停止主控",
        "未知命令",
        "开启摄像头",  // 应该被忽略，因为现在只支持主控命令
        "关闭灌溉系统"   // 应该被忽略
    ];
    
    // 模拟命令处理函数
    function handleVoiceCommand(command) {
        console.log(`\n处理命令: "${command}"`);
        
        const normalizedCommand = command.toLowerCase().trim();
        
        // 检查命令是否匹配主控相关指令
        if (normalizedCommand.includes('开启主控') || 
            normalizedCommand.includes('启动主控') || 
            normalizedCommand.includes('激活主控')) {
            console.log(`   ✅ 命令匹配: 激活AI主控`);
            return { action: 'activate_master_control', command };
        } 
        else if (normalizedCommand.includes('关闭主控') || 
                 normalizedCommand.includes('停止主控')) {
            console.log(`   ✅ 命令匹配: 关闭AI主控`);
            return { action: 'deactivate_master_control', command };
        } 
        else {
            console.log(`   ❌ 命令不匹配: 忽略未知命令`);
            return { action: 'unknown', command };
        }
    }
    
    // 执行所有测试命令
    const results = [];
    testCommands.forEach(command => {
        const result = handleVoiceCommand(command);
        results.push(result);
    });
    
    // 统计结果
    console.log("\n=== 测试结果统计 ===");
    const activatedCommands = results.filter(r => r.action === 'activate_master_control').length;
    const deactivatedCommands = results.filter(r => r.action === 'deactivate_master_control').length;
    const unknownCommands = results.filter(r => r.action === 'unknown').length;
    
    console.log(`总测试命令数: ${testCommands.length}`);
    console.log(`激活命令数: ${activatedCommands}`);
    console.log(`关闭命令数: ${deactivatedCommands}`);
    console.log(`未知命令数: ${unknownCommands}`);
    
    // 验证结果
    if (activatedCommands === 3 && deactivatedCommands === 2 && unknownCommands === 2) {
        console.log("✅ 所有测试通过! 语音命令处理逻辑正常工作。");
        return true;
    } else {
        console.log("❌ 测试失败! 命令处理逻辑存在问题。");
        return false;
    }
}

// 运行测试
if (typeof window === 'undefined') {
    // Node.js环境
    testVoiceCommandProcessing();
} else {
    // 浏览器环境
    document.addEventListener('DOMContentLoaded', () => {
        testVoiceCommandProcessing();
    });
}
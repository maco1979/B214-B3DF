"""
分布式DCNN系统测试
测试联邦学习、边缘计算、区块链奖励的集成功能
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from distributed_dcnn.main import DistributedDCNNApplication
from distributed_dcnn.config import DistributedDCNNConfig


class DistributedDCNNTestSuite:
    """分布式DCNN测试套件"""
    
    def __init__(self):
        self.test_results = []
        self.app = None
    
    async def setup(self):
        """设置测试环境"""
        print("=== 分布式DCNN系统测试 ===")
        print("设置测试环境...")
        
        # 创建测试配置
        config = DistributedDCNNConfig()
        
        # 修改配置为测试环境
        config.system['log_level'] = 'DEBUG'
        config.system['max_concurrent_tasks'] = 3
        
        # 创建应用实例
        self.app = DistributedDCNNApplication(config.to_dict())
        
        print("测试环境设置完成")
    
    async def test_system_startup(self):
        """测试系统启动"""
        print("\n1. 测试系统启动...")
        
        start_time = time.time()
        
        try:
            await self.app.startup()
            startup_time = time.time() - start_time
            
            # 检查系统状态
            status = await self.app.get_system_status()
            
            if status['is_running']:
                result = {
                    'test': 'system_startup',
                    'status': 'PASS',
                    'startup_time': startup_time,
                    'details': status
                }
                print(f"✓ 系统启动成功 - 耗时: {startup_time:.2f}秒")
            else:
                result = {
                    'test': 'system_startup', 
                    'status': 'FAIL',
                    'error': '系统未正确启动'
                }
                print("✗ 系统启动失败")
                
        except Exception as e:
            result = {
                'test': 'system_startup',
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"✗ 系统启动错误: {e}")
        
        self.test_results.append(result)
        return result
    
    async def test_image_processing(self):
        """测试图像处理功能"""
        print("\n2. 测试图像处理...")
        
        # 模拟图像数据
        test_data = {
            'batch_id': 'test_batch_001',
            'images': [
                {
                    'id': 'img_001',
                    'type': 'agricultural',
                    'size': (224, 224),
                    'format': 'JPEG'
                },
                {
                    'id': 'img_002', 
                    'type': 'agricultural',
                    'size': (224, 224),
                    'format': 'JPEG'
                },
                {
                    'id': 'img_003',
                    'type': 'agricultural', 
                    'size': (224, 224),
                    'format': 'JPEG'
                }
            ],
            'metadata': {
                'source': 'agricultural_drone',
                'location': '北京测试农场',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        start_time = time.time()
        
        try:
            result = await self.app.process_image_batch(test_data)
            processing_time = time.time() - start_time
            
            if result['success']:
                test_result = {
                    'test': 'image_processing',
                    'status': 'PASS',
                    'processing_time': processing_time,
                    'inference_count': len(result['inference_results']),
                    'reward_distributed': len(result['reward_distribution'])
                }
                print(f"✓ 图像处理成功 - 耗时: {processing_time:.2f}秒")
                print(f"  推理结果: {len(result['inference_results'])} 个")
                print(f"  奖励分配: {len(result['reward_distribution'])} 个节点")
            else:
                test_result = {
                    'test': 'image_processing',
                    'status': 'FAIL',
                    'error': result.get('error', '未知错误')
                }
                print("✗ 图像处理失败")
                
        except Exception as e:
            test_result = {
                'test': 'image_processing',
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"✗ 图像处理错误: {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    async def test_federated_learning(self):
        """测试联邦学习功能"""
        print("\n3. 测试联邦学习...")
        
        try:
            # 获取联邦学习状态
            status = await self.app.get_system_status()
            federated_status = status['federated_status']
            
            if federated_status['is_active']:
                result = {
                    'test': 'federated_learning',
                    'status': 'PASS',
                    'active_nodes': federated_status['active_nodes'],
                    'learning_rounds': federated_status['current_round']
                }
                print(f"✓ 联邦学习正常 - 活跃节点: {federated_status['active_nodes']}")
                print(f"  学习轮次: {federated_status['current_round']}")
            else:
                result = {
                    'test': 'federated_learning',
                    'status': 'FAIL',
                    'error': '联邦学习未激活'
                }
                print("✗ 联邦学习未激活")
                
        except Exception as e:
            result = {
                'test': 'federated_learning',
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"✗ 联邦学习错误: {e}")
        
        self.test_results.append(result)
        return result
    
    async def test_blockchain_rewards(self):
        """测试区块链奖励功能"""
        print("\n4. 测试区块链奖励...")
        
        try:
            # 获取奖励系统状态
            status = await self.app.get_system_status()
            reward_status = status['reward_status']
            
            if reward_status['is_connected']:
                result = {
                    'test': 'blockchain_rewards',
                    'status': 'PASS',
                    'reward_pool': reward_status['reward_pool'],
                    'distributed_rewards': reward_status['distributed_rewards']
                }
                print(f"✓ 区块链奖励正常 - 奖励池: {reward_status['reward_pool']} PHOTON")
                print(f"  已分发奖励: {reward_status['distributed_rewards']} PHOTON")
            else:
                result = {
                    'test': 'blockchain_rewards',
                    'status': 'FAIL',
                    'error': '区块链连接失败'
                }
                print("✗ 区块链连接失败")
                
        except Exception as e:
            result = {
                'test': 'blockchain_rewards',
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"✗ 区块链奖励错误: {e}")
        
        self.test_results.append(result)
        return result
    
    async def test_system_shutdown(self):
        """测试系统关闭"""
        print("\n5. 测试系统关闭...")
        
        start_time = time.time()
        
        try:
            await self.app.shutdown()
            shutdown_time = time.time() - start_time
            
            # 检查系统状态
            status = await self.app.get_system_status()
            
            if not status['is_running']:
                result = {
                    'test': 'system_shutdown',
                    'status': 'PASS',
                    'shutdown_time': shutdown_time
                }
                print(f"✓ 系统关闭成功 - 耗时: {shutdown_time:.2f}秒")
            else:
                result = {
                    'test': 'system_shutdown',
                    'status': 'FAIL',
                    'error': '系统未正确关闭'
                }
                print("✗ 系统关闭失败")
                
        except Exception as e:
            result = {
                'test': 'system_shutdown',
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"✗ 系统关闭错误: {e}")
        
        self.test_results.append(result)
        return result
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n=== 测试报告 ===")
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        error_tests = len([r for r in self.test_results if r['status'] == 'ERROR'])
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"错误: {error_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        # 详细结果
        print("\n详细结果:")
        for result in self.test_results:
            status_icon = '✓' if result['status'] == 'PASS' else '✗'
            print(f"{status_icon} {result['test']}: {result['status']}")
            if 'error' in result:
                print(f"   错误: {result['error']}")
        
        # 保存报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': passed_tests/total_tests*100
            },
            'detailed_results': self.test_results
        }
        
        with open('distributed_dcnn_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n测试报告已保存到: distributed_dcnn_test_report.json")
        
        return report


async def run_all_tests():
    """运行所有测试"""
    test_suite = DistributedDCNNTestSuite()
    
    try:
        # 设置测试环境
        await test_suite.setup()
        
        # 运行测试用例
        await test_suite.test_system_startup()
        await test_suite.test_image_processing()
        await test_suite.test_federated_learning()
        await test_suite.test_blockchain_rewards()
        await test_suite.test_system_shutdown()
        
        # 生成报告
        report = test_suite.generate_test_report()
        
        # 总体评估
        success_rate = report['summary']['success_rate']
        if success_rate >= 80:
            print("\n🎯 测试结果: 优秀 - 分布式DCNN系统运行正常")
        elif success_rate >= 60:
            print("\n✅ 测试结果: 良好 - 系统基本功能正常")
        else:
            print("\n⚠️ 测试结果: 需要改进 - 部分功能存在问题")
        
        return report
        
    except Exception as e:
        print(f"测试执行异常: {e}")
        return None


if __name__ == "__main__":
    print("分布式DCNN系统完整性测试")
    print("=" * 50)
    
    asyncio.run(run_all_tests())
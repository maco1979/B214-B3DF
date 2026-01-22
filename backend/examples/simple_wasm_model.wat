;; 简单的WASM模型示例
;; 这是一个演示性的线性回归模型

(module
  ;; 内存配置
  (memory (export "memory") 1)
  
  ;; 全局变量 - 模型参数 (斜率m和截距b)
  (global $m f32 (f32.const 2.0))
  (global $b f32 (f32.const 1.0))
  
  ;; 推理函数 - 线性回归: y = m*x + b
  (func $inference (export "inference") 
    (param $input_ptr i32) (param $input_len i32) (result i32)
    
    (local $i i32)
    (local $x f32)
    (local $y f32)
    (local $output_ptr i32)
    
    ;; 检查输入长度
    (if (i32.eqz (local.get $input_len))
      (then
        (return (i32.const -1))  ;; 错误: 输入为空
      )
    )
    
    ;; 分配输出内存 (每个输入对应一个输出)
    (local.set $output_ptr 
      (i32.mul (local.get $input_len) (i32.const 4))  ;; 每个输出4字节
    )
    
    ;; 处理每个输入
    (local.set $i (i32.const 0))
    (loop $process_loop
      ;; 读取输入值
      (local.set $x 
        (f32.load (local.get $input_ptr))
      )
      
      ;; 计算: y = m*x + b
      (local.set $y 
        (f32.add
          (f32.mul (global.get $m) (local.get $x))
          (global.get $b)
        )
      )
      
      ;; 存储结果
      (f32.store 
        (i32.add (local.get $output_ptr) (i32.mul (local.get $i) (i32.const 4)))
        (local.get $y)
      )
      
      ;; 移动输入指针
      (local.set $input_ptr (i32.add (local.get $input_ptr) (i32.const 4)))
      
      ;; 递增计数器
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      
      ;; 检查是否完成
      (br_if $process_loop (i32.lt_u (local.get $i) (local.get $input_len)))
    )
    
    ;; 返回输出指针
    (local.get $output_ptr)
  )
  
  ;; 批量推理函数
  (func $batch_inference (export "batch_inference")
    (param $batch_ptr i32) (param $batch_size i32) (param $input_len i32) (result i32)
    
    (local $i i32)
    (local $current_ptr i32)
    (local $output_ptr i32)
    
    ;; 计算总输出大小
    (local.set $output_ptr 
      (i32.mul 
        (i32.mul (local.get $batch_size) (local.get $input_len))
        (i32.const 4)
      )
    )
    
    ;; 处理每个批次
    (local.set $i (i32.const 0))
    (local.set $current_ptr (local.get $batch_ptr))
    
    (loop $batch_loop
      ;; 调用单次推理
      (call $inference 
        (local.get $current_ptr)
        (local.get $input_len)
      )
      
      ;; 移动到下一个批次
      (local.set $current_ptr 
        (i32.add 
          (local.get $current_ptr)
          (i32.mul (local.get $input_len) (i32.const 4))
        )
      )
      
      ;; 递增计数器
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      
      ;; 检查是否完成
      (br_if $batch_loop (i32.lt_u (local.get $i) (local.get $batch_size)))
    )
    
    ;; 返回输出指针
    (local.get $output_ptr)
  )
  
  ;; 模型信息函数
  (func $get_model_info (export "get_model_info") (result i32)
    ;; 返回模型信息字符串的指针
    ;; 这里简化处理，实际应该返回JSON字符串
    (i32.const 1000)  ;; 假设模型信息存储在内存位置1000
  )
  
  ;; 初始化函数
  (func $init (export "init")
    ;; 初始化模型参数
    ;; 这里可以加载预训练的参数
    nop
  )
  
  ;; 模型元数据
  (data (i32.const 1000) 
    "{\"name\":\"linear_regression\",\"version\":\"1.0.0\",\"input_size\":1,\"output_size\":1}"
  )
)
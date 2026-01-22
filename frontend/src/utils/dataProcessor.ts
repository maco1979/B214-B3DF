/**
 * 数据处理工具
 * 用于大体积数据传输的专项优化
 */

/**
 * 扁平化数据配置
 */
export interface FlattenConfig {

  /** 是否保留维度信息 */
  preserveDimensions?: boolean

  /** 维度分隔符 */
  dimensionSeparator?: string
}

/**
 * 数据裁剪配置
 */
export interface DataClipConfig {

  /** 保留的小数位数 */
  precision?: number

  /** 最大值 */
  maxValue?: number

  /** 最小值 */
  minValue?: number
}

/**
 * 数据处理工具类
 */
export class DataProcessor {
  /**
   * 将嵌套数组扁平化为一维数组
   * @param data 嵌套数组
   * @param config 配置
   * @returns 扁平化结果
   */
  static flatten<T>(
    data: T[][][] | T[][],
    config: FlattenConfig = {},
  ): { data: number[]; dimensions?: number[] } {
    const {
      preserveDimensions = true,
      dimensionSeparator = '_',
    } = config;

    // 确定数据维度
    const dimensions: number[] = [];
    let current: any = data;
    while (Array.isArray(current)) {
      dimensions.push(current.length);
      current = current[0];
    }

    // 将数据扁平化为一维数组
    const flattenedData = this.flattenArray(data);

    return preserveDimensions ?
      { data: flattenedData, dimensions } :
      { data: flattenedData };
  }

  /**
   * 递归扁平化数组
   * @param data 嵌套数组
   * @returns 一维数组
   */
  private static flattenArray<T>(data: any): number[] {
    if (typeof data === 'number') {
      return [data];
    }

    if (Array.isArray(data)) {
      return data.reduce<number[]>((acc, item) => acc.concat(this.flattenArray(item)), []);
    }

    return [];
  }

  /**
   * 将一维数组还原为嵌套数组
   * @param data 一维数组
   * @param dimensions 维度信息
   * @returns 嵌套数组
   */
  static unflatten<T>(
    data: number[],
    dimensions: number[],
  ): T[][][] | T[][] | T[] {
    if (dimensions.length === 0) {
      return data as any;
    }

    if (dimensions.length === 1) {
      return data as any;
    }

    if (dimensions.length === 2) {
      return this.unflatten2D(data, dimensions[0], dimensions[1]) as any;
    }

    if (dimensions.length === 3) {
      return this.unflatten3D(data, dimensions[0], dimensions[1], dimensions[2]) as any;
    }

    // 支持更高维度
    return this.unflattenND(data, dimensions);
  }

  /**
   * 将一维数组还原为二维数组
   * @param data 一维数组
   * @param rows 行数
   * @param cols 列数
   * @returns 二维数组
   */
  private static unflatten2D<T>(data: number[], rows: number, cols: number): T[][] {
    const result: T[][] = [];
    let index = 0;

    for (let i = 0; i < rows; i++) {
      const row: T[] = [];
      for (let j = 0; j < cols; j++) {
        row.push(data[index++] as T);
      }
      result.push(row);
    }

    return result;
  }

  /**
   * 将一维数组还原为三维数组
   * @param data 一维数组
   * @param depth 深度
   * @param rows 行数
   * @param cols 列数
   * @returns 三维数组
   */
  private static unflatten3D<T>(data: number[], depth: number, rows: number, cols: number): T[][][] {
    const result: T[][][] = [];
    let index = 0;

    for (let i = 0; i < depth; i++) {
      const layer: T[][] = [];
      for (let j = 0; j < rows; j++) {
        const row: T[] = [];
        for (let k = 0; k < cols; k++) {
          row.push(data[index++] as T);
        }
        layer.push(row);
      }
      result.push(layer);
    }

    return result;
  }

  /**
   * 将一维数组还原为N维数组
   * @param data 一维数组
   * @param dimensions 维度信息
   * @returns N维数组
   */
  private static unflattenND<T>(data: number[], dimensions: number[]): any {
    if (dimensions.length === 0) {
      return data[0] as T;
    }

    const result: any[] = [];
    const currentDim = dimensions[0];
    const nextDimensions = dimensions.slice(1);
    const chunkSize = data.length / currentDim;

    for (let i = 0; i < currentDim; i++) {
      const chunk = data.slice(i * chunkSize, (i + 1) * chunkSize);
      result.push(this.unflattenND(chunk, nextDimensions));
    }

    return result;
  }

  /**
   * 裁剪数据精度
   * @param data 数据
   * @param config 配置
   * @returns 裁剪后的数据
   */
  static clipData<T>(
    data: T | T[] | T[][],
    config: DataClipConfig = {},
  ): T | T[] | T[][] {
    const {
      precision = 3,
      maxValue,
      minValue,
    } = config;

    // 递归处理数据
    const processValue = (value: any): any => {
      if (typeof value === 'number') {
        // 裁剪精度
        let processedValue = Number(value.toFixed(precision));

        // 裁剪最大值和最小值
        if (maxValue !== undefined) {
          processedValue = Math.min(processedValue, maxValue);
        }
        if (minValue !== undefined) {
          processedValue = Math.max(processedValue, minValue);
        }

        return processedValue;
      }

      if (Array.isArray(value)) {
        return value.map(processValue);
      }

      return value;
    };

    return processValue(data);
  }

  /**
   * 检查浏览器是否支持CompressionStream API
   * @returns 是否支持
   */
  static isCompressionSupported(): boolean {
    return typeof CompressionStream !== 'undefined';
  }

  /**
   * 使用原生CompressionStream压缩数据
   * @param data 要压缩的数据
   * @param config 压缩配置
   * @returns 压缩结果
   */
  private static async compressWithStream(
    data: any,
    config: CompressionConfig = {},
  ): Promise<CompressionResult> {
    const {
      algorithm = 'gzip',
      level = 6,
      preserveMetadata = true,
    } = config;

    const startTime = performance.now();

    // 将数据转换为字符串
    const jsonString = JSON.stringify(data);
    const originalBytes = new TextEncoder().encode(jsonString);
    const originalSize = originalBytes.length;

    try {
      // 创建压缩流
      const compressionStream = new CompressionStream(algorithm, {
        level,
      });

      // 创建可写流和可读流
      const writer = compressionStream.writable.getWriter();
      const reader = compressionStream.readable.getReader();

      // 写入数据
      writer.write(originalBytes);
      await writer.close();

      // 读取压缩结果
      const chunks: Uint8Array[] = [];
      let done = false;
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        if (value) {
          chunks.push(value);
        }
        done = readerDone;
      }

      // 合并压缩后的字节
      const compressedBytes = new Uint8Array(
        chunks.reduce((acc, chunk) => acc + chunk.length, 0),
      );

      let offset = 0;
      for (const chunk of chunks) {
        compressedBytes.set(chunk, offset);
        offset += chunk.length;
      }

      // 转换为base64字符串
      const compressedData = btoa(String.fromCharCode(...compressedBytes));
      const compressedSize = compressedBytes.length;
      const timeSpent = performance.now() - startTime;

      // 计算压缩率
      const compressionRatio = originalSize > 0 ?
        Math.round((1 - compressedSize / originalSize) * 100) :
        0;

      const result: CompressionResult = {
        data: compressedData,
        originalSize,
        compressedSize,
        compressionRatio,
        timeSpent,
      };

      if (preserveMetadata) {
        // 如果需要保留元数据，将元数据作为前缀添加到压缩数据中
        const metadata = JSON.stringify({
          originalSize,
          compressedSize,
          compressionRatio,
          algorithm,
          level,
          timestamp: Date.now(),
        });
        const metadataBytes = new TextEncoder().encode(metadata);
        const metadataBase64 = btoa(String.fromCharCode(...metadataBytes));
        result.data = `${metadataBase64}:::${compressedData}`;
      }

      return result;
    } catch (error) {
      console.error('压缩失败:', error);
      // 压缩失败时，返回原始数据的JSON字符串
      return {
        data: jsonString,
        originalSize,
        compressedSize: originalSize,
        compressionRatio: 0,
        timeSpent: performance.now() - startTime,
      };
    }
  }

  /**
   * 使用原生DecompressionStream解压数据
   * @param compressedData 压缩后的数据
   * @returns 解压结果
   */
  private static async decompressWithStream(
    compressedData: string,
  ): Promise<any> {
    const startTime = performance.now();

    try {
      let metadata = null;
      let actualCompressedData = compressedData;

      // 检查是否包含元数据
      if (compressedData.includes(':::')) {
        const [metadataBase64, data] = compressedData.split(':::');
        actualCompressedData = data;

        // 解析元数据
        const metadataBytes = Uint8Array.from(atob(metadataBase64), c => c.charCodeAt(0));
        const metadataString = new TextDecoder().decode(metadataBytes);
        metadata = JSON.parse(metadataString);
      }

      // 解码base64字符串为Uint8Array
      const compressedBytes = Uint8Array.from(atob(actualCompressedData), c => c.charCodeAt(0));

      // 检测压缩算法（简化实现，实际可以从元数据获取）
      let algorithm = 'gzip';
      if (compressedBytes[0] === 0x78) {
        algorithm = 'deflate';
      }

      // 创建解压缩流
      const decompressionStream = new DecompressionStream(algorithm);

      // 创建可写流和可读流
      const writer = decompressionStream.writable.getWriter();
      const reader = decompressionStream.readable.getReader();

      // 写入压缩数据
      writer.write(compressedBytes);
      await writer.close();

      // 读取解压结果
      const chunks: Uint8Array[] = [];
      let done = false;
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        if (value) {
          chunks.push(value);
        }
        done = readerDone;
      }

      // 合并解压后的字节
      const decompressedBytes = new Uint8Array(
        chunks.reduce((acc, chunk) => acc + chunk.length, 0),
      );

      let offset = 0;
      for (const chunk of chunks) {
        decompressedBytes.set(chunk, offset);
        offset += chunk.length;
      }

      // 转换为字符串并解析为JSON
      const decompressedString = new TextDecoder().decode(decompressedBytes);
      const result = JSON.parse(decompressedString);

      return result;
    } catch (error) {
      console.error('解压失败:', error);
      // 解压失败时，尝试直接解析为JSON
      try {
        return JSON.parse(compressedData);
      } catch (jsonError) {
        throw new Error('解压失败且无法解析为JSON');
      }
    }
  }

  /**
   * 压缩数据（客户端使用，配合后端的压缩）
   * @param data 数据
   * @param config 压缩配置
   * @returns 压缩后的数据或压缩结果
   */
  static async compressData(
    data: any,
    config: CompressionConfig = {},
  ): Promise<string | CompressionResult> {
    // 检查浏览器是否支持CompressionStream API
    if (this.isCompressionSupported()) {
      const result = await this.compressWithStream(data, config);
      return config.preserveMetadata ? result.data : result;
    }
      // 不支持CompressionStream API时，降级为简单的JSON序列化
      const jsonString = JSON.stringify(data);
      const originalSize = new TextEncoder().encode(jsonString).length;

      const result: CompressionResult = {
        data: jsonString,
        originalSize,
        compressedSize: originalSize,
        compressionRatio: 0,
        timeSpent: 0,
      };

      return config.preserveMetadata ? result.data : result;
  }

  /**
   * 解压数据（客户端使用，配合后端的压缩）
   * @param compressedData 压缩后的数据
   * @returns 解压后的数据
   */
  static async decompressData<T>(compressedData: string): Promise<T> {
    // 检查浏览器是否支持DecompressionStream API
    if (this.isCompressionSupported()) {
      return this.decompressWithStream(compressedData) as Promise<T>;
    }
      // 不支持DecompressionStream API时，降级为简单的JSON解析
      try {
        return JSON.parse(compressedData) as T;
      } catch (error) {
        console.error('解压失败:', error);
        throw new Error('解压失败且无法解析为JSON');
      }
  }

  /**
   * 归一化数据到0-1区间
   * @param data 数据
   * @returns 归一化后的数据
   */
  static normalize<T>(data: T[] | T[][] | T[][][]): T[] | T[][] | T[][][] {
    // 递归处理数据
    const processArray = (arr: any[]): any[] => {
      if (arr.length === 0) {
        return arr;
      }

      // 检查是否是二维数组
      if (Array.isArray(arr[0])) {
        return arr.map(processArray);
      }

      // 检查是否是数值数组
      if (arr.every(item => typeof item === 'number')) {
        const min = Math.min(...(arr));
        const max = Math.max(...(arr));
        const range = max - min;

        if (range === 0) {
          // 如果所有值都相同，返回全1数组
          return arr.map(() => 1);
        }

        // 归一化到0-1区间
        return arr.map((value: number) => (value - min) / range);
      }

      return arr;
    };

    return processArray(data as any);
  }

  /**
   * 标准化数据（Z-Score）
   * @param data 数据
   * @returns 标准化后的数据
   */
  static standardize<T>(data: T[] | T[][] | T[][][]): T[] | T[][] | T[][][] {
    // 递归处理数据
    const processArray = (arr: any[]): any[] => {
      if (arr.length === 0) {
        return arr;
      }

      // 检查是否是二维数组
      if (Array.isArray(arr[0])) {
        return arr.map(processArray);
      }

      // 检查是否是数值数组
      if (arr.every(item => typeof item === 'number')) {
        const numbers = arr;
        const mean = numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
        const variance = numbers.reduce((sum, value) => sum + (value - mean) ** 2, 0) / numbers.length;
        const stdDev = Math.sqrt(variance);

        if (stdDev === 0) {
          // 如果标准差为0，返回全0数组
          return numbers.map(() => 0);
        }

        // 标准化（Z-Score）
        return numbers.map(value => (value - mean) / stdDev);
      }

      return arr;
    };

    return processArray(data as any);
  }
}

/**
 * 压缩配置
 */
export interface CompressionConfig {

  /** 压缩算法，支持gzip和deflate */
  algorithm?: 'gzip' | 'deflate';

  /** 压缩级别，0-9，0表示无压缩，9表示最大压缩 */
  level?: 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

  /** 是否保留元数据 */
  preserveMetadata?: boolean;
}

/**
 * 压缩结果
 */
export interface CompressionResult {

  /** 压缩后的数据 */
  data: string;

  /** 压缩前大小（字节） */
  originalSize: number;

  /** 压缩后大小（字节） */
  compressedSize: number;

  /** 压缩率（百分比） */
  compressionRatio: number;

  /** 压缩耗时（毫秒） */
  timeSpent: number;
}

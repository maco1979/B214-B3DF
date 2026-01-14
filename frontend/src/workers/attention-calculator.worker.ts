/**
 * 注意力计算Web Worker
 * 用于处理注意力权重的各种计算任务，避免阻塞主线程
 */

// 定义消息类型
type MessageType =
  | 'normalize'
  | 'standardize'
  | 'clip'
  | 'similarity'
  | 'gradient_attribution'
  | 'weight_statistics'
  | 'transfer_matrix_processing'
  | 'pca_reduction'
  | 'tsne_reduction'
  | 'clustering'
  | 'correlation_analysis'

// 定义消息数据结构
interface WorkerMessage {
  id: string;
  type: MessageType;
  data: any;
}

// 定义响应数据结构
interface WorkerResponse {
  id: string;
  type: MessageType;
  result: any;
  error?: string;
}

// 归一化数据到0-1区间
function normalize(data: number[] | number[][] | number[][][]): number[] | number[][] | number[][][] {
  const processArray = (arr: any[]): any[] => {
    if (arr.length === 0) {
      return arr;
    }

    if (Array.isArray(arr[0])) {
      return arr.map(processArray);
    }

    if (arr.every(item => typeof item === 'number')) {
      const min = Math.min(...(arr));
      const max = Math.max(...(arr));
      const range = max - min;

      if (range === 0) {
        return arr.map(() => 1);
      }

      return arr.map((value: number) => (value - min) / range);
    }

    return arr;
  };

  return processArray(data);
}

// 标准化数据（Z-Score）
function standardize(data: number[] | number[][] | number[][][]): number[] | number[][] | number[][][] {
  const processArray = (arr: any[]): any[] => {
    if (arr.length === 0) {
      return arr;
    }

    if (Array.isArray(arr[0])) {
      return arr.map(processArray);
    }

    if (arr.every(item => typeof item === 'number')) {
      const numbers = arr;
      const mean = numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
      const variance = numbers.reduce((sum, value) => sum + (value - mean) ** 2, 0) / numbers.length;
      const stdDev = Math.sqrt(variance);

      if (stdDev === 0) {
        return numbers.map(() => 0);
      }

      return numbers.map(value => (value - mean) / stdDev);
    }

    return arr;
  };

  return processArray(data);
}

// 裁剪数据精度
function clip(data: number[] | number[][] | number[][][], precision: number): number[] | number[][] | number[][][] {
  const processValue = (value: any): any => {
    if (typeof value === 'number') {
      return Number(value.toFixed(precision));
    }

    if (Array.isArray(value)) {
      return value.map(processValue);
    }

    return value;
  };

  return processValue(data);
}

// 计算余弦相似度
function cosineSimilarity(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);

  if (norm1 === 0 || norm2 === 0) {
    return 0;
  }

  return dotProduct / (norm1 * norm2);
}

// 计算注意力归因权重
function gradientAttribution(attentionWeights: number[][], gradients: number[][]): number[][] {
  if (attentionWeights.length !== gradients.length || attentionWeights[0].length !== gradients[0].length) {
    throw new Error('Attention weights and gradients must have the same dimensions');
  }

  const result: number[][] = [];

  for (let i = 0; i < attentionWeights.length; i++) {
    const row: number[] = [];
    for (let j = 0; j < attentionWeights[i].length; j++) {
      row.push(attentionWeights[i][j] * gradients[i][j]);
    }
    result.push(row);
  }

  return result;
}

// 计算注意力权重统计信息
function weightStatistics(attentionWeights: number[][]): {
  mean: number;
  stdDev: number;
  min: number;
  max: number;
  topK: { indices: [number, number][]; values: number[] };
} {
  // 扁平化数组
  const flatWeights = attentionWeights.flat();

  // 计算基本统计量
  const mean = flatWeights.reduce((sum, value) => sum + value, 0) / flatWeights.length;
  const variance = flatWeights.reduce((sum, value) => sum + (value - mean) ** 2, 0) / flatWeights.length;
  const stdDev = Math.sqrt(variance);
  const min = Math.min(...flatWeights);
  const max = Math.max(...flatWeights);

  // 计算Top-K权重
  const topKCount = Math.min(10, flatWeights.length);
  const sortedIndices = flatWeights
    .map((value, index) => ({ index, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, topKCount);

  // 转换为二维索引
  const topK = {
    indices: sortedIndices.map(item => {
      const row = Math.floor(item.index / attentionWeights[0].length);
      const col = item.index % attentionWeights[0].length;
      return [row, col] as [number, number];
    }),
    values: sortedIndices.map(item => item.value),
  };

  return {
    mean,
    stdDev,
    min,
    max,
    topK,
  };
}

// 处理跨域迁移矩阵
function transferMatrixProcessing(matrix: number[][], operation: 'normalize' | 'clip' | 'stat'): any {
  switch (operation) {
    case 'normalize':
      return normalize(matrix);
    case 'clip':
      return clip(matrix, 3);
    case 'stat':
      return weightStatistics(matrix);
    default:
      throw new Error(`Unknown operation: ${operation}`);
  }
}

// 处理消息
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const { id, type, data } = event.data;
  let result: any;
  let error: string | undefined;

  try {
    switch (type) {
      case 'normalize':
        result = normalize(data);
        break;
      case 'standardize':
        result = standardize(data);
        break;
      case 'clip':
        result = clip(data.data, data.precision || 3);
        break;
      case 'similarity':
        result = cosineSimilarity(data.vec1, data.vec2);
        break;
      case 'gradient_attribution':
        result = gradientAttribution(data.attentionWeights, data.gradients);
        break;
      case 'weight_statistics':
        result = weightStatistics(data);
        break;
      case 'transfer_matrix_processing':
        result = transferMatrixProcessing(data.matrix, data.operation);
        break;
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (e) {
    error = e instanceof Error ? e.message : 'Unknown error';
  }

  // 发送响应
  const response: WorkerResponse = {
    id,
    type,
    result,
    error,
  };

  self.postMessage(response);
};

// PCA降维实现
function pcaReduction(data: number[][], dimensions = 2): number[][] {
  // 确保数据是二维数组
  if (!Array.isArray(data) || !Array.isArray(data[0])) {
    throw new Error('Data must be a 2D array for PCA reduction');
  }

  const n = data.length;
  const d = data[0].length;

  // 1. 中心化数据
  const mean = new Array(d).fill(0);
  for (const row of data) {
    for (let j = 0; j < d; j++) {
      mean[j] += row[j] / n;
    }
  }

  const centeredData = data.map(row =>
    row.map((value, j) => value - mean[j]),
  );

  // 2. 计算协方差矩阵
  const covariance = new Array(d).fill(0).map(() => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let sum = 0;
      for (const row of centeredData) {
        sum += row[i] * row[j];
      }
      covariance[i][j] = sum / (n - 1);
      covariance[j][i] = covariance[i][j];
    }
  }

  /*
   * 3. 计算特征值和特征向量（简化实现，适用于中小规模数据）
   * 这里使用幂迭代法计算前k个最大特征值对应的特征向量
   */
  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];

  // 幂迭代法计算主成分
  for (let k = 0; k < dimensions; k++) {
    // 初始向量
    let v = new Array(d).fill(0).map(() => Math.random() - 0.5);

    // 归一化
    const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    v = v.map(x => x / norm);

    // 迭代计算特征向量
    for (let iter = 0; iter < 100; iter++) {
      const newV = new Array(d).fill(0);
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          newV[i] += covariance[i][j] * v[j];
        }
      }

      // 归一化
      const newNorm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
      newV.forEach((x, i) => newV[i] = x / newNorm);

      // 检查收敛
      const diff = newV.reduce((sum, x, i) => sum + Math.abs(x - v[i]), 0);
      if (diff < 1e-6) {
 break;
}

      v = newV;
    }

    // 计算特征值
    let eigenvalue = 0;
    for (let i = 0; i < d; i++) {
      let sum = 0;
      for (let j = 0; j < d; j++) {
        sum += covariance[i][j] * v[j];
      }
      eigenvalue += sum * v[i];
    }

    eigenvalues.push(eigenvalue);
    eigenvectors.push(v);

    // 从协方差矩阵中减去当前特征值对应的分量（幂迭代法去相关）
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        covariance[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  // 4. 投影数据到新的低维空间
  const projectedData = centeredData.map(row => {
    const newRow = new Array(dimensions).fill(0);
    for (let k = 0; k < dimensions; k++) {
      for (let j = 0; j < d; j++) {
        newRow[k] += row[j] * eigenvectors[k][j];
      }
    }
    return newRow;
  });

  return projectedData;
}

// t-SNE降维实现（简化版，适用于中小规模数据）
function tsneReduction(data: number[][], dimensions = 2, perplexity = 30): number[][] {
  // 确保数据是二维数组
  if (!Array.isArray(data) || !Array.isArray(data[0])) {
    throw new Error('Data must be a 2D array for t-SNE reduction');
  }

  const n = data.length;
  const d = data[0].length;
  const targetDims = dimensions;

  // 1. 计算高维空间中的相似度矩阵
  const distances = new Array(n).fill(0).map(() => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dist = 0;
      for (let k = 0; k < d; k++) {
        const diff = data[i][k] - data[j][k];
        dist += diff * diff;
      }
      distances[i][j] = Math.sqrt(dist);
      distances[j][i] = distances[i][j];
    }
  }

  // 2. 计算条件概率 P(j|i)
  const perplexityTolerance = 1e-5;
  const maxIterations = 100;
  const P = new Array(n).fill(0).map(() => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    // 使用二分搜索找到合适的sigma
    let sigmaMin = 0;
    let sigmaMax = Infinity;
    let sigma = 1.0;

    for (let iter = 0; iter < maxIterations; iter++) {
      // 计算当前sigma下的条件概率
      const prob = new Array(n).fill(0);
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          prob[j] = Math.exp(-distances[i][j] * distances[i][j] / (2 * sigma * sigma));
        }
      }

      const sumProb = prob.reduce((a, b) => a + b, 0);
      if (sumProb === 0) {
        sigma *= 2;
        continue;
      }

      const normalizedProb = prob.map(p => p / sumProb);

      // 计算困惑度
      let H = 0;
      for (const p of normalizedProb) {
        if (p > 0) {
          H -= p * Math.log2(p);
        }
      }
      const currentPerplexity = 2 ** H;

      // 调整sigma
      if (currentPerplexity > perplexity + perplexityTolerance) {
        sigmaMax = sigma;
        sigma = (sigmaMin + sigmaMax) / 2;
      } else if (currentPerplexity < perplexity - perplexityTolerance) {
        sigmaMin = sigma;
        sigma = sigmaMax === Infinity ? sigma * 2 : (sigmaMin + sigmaMax) / 2;
      } else {
        break;
      }
    }

    // 保存条件概率
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const prob = Math.exp(-distances[i][j] * distances[i][j] / (2 * sigma * sigma));
        P[i][j] = prob;
      }
    }

    // 归一化
    const sumP = P[i].reduce((a, b) => a + b, 0);
    for (let j = 0; j < n; j++) {
      P[i][j] /= sumP;
    }
  }

  // 3. 初始化低维空间中的点
  const Y = new Array(n).fill(0).map(() =>
    new Array(targetDims).fill(0).map(() => (Math.random() - 0.5) * 0.0001),
  );

  // 4. 迭代优化
  const maxIter = 1000;
  const learningRate = 100;
  const momentum = 0.8;
  const YPrev = new Array(n).fill(0).map(() => new Array(targetDims).fill(0));

  for (let iter = 0; iter < maxIter; iter++) {
    // 计算低维空间中的相似度矩阵 Q
    const Q = new Array(n).fill(0).map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dist = 0;
        for (let k = 0; k < targetDims; k++) {
          const diff = Y[i][k] - Y[j][k];
          dist += diff * diff;
        }
        const q = 1 / (1 + dist);
        Q[i][j] = q;
        Q[j][i] = q;
      }
    }

    // 归一化 Q
    const sumQ = Q.reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0) / 2;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Q[i][j] /= sumQ;
      }
    }

    // 计算梯度
    const grad = new Array(n).fill(0).map(() => new Array(targetDims).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const factor = (P[i][j] - Q[i][j]) * (1 + Math.sqrt(Q[i][j]));
          for (let k = 0; k < targetDims; k++) {
            grad[i][k] += factor * (Y[i][k] - Y[j][k]);
          }
        }
      }
    }

    // 更新低维空间中的点
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < targetDims; k++) {
        const delta = learningRate * grad[i][k] + momentum * (Y[i][k] - YPrev[i][k]);
        YPrev[i][k] = Y[i][k];
        Y[i][k] -= delta;
      }
    }

    // 每100次迭代降低学习率
    if (iter % 100 === 0 && iter > 0) {
      learningRate *= 0.9;
    }
  }

  return Y;
}

// K-means聚类实现
function clustering(data: number[][], k = 5): {
  clusters: number[];
  centroids: number[][];
  clusterStats: { [key: number]: { size: number; mean: number[]; stdDev: number[] } };
} {
  // 确保数据是二维数组
  if (!Array.isArray(data) || !Array.isArray(data[0])) {
    throw new Error('Data must be a 2D array for clustering');
  }

  const n = data.length;
  const d = data[0].length;

  // 1. 初始化聚类中心
  const centroids: number[][] = [];
  const usedIndices = new Set<number>();

  // 使用K-means++初始化
  for (let i = 0; i < k; i++) {
    if (i === 0) {
      // 随机选择第一个中心
      const idx = Math.floor(Math.random() * n);
      centroids.push([...data[idx]]);
      usedIndices.add(idx);
    } else {
      // 基于距离选择下一个中心
      const distances = new Array(n).fill(0);
      for (let j = 0; j < n; j++) {
        if (!usedIndices.has(j)) {
          let minDist = Infinity;
          for (const centroid of centroids) {
            let dist = 0;
            for (let dim = 0; dim < d; dim++) {
              const diff = data[j][dim] - centroid[dim];
              dist += diff * diff;
            }
            if (dist < minDist) {
              minDist = dist;
            }
          }
          distances[j] = minDist;
        }
      }

      // 基于距离概率选择
      const sumDist = distances.reduce((a, b) => a + b, 0);
      let rand = Math.random() * sumDist;
      let chosenIdx = 0;
      while (rand > 0) {
        rand -= distances[chosenIdx];
        if (rand <= 0) {
 break;
}
        chosenIdx++;
      }

      centroids.push([...data[chosenIdx]]);
      usedIndices.add(chosenIdx);
    }
  }

  let clusters: number[] = [];
  let converged = false;

  // 2. 迭代更新聚类中心
  for (let iter = 0; iter < 100 && !converged; iter++) {
    // 分配每个点到最近的聚类中心
    clusters = data.map(point => {
      let minDist = Infinity;
      let clusterId = 0;

      for (let c = 0; c < k; c++) {
        let dist = 0;
        for (let dim = 0; dim < d; dim++) {
          const diff = point[dim] - centroids[c][dim];
          dist += diff * diff;
        }
        if (dist < minDist) {
          minDist = dist;
          clusterId = c;
        }
      }

      return clusterId;
    });

    // 更新聚类中心
    const newCentroids = new Array(k).fill(0).map(() => new Array(d).fill(0));
    const clusterSizes = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const clusterId = clusters[i];
      clusterSizes[clusterId]++;

      for (let dim = 0; dim < d; dim++) {
        newCentroids[clusterId][dim] += data[i][dim];
      }
    }

    // 计算平均值
    for (let c = 0; c < k; c++) {
      if (clusterSizes[c] > 0) {
        for (let dim = 0; dim < d; dim++) {
          newCentroids[c][dim] /= clusterSizes[c];
        }
      }
    }

    // 检查是否收敛
    converged = true;
    for (let c = 0; c < k; c++) {
      for (let dim = 0; dim < d; dim++) {
        if (Math.abs(newCentroids[c][dim] - centroids[c][dim]) > 1e-6) {
          converged = false;
          break;
        }
      }
      if (!converged) {
 break;
}
    }

    centroids.splice(0, centroids.length, ...newCentroids);
  }

  // 3. 计算聚类统计信息
  const clusterStats: { [key: number]: { size: number; mean: number[]; stdDev: number[] } } = {};

  for (let c = 0; c < k; c++) {
    const clusterPoints = data.filter((_, i) => clusters[i] === c);
    const size = clusterPoints.length;

    if (size === 0) {
 continue;
}

    // 计算均值
    const mean = new Array(d).fill(0);
    for (const point of clusterPoints) {
      for (let dim = 0; dim < d; dim++) {
        mean[dim] += point[dim] / size;
      }
    }

    // 计算标准差
    const stdDev = new Array(d).fill(0);
    for (const point of clusterPoints) {
      for (let dim = 0; dim < d; dim++) {
        const diff = point[dim] - mean[dim];
        stdDev[dim] += diff * diff / size;
      }
    }

    for (let dim = 0; dim < d; dim++) {
      stdDev[dim] = Math.sqrt(stdDev[dim]);
    }

    clusterStats[c] = { size, mean, stdDev };
  }

  return { clusters, centroids, clusterStats };
}

// 相关性分析实现
function correlationAnalysis(data: number[][]): {
  correlationMatrix: number[][];
  significantCorrelations: { i: number; j: number; value: number }[];
} {
  // 确保数据是二维数组
  if (!Array.isArray(data) || !Array.isArray(data[0])) {
    throw new Error('Data must be a 2D array for correlation analysis');
  }

  const n = data.length;
  const d = data[0].length;

  // 1. 计算均值和标准差
  const means = new Array(d).fill(0);
  const stdDevs = new Array(d).fill(0);

  for (let j = 0; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += data[i][j];
    }
    means[j] = sum / n;

    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const diff = data[i][j] - means[j];
      sumSq += diff * diff;
    }
    stdDevs[j] = Math.sqrt(sumSq / (n - 1));
  }

  // 2. 计算相关系数矩阵
  const correlationMatrix = new Array(d).fill(0).map(() => new Array(d).fill(0));

  for (let i = 0; i < d; i++) {
    correlationMatrix[i][i] = 1.0; // 自身相关系数为1

    for (let j = i + 1; j < d; j++) {
      let covariance = 0;
      for (let k = 0; k < n; k++) {
        const diffI = data[k][i] - means[i];
        const diffJ = data[k][j] - means[j];
        covariance += diffI * diffJ;
      }

      covariance /= (n - 1);
      const correlation = stdDevs[i] * stdDevs[j] !== 0 ? covariance / (stdDevs[i] * stdDevs[j]) : 0;

      correlationMatrix[i][j] = correlation;
      correlationMatrix[j][i] = correlation;
    }
  }

  // 3. 找出显著相关性（绝对值大于0.7）
  const significantCorrelations: { i: number; j: number; value: number }[] = [];

  for (let i = 0; i < d; i++) {
    for (let j = i + 1; j < d; j++) {
      const corr = correlationMatrix[i][j];
      if (Math.abs(corr) > 0.7) {
        significantCorrelations.push({ i, j, value: corr });
      }
    }
  }

  return { correlationMatrix, significantCorrelations };
}

// 更新消息处理逻辑
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const { id, type, data } = event.data;
  let result: any;
  let error: string | undefined;

  try {
    switch (type) {
      case 'normalize':
        result = normalize(data);
        break;
      case 'standardize':
        result = standardize(data);
        break;
      case 'clip':
        result = clip(data.data, data.precision || 3);
        break;
      case 'similarity':
        result = cosineSimilarity(data.vec1, data.vec2);
        break;
      case 'gradient_attribution':
        result = gradientAttribution(data.attentionWeights, data.gradients);
        break;
      case 'weight_statistics':
        result = weightStatistics(data);
        break;
      case 'transfer_matrix_processing':
        result = transferMatrixProcessing(data.matrix, data.operation);
        break;
      case 'pca_reduction':
        result = pcaReduction(data.data, data.dimensions || 2);
        break;
      case 'tsne_reduction':
        result = tsneReduction(data.data, data.dimensions || 2, data.perplexity || 30);
        break;
      case 'clustering':
        result = clustering(data.data, data.k || 5);
        break;
      case 'correlation_analysis':
        result = correlationAnalysis(data.data);
        break;
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (e) {
    error = e instanceof Error ? e.message : 'Unknown error';
  }

  // 发送响应
  const response: WorkerResponse = {
    id,
    type,
    result,
    error,
  };

  self.postMessage(response);
};

// 暴露模块（用于TypeScript）
export {};

import { Sprout, Cpu, Home, Shield, Building, Car, Truck, Lightbulb } from 'lucide-react';

export interface Industry {
  value: string;
  label: string;
  icon: React.ElementType;
  description: string;
}

export const industries: Industry[] = [
  {
    value: 'agriculture',
    label: '农业',
    icon: Sprout,
    description: '农业生产与管理',
  },
  {
    value: 'industry',
    label: '工业',
    icon: Cpu,
    description: '工业制造与自动化',
  },
  {
    value: 'home',
    label: '家庭',
    icon: Home,
    description: '智能家居与生活',
  },
  {
    value: 'healthcare',
    label: '医疗',
    icon: Shield,
    description: '医疗健康服务',
  },
  {
    value: 'commercial',
    label: '商业',
    icon: Building,
    description: '商业运营与管理',
  },
  {
    value: 'automotive',
    label: '汽车',
    icon: Car,
    description: '汽车行业解决方案',
  },
  {
    value: 'logistics',
    label: '物流',
    icon: Truck,
    description: '物流与供应链管理',
  },
  {
    value: 'energy',
    label: '能源',
    icon: Lightbulb,
    description: '能源管理与优化',
  },
];

export const getIndustryByValue = (value: string): Industry | undefined => industries.find(industry => industry.value === value);


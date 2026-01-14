import React from 'react';
import { AIFFactoryLogo } from '@/components/AIFFactoryLogo';

const LogoTest: React.FC = () => (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-gray-900">
      <div className="flex flex-col items-center justify-center space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-4 text-white">AI-Factory Logo Test</h1>
          <p className="text-gray-300">Logo should display correctly in dark theme</p>
        </div>

        <div className="flex items-center space-x-4 p-6 bg-gray-800 rounded-lg shadow-lg">
          <AIFFactoryLogo size={60} color="#0099ff" />
          <span className="text-2xl font-bold text-white">AI-Factory</span>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col items-center p-4 bg-gray-800 rounded-lg shadow">
            <AIFFactoryLogo size={40} color="#0099ff" />
            <p className="mt-2 text-gray-300">Size: 40px</p>
          </div>
          <div className="flex flex-col items-center p-4 bg-gray-800 rounded-lg shadow">
            <AIFFactoryLogo size={80} color="#0099ff" />
            <p className="mt-2 text-gray-300">Size: 80px</p>
          </div>
        </div>

        <div className="flex flex-col items-center p-6 bg-gray-800 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4 text-white">Horizontal Layout</h2>
          <div className="flex items-center space-x-4">
            <AIFFactoryLogo size={50} color="#0099ff" />
            <span className="text-xl font-bold text-white">AI-Factory</span>
          </div>
        </div>

        {/* 新增：测试带文字的Logo */}
        <div className="flex flex-col items-center p-6 bg-gray-800 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4 text-white">Logo with Text</h2>
          <AIFFactoryLogo size={60} color="#0099ff" showText={true} />
        </div>
      </div>
    </div>
  );

export default LogoTest;

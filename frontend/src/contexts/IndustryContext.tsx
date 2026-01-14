import type { ReactNode } from 'react';
import React, { createContext, useContext, useState } from 'react';
import type { Industry } from '@/config/industries';
import { industries } from '@/config/industries';

interface IndustryContextType {
  currentIndustry: string;
  setCurrentIndustry: (industry: string) => void;
  industryList: Industry[];
}

const IndustryContext = createContext<IndustryContextType | undefined>(undefined);

export const IndustryProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentIndustry, setCurrentIndustry] = useState<string>('agriculture');

  return (
    <IndustryContext.Provider
      value={{
        currentIndustry,
        setCurrentIndustry,
        industryList: industries,
      }}
    >
      {children}
    </IndustryContext.Provider>
  );
};

export const useIndustry = (): IndustryContextType => {
  const context = useContext(IndustryContext);
  if (context === undefined) {
    throw new Error('useIndustry must be used within an IndustryProvider');
  }
  return context;
};

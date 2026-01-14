import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="inline-flex items-center justify-center rounded-md p-2 hover:bg-muted text-muted-foreground transition-colors duration-200"
      aria-label={`切换到${theme === 'light' ? '深色' : '浅色'}主题`}
    >
      {theme === 'light' ?
(
        <Moon className="h-5 w-5" />
      ) :
(
        <Sun className="h-5 w-5" />
      )}
    </button>
  );
};

export default ThemeToggle;

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface BentoCardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  icon?: React.ComponentType<{ className?: string }>;
  description?: string;
}

export const BentoCard: React.FC<BentoCardProps> = ({
  children,
  className,
  title,
  icon: Icon,
  description,
}) => (
    <motion.div
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={cn(
        'glass-card rounded-2xl p-6 relative overflow-hidden group transition-all duration-300',
        className,
      )}
    >
      {/* Decorative background element */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-cyber-cyan/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-bl-full" />

      {(title || Icon) && (
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {Icon && (
              <div className="p-2 rounded-lg bg-cyber-cyan/5 border border-cyber-cyan/20 group-hover:neon-glow-cyan transition-all duration-300">
                <Icon className="w-5 h-5 text-cyber-cyan" />
              </div>
            )}
            <div>
              {title && <h3 className="font-bold text-lg tracking-tight text-white/90">{title}</h3>}
              {description && <p className="text-xs text-gray-500 uppercase tracking-widest">{description}</p>}
            </div>
          </div>
        </div>
      )}

      <div className="relative z-10">
        {children}
      </div>
    </motion.div>
  );

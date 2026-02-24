'use client';

import { cn } from '../lib/utils';

export interface SkeletonProps {
  variant?: 'text' | 'circle' | 'rect';
  width?: string | number;
  height?: string | number;
  className?: string;
}

export function Skeleton({ variant = 'rect', width, height, className }: SkeletonProps) {
  const baseStyles = 'animate-pulse bg-white/10';

  const variants = {
    text: 'rounded',
    circle: 'rounded-full',
    rect: 'rounded-lg',
  };

  return (
    <div
      className={cn(baseStyles, variants[variant], className)}
      style={{
        width: width || (variant === 'circle' ? 40 : '100%'),
        height: height || (variant === 'text' ? 16 : variant === 'circle' ? 40 : 100),
      }}
    />
  );
}
